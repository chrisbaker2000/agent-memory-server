import json
import os
import re
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import ulid
from docket import Timeout
from tenacity.asyncio import AsyncRetrying
from tenacity.stop import stop_after_attempt

# Lazy-import transformers in get_ner_model to avoid heavy deps at startup
from agent_memory_server.config import settings
from agent_memory_server.filters import DiscreteMemoryExtracted, MemoryType
from agent_memory_server.llm import LLMClient
from agent_memory_server.logging import get_logger
from agent_memory_server.models import VISIBILITY_RANK, MemoryRecord, MemoryTypeEnum


if TYPE_CHECKING:
    from bertopic import BERTopic


logger = get_logger(__name__)

# Set tokenizer parallelism environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Valid epistemic `kind` values (MEMORY-MODEL.md / LAB-395). Keep in lockstep with
# the MemoryRecord.kind Literal in models.py.
VALID_MEMORY_KINDS = frozenset(
    {"fact", "event", "preference", "opinion", "belief", "summary"}
)


# Strategies whose epistemic `kind` is invariant get it stamped deterministically
# rather than relying on LLM prompt compliance (the discrete strategy, by contrast,
# emits a per-memory kind that varies, so it has no default here). LAB-397.
_STRATEGY_DEFAULT_KIND = {"summary": "summary", "preferences": "preference"}


def default_kind_for_strategy(strategy_name: str | None) -> str | None:
    """Return the invariant `kind` for a strategy whose kind is fixed, else None."""
    return _STRATEGY_DEFAULT_KIND.get(strategy_name or "")


def coerce_extracted_kind(value: object) -> str | None:
    """Coerce an LLM-emitted `kind` to a valid MemoryKind, else None.

    An off-vocabulary, missing, wrong-type, OR unhashable `kind` becomes None —
    which the read path treats as ``fact`` — so a stray extraction value can never
    raise on MemoryRecord construction (the Literal would reject it) nor silently
    mislabel. The ``isinstance(value, str)`` guard is load-bearing: a bare
    ``value in VALID_MEMORY_KINDS`` raises TypeError if the LLM emits an unhashable
    ``kind`` (e.g. a list/dict like ``["fact"]``). The value is normalized
    (strip + lowercase) first, so a capitalized/padded emission like ``"Opinion"``
    or ``"FACT "`` maps to its canonical form rather than being dropped to None.
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized if normalized in VALID_MEMORY_KINDS else None


# Valid LLM-emitted memory types for EXTRACTED memories. MemoryTypeEnum also
# includes "message", but that is reserved for raw conversation-message records —
# an extracted/derived fact must never be typed "message" (the session-thread path
# also stamps session_id, so a "message"-typed extracted fact would pollute
# message-only reconstruction/search paths). So "message" is deliberately excluded
# here and coerces to the call-site default, like any other off-enum value.
VALID_MEMORY_TYPES = frozenset(
    {MemoryTypeEnum.EPISODIC.value, MemoryTypeEnum.SEMANTIC.value}
)

# The established default for discrete-extracted memories (matches the historical
# `new_memory.get("type", "episodic")` at the construction site below).
_DEFAULT_MEMORY_TYPE = "episodic"


def coerce_memory_type(value: object, default: str = _DEFAULT_MEMORY_TYPE) -> str:
    """Coerce an LLM-emitted ``memory_type`` to a valid EXTRACTED memory type.

    Returns ``value`` only if it is ``"episodic"`` or ``"semantic"`` (the types an
    extraction can legitimately produce — NOT ``"message"``, which is reserved for
    raw conversation records, see VALID_MEMORY_TYPES). An off-enum (e.g. the live
    ``"epistemic"`` from LAB-487), ``"message"``, missing, wrong-type, OR unhashable
    ``memory_type`` becomes ``default`` instead of raising a pydantic
    ``ValidationError`` on MemoryRecord construction — which, in the batched
    extraction below, would abort EVERY record in the batch, not just the bad one.
    This mirrors :func:`coerce_extracted_kind` for the sibling ``kind`` field; the
    ``isinstance(value, str)`` guard is load-bearing (a bare ``in`` membership test
    raises ``TypeError`` on an unhashable emission like ``["semantic"]``). The value
    is normalized (strip + lowercase) so ``"Episodic"`` / ``"SEMANTIC "`` map to
    canonical form. Unlike ``coerce_extracted_kind`` (None → read as ``fact``),
    ``memory_type`` has no None sentinel, so this always returns a valid type.
    """
    if not isinstance(value, str):
        return default
    normalized = value.strip().lower()
    return normalized if normalized in VALID_MEMORY_TYPES else default


# ============================================================================
# Vocabulary loading — path configured via settings.vocabulary_path.
# Falls back to inline defaults if the config file is missing (e.g. in tests).
# ============================================================================


def _load_vocabulary() -> dict:
    """Load vocabulary from shared config. Falls back to minimal inline defaults.

    Supports two vocabulary file formats:
    1. Flat format: { "controlled_topics": [...], "topic_map": {...} }
    2. Structured format: { "topics": { "name": { "synonyms": [...] } } }
       (used by Finley's gateway-side vocabulary system)

    Format 2 is automatically converted to format 1 for compatibility.
    """
    vocab_path = os.path.expanduser(settings.vocabulary_path)
    try:
        with open(vocab_path) as f:
            vocab = json.load(f)

            # Auto-convert structured format → flat format
            if (
                "topics" in vocab
                and isinstance(vocab["topics"], dict)
                and "controlled_topics" not in vocab
            ):
                topics_dict = vocab["topics"]
                controlled = list(topics_dict.keys())
                topic_map: dict[str, str] = {}
                for topic_name, topic_def in topics_dict.items():
                    if isinstance(topic_def, dict):
                        for synonym in topic_def.get("synonyms", []):
                            syn_lower = synonym.lower().strip()
                            if syn_lower and syn_lower != topic_name:
                                topic_map[syn_lower] = topic_name
                vocab["controlled_topics"] = controlled
                vocab["topic_map"] = topic_map
                logger.info(
                    "Loaded vocabulary from %s (structured format): %d topics, %d synonyms",
                    vocab_path,
                    len(controlled),
                    len(topic_map),
                )
            else:
                logger.info(
                    "Loaded vocabulary from %s: %d topics, %d mappings",
                    vocab_path,
                    len(vocab.get("controlled_topics", [])),
                    len(vocab.get("topic_map", {})),
                )
            return vocab
    except FileNotFoundError:
        logger.warning(
            "Vocabulary config not found at %s, using inline defaults", vocab_path
        )
        return {}
    except json.JSONDecodeError as e:
        logger.error(
            "Invalid JSON in vocabulary config %s: %s, using inline defaults",
            vocab_path,
            e,
        )
        return {}


_vocab = _load_vocabulary()


# ============================================================================
# Family registry loading — resolves source_user IDs to display names.
# Used by extraction strategies to replace "User" with real names in prompts.
# ============================================================================


def _load_family_registry() -> tuple[dict[str, str], dict[str, str]]:
    """Load family registry and build two lookup maps.

    Returns:
        Tuple of:
        - name_map: source_user → full display name (e.g. {"chris": "Chris Baker"})
        - identity_map: platform_id → source_user (e.g. {"773316001147256832": "chris"})
    Falls back to empty dicts if the file is missing (e.g. in tests).
    """
    family_path = os.path.expanduser(settings.family_json_path)
    try:
        with open(family_path) as f:
            data = json.load(f)
            users = data.get("users", {})
            name_map: dict[str, str] = {}
            identity_map: dict[str, str] = {}
            for user_id, info in users.items():
                display_name = info.get("displayName", user_id.title())
                # Construct full name — default last name "Baker" matches
                # session-memory-bridge convention for the Baker household
                last_name = info.get("lastName", "Baker")
                full_name = f"{display_name} {last_name}"
                name_map[user_id] = full_name
                # Build reverse identity map: platform ID → user_id
                identities = info.get("identities", {})
                for platform_id in identities.values():
                    if platform_id and isinstance(platform_id, str):
                        # Normalize: strip + prefix for phone numbers, lowercase
                        normalized = platform_id.lower().lstrip("+")
                        identity_map[normalized] = user_id
                        identity_map[platform_id.lower()] = user_id
            logger.info(
                "Loaded family registry from %s: %d users, %d identities",
                family_path,
                len(name_map),
                len(identity_map),
            )
            return name_map, identity_map
    except FileNotFoundError:
        logger.info(
            "Family registry not found at %s, will use source_user IDs as-is",
            family_path,
        )
        return {}, {}
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(
            "Error loading family registry from %s: %s",
            family_path,
            e,
        )
        return {}, {}


_family_names, _family_identities = _load_family_registry()


def resolve_user_display_name(source_user: str | None) -> str:
    """Resolve a source_user ID to a human-readable display name.

    Lookup order:
    1. Family registry (e.g. "chris" → "Chris Baker")
    2. Title-cased source_user (e.g. "chris" → "Chris")
    3. Fallback to "User" if source_user is None/empty

    Args:
        source_user: The source_user ID (e.g. "chris", "lindsey", "system")

    Returns:
        Display name string suitable for use in extraction prompts.
    """
    if not source_user or source_user == "system":
        return "User"

    # Check family registry first
    if source_user in _family_names:
        return _family_names[source_user]

    # Fallback: title-case the ID
    return source_user.title()


def resolve_user_from_session_id(session_id: str | None) -> str | None:
    """Try to resolve a source_user ID from a session key by matching peer IDs.

    Session keys follow the pattern: agent:main:channel:type:peerId
    (e.g., "agent:main:discord:direct:773316001147256832").

    Looks up the peer ID against family.json identities to find the user.

    Args:
        session_id: The session key string.

    Returns:
        The source_user ID (e.g. "chris") if found, None otherwise.
    """
    if not session_id:
        return None

    parts = session_id.lower().split(":")
    # Look for "direct" or "dm" scope — these indicate a 1:1 conversation
    for scope in ("direct", "dm"):
        try:
            idx = parts.index(scope)
            if idx + 1 < len(parts):
                peer_id = parts[idx + 1]
                # Look up in identity map (handles normalized + raw forms)
                if peer_id in _family_identities:
                    return _family_identities[peer_id]
                # Also try without + prefix for phone numbers
                stripped = peer_id.lstrip("+")
                if stripped in _family_identities:
                    return _family_identities[stripped]
        except ValueError:
            continue

    # Fallback: channel/group sessions (e.g., "agent:main:discord:channel:123456")
    # These don't have a user peer ID. Default to "chris" since he is the primary
    # user in channel conversations. This prevents empty source_user and "[User]"
    # labels in extraction prompts.
    if "channel" in parts:
        return "chris"

    # Fallback: local/main sessions (e.g., "agent:main:main") and system sessions
    # (e.g., "agent:main:hook:email:"). These have no peer ID or channel scope.
    # Default to "chris" — the primary (and only direct) user of the local gateway.
    if "main" in parts or "hook" in parts:
        return "chris"

    return None


# Entity quality constants — loaded from shared config
ENTITY_STOP_WORDS: set[str] = set(
    _vocab.get(
        "entity_stop_words",
        [
            "the",
            "this",
            "that",
            "a",
            "an",
            "it",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "must",
            "he",
            "she",
            "we",
            "they",
            "i",
            "you",
            "me",
            "him",
            "her",
            "us",
            "them",
            "my",
            "your",
            "his",
            "its",
            "our",
            "their",
            "what",
            "which",
            "who",
            "whom",
            "not",
            "no",
            "yes",
            "all",
            "each",
            "every",
            "both",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "also",
            "but",
            "or",
            "and",
            "if",
            "then",
            "so",
            "for",
            "with",
            "from",
            "to",
            "of",
            "on",
            "in",
            "at",
            "by",
            "up",
            "out",
            "off",
            "user",
            "assistant",
            "system",
            "none",
            "null",
            "true",
            "false",
            "ok",
            "okay",
        ],
    )
)

_limits = _vocab.get("limits", {})
MAX_ENTITY_COUNT = _limits.get("max_entity_count", 30)

# Controlled topic vocabulary — loaded from shared config.
# Stored as a set for O(1) lookup, with a sorted list for deterministic
# iteration in substring matching (set iteration order is non-deterministic).
CONTROLLED_TOPICS: set[str] = set(
    _vocab.get(
        "controlled_topics",
        [
            "family",
            "health",
            "education",
            "heritage",
            "home",
            "food",
            "travel",
            "entertainment",
            "sports",
            "media",
            "collecting",
            "work",
            "finances",
            "openclaw",
            "infrastructure",
            "analytics",
            "communication",
            "documents",
            "security",
        ],
    )
)
# Sort longest-first so "infrastructure" matches before "infra" substring,
# and deterministic across Python restarts (sets have non-deterministic order).
CONTROLLED_TOPICS_SORTED: list[str] = sorted(
    CONTROLLED_TOPICS, key=lambda t: (-len(t), t)
)

# Map common off-vocabulary terms to controlled topics (or None to drop)
_raw_topic_map = _vocab.get("topic_map", {})
# Filter out the _comment key if present
TOPIC_MAP: dict[str, str | None] = {
    k: v for k, v in _raw_topic_map.items() if not k.startswith("_")
}

# Global model instances
_topic_model: "BERTopic | None" = None
_ner_model: Any | None = None
_ner_tokenizer: Any | None = None


def get_topic_model() -> "BERTopic":
    """
    Get or initialize the BERTopic model.

    Returns:
        The BERTopic model instance
    """
    from bertopic import BERTopic

    global _topic_model
    if _topic_model is None:
        # TODO: Expose this as a config option
        _topic_model = BERTopic.load(
            settings.topic_model, embedding_model="all-MiniLM-L6-v2"
        )
    return _topic_model  # type: ignore


def get_ner_model() -> Any:
    """
    Get or initialize the NER model and tokenizer.

    Returns:
        The NER pipeline instance
    """
    global _ner_model, _ner_tokenizer
    if _ner_model is None:
        # Lazy import to avoid importing heavy ML frameworks at process startup
        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline as hf_pipeline,
            )
        except Exception as e:
            logger.warning(
                "Transformers not available or failed to import; NER disabled: %s", e
            )
            raise

        _ner_tokenizer = AutoTokenizer.from_pretrained(settings.ner_model)
        _ner_model = AutoModelForTokenClassification.from_pretrained(settings.ner_model)
        return hf_pipeline("ner", model=_ner_model, tokenizer=_ner_tokenizer)

    # If already initialized, import the lightweight symbol and return a new pipeline
    from transformers import pipeline as hf_pipeline  # type: ignore

    return hf_pipeline("ner", model=_ner_model, tokenizer=_ner_tokenizer)


def extract_entities_bert(text: str) -> list[str]:
    """
    Extract named entities from text using a BERT-based NER model.

    Requires PyTorch and transformers to be installed.

    Args:
        text: The text to extract entities from

    Returns:
        List of unique entity names
    """
    try:
        ner = get_ner_model()
        results = ner(text)

        # Group tokens by entity
        current_entity = []
        entities = []

        for result in results:
            if result["word"].startswith("##"):
                # This is a continuation of the previous entity
                current_entity.append(result["word"][2:])
            else:
                # This is a new entity
                if current_entity:
                    entities.append("".join(current_entity))
                current_entity = [result["word"]]

        # Add the last entity if exists
        if current_entity:
            entities.append("".join(current_entity))

        return list(set(entities))  # Remove duplicates

    except Exception as e:
        logger.error(f"Error extracting entities with BERT: {e}")
        return []


async def extract_entities_llm(text: str) -> list[str]:
    """
    Extract named entities from text using an LLM.

    Retries up to 3 times on JSON parse failure or empty results.
    Returns [] if all attempts are exhausted.

    Args:
        text: The text to extract entities from

    Returns:
        List of unique entity names (people, organizations, locations, etc.)
    """
    prompt = f"""Extract named entities (people, organizations, locations, products, etc.) from the following text.

Text:
{text}

Return a JSON object with an "entities" array containing the entity names as strings.
Example: {{"entities": ["John Smith", "Apple Inc.", "New York"]}}
"""
    entities: list[str] = []

    try:
        async for attempt in AsyncRetrying(stop=stop_after_attempt(3)):
            with attempt:
                response = await LLMClient.create_chat_completion(
                    model=settings.fast_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                # Let JSONDecodeError propagate so tenacity retries.
                # The old code caught it and returned [], preventing retries.
                parsed = json.loads(response.content)
                entities = parsed.get("entities", [])
                if not entities:
                    raise ValueError("LLM returned empty entities list")
    except Exception:
        # All retries exhausted — return whatever we have (likely [])
        logger.warning(
            "Entity extraction failed after 3 attempts for text: %s...", text[:80]
        )

    return list(set(entities))  # Remove duplicates


async def extract_topics_llm(
    text: str,
    num_topics: int | None = None,
) -> list[str]:
    """
    Extract topics from text using an LLM.

    Retries up to 3 times on JSON parse failure or empty results.
    Returns [] if all attempts are exhausted. Uses settings.fast_model.
    """
    _num_topics = num_topics if num_topics is not None else settings.top_k_topics

    prompt = f"""Extract the top {_num_topics} topics from the following text.

Text:
{text}

Return a JSON object with a "topics" array containing topic strings.
Example: {{"topics": ["machine learning", "data science", "python"]}}
"""
    topics: list[str] = []

    try:
        async for attempt in AsyncRetrying(stop=stop_after_attempt(3)):
            with attempt:
                response = await LLMClient.create_chat_completion(
                    model=settings.fast_model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                )
                # Let JSONDecodeError propagate so tenacity retries.
                parsed = json.loads(response.content)
                topics = parsed.get("topics", [])
                if not topics:
                    raise ValueError("LLM returned empty topics list")
                topics = topics[:_num_topics]
    except Exception:
        # All retries exhausted — return whatever we have (likely [])
        logger.warning(
            "Topic extraction failed after 3 attempts for text: %s...", text[:80]
        )

    return topics


def clean_entities(entities: list[str]) -> list[str]:
    """
    Quality-filter an entity list:
      0. Filter non-string values (LLM may return null/int/dict in entity lists)
      1. Strip whitespace and empty strings
      2. Remove single-word common English stop words
      3. Deduplicate variants (keep most specific form)
      4. Remove URLs, file paths, hex IDs, and other non-entity junk
      5. Cap at MAX_ENTITY_COUNT

    Returns a cleaned list preserving original order (most specific first).
    """
    if not entities:
        return []

    cleaned: list[str] = []
    seen_lower: set[str] = set()

    for entity in entities:
        if not isinstance(entity, str):
            continue
        entity = entity.strip().strip("\"'[]{}")
        if not entity:
            continue

        # Skip URLs
        if entity.startswith(("http://", "https://", "ftp://")):
            continue

        # Skip file paths (starting with / or ~/ or ./)
        if re.match(r"^[~/.]/", entity):
            continue

        # Skip hex IDs (>= 16 hex chars)
        if re.match(r"^[0-9A-Fa-f]{16,}$", entity):
            continue

        # Skip chmod-style permissions (e.g., "0600")
        if re.match(r"^0[0-7]{3}$", entity):
            continue

        # Strip @ and # prefixes
        if entity.startswith(("@", "#")):
            entity = entity[1:]
            if not entity:
                continue

        # Single-word stop word check (case-insensitive)
        if " " not in entity and entity.lower() in ENTITY_STOP_WORDS:
            continue

        # Deduplicate case-insensitively
        key = entity.lower()
        if key in seen_lower:
            continue
        seen_lower.add(key)

        cleaned.append(entity)

    # Deduplicate variants: if "photos" and "photo organization" both exist,
    # keep the more specific one ("photo organization")
    cleaned = _deduplicate_entity_variants(cleaned)

    # Cap at max
    if len(cleaned) > MAX_ENTITY_COUNT:
        cleaned = cleaned[:MAX_ENTITY_COUNT]

    return cleaned


def _deduplicate_entity_variants(entities: list[str]) -> list[str]:
    """
    Remove less-specific entity variants.
    E.g., if both "photo" and "photo organization" exist, drop "photo".
    If both "Baker" and "Chris Baker" exist, drop "Baker".
    Only removes single-word entities that are substrings of multi-word entities.
    """
    if len(entities) <= 1:
        return entities

    multi_word = [e for e in entities if " " in e]
    if not multi_word:
        return entities

    # Build a set of single-word entities that appear as part of a multi-word entity
    to_remove: set[str] = set()
    for entity in entities:
        if " " in entity:
            continue  # Only consider single-word entities for removal
        entity_lower = entity.lower()
        for mw in multi_word:
            # Check if this single word is a component of a multi-word entity
            mw_words = {w.lower() for w in mw.split()}
            if entity_lower in mw_words:
                to_remove.add(entity)
                break

    if not to_remove:
        return entities

    return [e for e in entities if e not in to_remove]


def enforce_topics(topics: list[str]) -> list[str]:
    """
    Enforce the controlled topic vocabulary on a list of topics.

    1. Lowercase and strip
    2. Direct match against CONTROLLED_TOPICS (O(1) set lookup)
    3. Map known synonyms via TOPIC_MAP
    4. Word-boundary match against CONTROLLED_TOPICS_SORTED (longest-first
       for deterministic resolution when a topic contains multiple
       controlled-topic substrings, e.g. "home_security")
    5. Deduplicate
    6. Drop unrecognized terms

    Step 4 uses word-boundary anchors (whitespace, underscore, hyphen, slash)
    rather than bare ``in`` to prevent short topics like "ai" from matching
    inside unrelated words like "email" or "maintain".

    The taxonomy size is dynamic (loaded from memory-vocabulary.json at
    startup). Matching iterates CONTROLLED_TOPICS_SORTED, not the set, to
    guarantee deterministic results across Python restarts.
    """
    if not topics:
        return []

    cleaned: list[str] = []
    seen: set[str] = set()

    for topic in topics:
        topic = topic.strip().lower()
        if not topic:
            continue

        # Drop long descriptions (>30 chars with spaces — likely sentences)
        if len(topic) > 30 and " " in topic:
            continue

        # Direct match to controlled vocabulary
        if topic in CONTROLLED_TOPICS:
            if topic not in seen:
                seen.add(topic)
                cleaned.append(topic)
            continue

        # Try mapping
        mapped = TOPIC_MAP.get(topic)
        if mapped is not None:
            if mapped not in seen:
                seen.add(mapped)
                cleaned.append(mapped)
            continue

        # Try word-boundary match against controlled topics (sorted for determinism).
        # Use regex \b to prevent "ai" matching inside "email" or "maintain".
        matched = False
        for ct in CONTROLLED_TOPICS_SORTED:
            if re.search(r"(?:^|[\s_\-/])" + re.escape(ct) + r"(?:$|[\s_\-/])", topic):
                if ct not in seen:
                    seen.add(ct)
                    cleaned.append(ct)
                matched = True
                break

        if not matched:
            # Try word-boundary match against topic map keys
            for map_key, map_val in TOPIC_MAP.items():
                if (
                    map_key
                    and map_val is not None
                    and re.search(
                        r"(?:^|[\s_\-/])" + re.escape(map_key) + r"(?:$|[\s_\-/])",
                        topic,
                    )
                ):
                    if map_val not in seen:
                        seen.add(map_val)
                        cleaned.append(map_val)
                    matched = True
                    break

        # If still no match, drop (not in controlled vocabulary)

    return cleaned


def extract_topics_bertopic(text: str, num_topics: int | None = None) -> list[str]:
    """
    Extract topics from text using the BERTopic model.

    TODO: Cache this output.

    Args:
        text: The text to extract topics from

    Returns:
        List of topic labels
    """
    # Get model instance
    model = get_topic_model()

    _num_topics = num_topics if num_topics is not None else settings.top_k_topics

    # Get topic indices and probabilities
    topic_indices, _ = model.transform([text])

    topics = []
    for i, topic_idx in enumerate(topic_indices):
        if _num_topics and i >= _num_topics:
            break
        # Convert possible numpy integer to Python int
        topic_idx_int = int(topic_idx)
        if topic_idx_int != -1:  # Skip outlier topic (-1)
            topic_info: list[tuple[str, float]] = model.get_topic(topic_idx_int)  # type: ignore
            if topic_info:
                topics.extend([info[0] for info in topic_info])

    return topics


async def handle_extraction(text: str) -> tuple[list[str], list[str]]:
    """
    Handle topic and entity extraction for a message.

    Args:
        text: The text to process

    Returns:
        Tuple of extracted topics and entities
    """
    # Extract topics if enabled
    topics: list[str] = []
    if settings.enable_topic_extraction:
        if settings.topic_model_source == "BERT":
            topics = extract_topics_bertopic(text)
        else:
            topics = await extract_topics_llm(text)

    # Extract entities if enabled
    entities: list[str] = []
    if settings.enable_ner:
        if settings.ner_model_source == "BERT":
            entities = extract_entities_bert(text)
        else:
            entities = await extract_entities_llm(text)

    # Quality filter: enforce controlled topic vocabulary
    topics = enforce_topics(topics)

    # Quality filter: clean entities (stop words, variants, cap)
    entities = clean_entities(entities)

    return topics, entities


def _resolve_parent_attribution(
    memories: list[MemoryRecord],
    source_user: str | None = None,
    source_channel: str | None = None,
    visibility: str | None = None,
) -> tuple[str | None, str | None, str]:
    """
    Resolve attribution fields from explicit parameters and parent memories.

    Explicit parameters take priority. If not provided, inherits from the first
    non-None value across parent memories. For visibility, uses the most
    restrictive value.

    Args:
        memories: Parent memory records to inherit from
        source_user: Explicit source_user override
        source_channel: Explicit source_channel override
        visibility: Explicit visibility override

    Returns:
        Tuple of (resolved_source_user, resolved_source_channel, resolved_visibility)
    """
    # Resolve source_user: explicit param > first non-None parent
    resolved_user = source_user
    if resolved_user is None:
        resolved_user = next(
            (m.source_user for m in memories if m.source_user is not None),
            None,
        )

    # Resolve source_channel: explicit param > first non-None parent
    resolved_channel = source_channel
    if resolved_channel is None:
        resolved_channel = next(
            (m.source_channel for m in memories if m.source_channel is not None),
            None,
        )

    # Resolve visibility: explicit param > most restrictive across parents
    if visibility is not None:
        resolved_visibility = visibility
    else:
        parent_visibilities = [m.visibility for m in memories]
        if parent_visibilities:
            resolved_visibility = max(
                parent_visibilities, key=lambda v: VISIBILITY_RANK.get(v, 0)
            )
        else:
            resolved_visibility = "everyone"

    return resolved_user, resolved_channel, resolved_visibility


async def extract_memories_with_strategy(
    memories: list[MemoryRecord] | None = None,
    deduplicate: bool = True,
    source_user: str | None = None,
    source_channel: str | None = None,
    visibility: str | None = None,
    timeout: Timeout = Timeout(timedelta(minutes=settings.llm_task_timeout_minutes)),
):
    """
    Extract memories using their configured strategies.

    This function replaces extract_discrete_memories for strategy-aware extraction.
    Each memory record contains its extraction strategy configuration.

    Args:
        memories: List of memory records to process, or None to search for unprocessed messages
        deduplicate: Whether to deduplicate extracted memories
        source_user: Attribution source user to propagate to child memories.
            If None, inherits from parent memory's source_user.
        source_channel: Attribution source channel to propagate to child memories.
            If None, inherits from parent memory's source_channel.
        visibility: Visibility scope to propagate to child memories.
            If None, uses most restrictive visibility from parent memories.
        timeout: Docket timeout for this task (defaults to llm_task_timeout_minutes from settings)
    """
    # Local imports to avoid circular dependencies:
    # long_term_memory imports from extraction, so we import locally here
    from agent_memory_server.long_term_memory import index_long_term_memories
    from agent_memory_server.memory_strategies import get_memory_strategy
    from agent_memory_server.memory_vector_db_factory import get_memory_vector_db

    db = await get_memory_vector_db()

    if not memories:
        # If no memories are provided, search for any messages in long-term memory
        # that haven't been processed for extraction using filter-only query
        # (no embedding required)
        memories = []
        offset = 0
        while True:
            search_result = await db.list_memories(
                memory_type=MemoryType(eq="message"),
                discrete_memory_extracted=DiscreteMemoryExtracted(eq="f"),
                limit=25,
                offset=offset,
            )

            logger.info(
                f"Found {len(search_result.memories)} memories to extract: {[m.id for m in search_result.memories]}"
            )

            memories += search_result.memories

            if len(search_result.memories) < 25:
                break

            offset += 25

    # Group memories by extraction strategy for batch processing
    strategy_groups = {}
    for memory in memories:
        if not memory or not memory.text:
            logger.info(f"Deleting memory with no text: {memory}")
            await db.delete_memories([memory.id])
            continue

        # JSON-serialize config for a hashable group key. The old approach
        # tuple(sorted(config.items())) crashes with TypeError when config
        # values contain nested dicts or lists (unhashable types).
        strategy_key = (
            memory.extraction_strategy,
            json.dumps(memory.extraction_strategy_config, sort_keys=True),
        )
        if strategy_key not in strategy_groups:
            strategy_groups[strategy_key] = []
        strategy_groups[strategy_key].append(memory)

    all_new_memories = []
    all_updated_memories = []

    # Process each strategy group
    for (strategy_name, config_json), strategy_memories in strategy_groups.items():
        logger.info(
            f"Processing {len(strategy_memories)} memories with strategy: {strategy_name}"
        )

        # Get strategy instance — deserialize the JSON config key back to dict
        config_dict = json.loads(config_json)
        try:
            strategy = get_memory_strategy(strategy_name, **config_dict)
        except ValueError as e:
            logger.error(f"Unknown strategy {strategy_name}: {e}")
            # Fall back to discrete strategy
            strategy = get_memory_strategy("discrete")

        # Process memories with this strategy
        for memory in strategy_memories:
            try:
                # Resolve the display name for the source user so the
                # extraction prompt uses a real name instead of "User".
                parent_source_user = source_user or memory.source_user
                resolved_name = resolve_user_display_name(parent_source_user)

                extracted_memories = await strategy.extract_memories(
                    memory.text, source_user_name=resolved_name
                )

                # Resolve attribution for this parent memory
                parent_user, parent_channel, parent_visibility = (
                    _resolve_parent_attribution(
                        [memory],
                        source_user=source_user,
                        source_channel=source_channel,
                        visibility=visibility,
                    )
                )

                # Tag each extracted dict with parent attribution for later use
                for em in extracted_memories:
                    em["_source_user"] = parent_user
                    em["_source_channel"] = parent_channel
                    em["_visibility"] = parent_visibility
                    em["_strategy"] = strategy_name

                all_new_memories.extend(extracted_memories)

                # Update the memory to mark it as processed
                updated_memory = memory.model_copy(
                    update={"discrete_memory_extracted": "t"}
                )
                all_updated_memories.append(updated_memory)

            except Exception as e:
                logger.error(
                    f"Error extracting memory {memory.id} with strategy {strategy_name}: {e}"
                )
                # Still mark as processed to avoid infinite retry
                updated_memory = memory.model_copy(
                    update={"discrete_memory_extracted": "t"}
                )
                all_updated_memories.append(updated_memory)

    # Update processed memories
    if all_updated_memories:
        await db.update_memories(all_updated_memories)

    # Index new extracted memories.
    # LAB-487: build each record under its OWN try/except. A single malformed
    # extracted dict (e.g. an off-enum memory_type, or a missing required "text")
    # must skip ONLY that record — the historical list comprehension let one
    # ValidationError abort the whole batch, silently dropping every other fact
    # extracted from the same conversation thread.
    if all_new_memories:
        long_term_memories: list[MemoryRecord] = []
        for new_memory in all_new_memories:
            try:
                long_term_memories.append(
                    MemoryRecord(
                        id=str(ulid.ULID()),
                        text=new_memory["text"],
                        memory_type=coerce_memory_type(new_memory.get("type")),
                        # F0 (LAB-395/LAB-397): a strategy with an INVARIANT kind
                        # (summary→summary, preferences→preference) is AUTHORITATIVE —
                        # it overrides whatever the LLM emitted (a summary is always a
                        # summary, even if the model labels it "fact"). Discrete has no
                        # invariant (default None), so it falls through to the coerced
                        # LLM kind; off-vocab/missing → None = read as 'fact'.
                        # Confidence is the model-paraphrase tier (distinct from an
                        # unscored first-hand memory_store write).
                        kind=default_kind_for_strategy(new_memory.get("_strategy"))
                        or coerce_extracted_kind(new_memory.get("kind")),
                        confidence=settings.extraction_confidence,
                        topics=enforce_topics(new_memory.get("topics", [])),
                        entities=clean_entities(new_memory.get("entities", [])),
                        discrete_memory_extracted="t",
                        extraction_strategy="discrete",  # These are already extracted
                        extraction_strategy_config={},
                        source_user=new_memory.get("_source_user"),
                        source_channel=new_memory.get("_source_channel"),
                        visibility=new_memory.get("_visibility", "everyone"),
                    )
                )
            except Exception as e:
                # coerce_memory_type / coerce_extracted_kind already neutralize the
                # known enum hazards; this guards any residual construction failure
                # (e.g. a missing "text") so the rest of the batch still persists.
                # new_memory may be non-dict debris, so resolve the preview
                # defensively (a bare `.get()` would raise inside this guard).
                text_preview = (
                    str(new_memory.get("text", ""))[:80]
                    if isinstance(new_memory, dict)
                    else str(new_memory)[:80]
                )
                logger.warning(
                    "Skipping malformed extracted memory (text=%r): %s",
                    text_preview,
                    e,
                )

        if long_term_memories:
            await index_long_term_memories(
                long_term_memories,
                deduplicate=deduplicate,
            )

import json
import logging
import numbers
import re
import time
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from docket import Timeout
from docket.dependencies import Perpetual
from redis.asyncio import Redis
from ulid import ULID

from agent_memory_server.config import settings
from agent_memory_server.dependencies import get_background_tasks
from agent_memory_server.extraction import (
    _resolve_parent_attribution,
    clean_entities,
    coerce_extracted_kind,
    enforce_topics,
    extract_memories_with_strategy,
    handle_extraction,
)
from agent_memory_server.filters import (
    CreatedAt,
    Entities,
    EventDate,
    Kind,
    LastAccessed,
    MemoryHash,
    MemoryType,
    MinConfidence,
    Namespace,
    SessionId,
    SourceChannel,
    SourceUser,
    StaleAfter,
    Topics,
    UserId,
    VisibilityFilter,
)
from agent_memory_server.llm import LLMClient, optimize_query_for_vector_search
from agent_memory_server.memory_vector_db_factory import get_memory_vector_db
from agent_memory_server.models import (
    VISIBILITY_RANK,
    ExtractedMemoryRecord,
    MemoryRecord,
    MemoryRecordResult,
    MemoryRecordResults,
    MemoryTypeEnum,
)
from agent_memory_server.telemetry import record_counter
from agent_memory_server.utils.content_security import apply_content_security
from agent_memory_server.utils.content_trust import (
    ReferenceProtectedError,
    TrustLevel,
    is_reference_protected_mutation,
)
from agent_memory_server.utils.egress_guard import (
    config_from_settings as egress_config_from_settings,
    record_and_check as egress_record_and_check,
)
from agent_memory_server.utils.keys import Keys
from agent_memory_server.utils.recency import (
    _days_between,
    generate_memory_hash,
    rerank_with_recency,
    update_memory_hash_if_text_changed,
)
from agent_memory_server.utils.redis import get_redis_conn
from agent_memory_server.utils.relevance import apply_relevance_gate


# Track pending extraction tasks to prevent garbage collection
# This is only used when running without Docket (asyncio mode)
_pending_extraction_tasks: set = set()


def _parse_extraction_response_with_fallback(content: str, logger) -> dict:
    """
    Parse JSON response with fallback mechanisms for malformed responses.

    Args:
        content: The JSON content to parse
        logger: Logger instance for error reporting

    Returns:
        Parsed JSON dictionary with 'memories' key

    Raises:
        json.JSONDecodeError: If all parsing attempts fail
    """
    # Try standard JSON parsing first
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Attempt to repair common JSON issues
        logger.warning(
            f"Initial JSON parsing failed, attempting repair on content: {content[:500]}..."
        )

        # Try to extract just the memories array if it exists.
        # IMPORTANT: greedy (.*) must come first — non-greedy (.*?) truncates
        # at the first ']' it finds, which breaks when memory text contains
        # literal brackets like "[1,2,3]".  Greedy captures through nested
        # brackets to the LAST ']', which is correct for the outer array.
        # If the greedy capture produces invalid JSON (extra trailing content),
        # we fall back to non-greedy as a last resort.
        for pattern in [
            r'"memories"\s*:\s*\[(.*)\]',  # greedy — handles nested brackets
            r'"memories"\s*:\s*\[(.*?)\]',  # non-greedy — last resort
        ]:
            memories_match = re.search(pattern, content, re.DOTALL)
            if memories_match:
                try:
                    # Try to reconstruct a valid JSON object
                    memories_json = '{"memories": [' + memories_match.group(1) + "]}"
                    extraction_result = json.loads(memories_json)
                    logger.info("Successfully repaired malformed JSON response")
                    return extraction_result
                except json.JSONDecodeError:
                    continue  # Try next pattern

        # All repair attempts failed
        logger.error("JSON repair attempt failed — no pattern matched or parsed")
        # `from None`: the terminal repair-failure is the real error; chaining the
        # last per-pattern JSONDecodeError (each a "try next pattern" miss) is noise.
        raise json.JSONDecodeError("All repair attempts failed", content, 0) from None


# Prompt for extracting memories from messages in working memory context
WORKING_MEMORY_EXTRACTION_PROMPT = """
You are a memory extraction assistant. Your job is to analyze conversation
messages and extract information that might be useful in future conversations.

Extract two types of memories from the following message:
1. EPISODIC: Experiences or events that have a time dimension.
   (They MUST have a time dimension to be "episodic.")
   Example: "User mentioned they visited Paris in August of 2025" or "User had trouble with the login process on 2025-01-15"

2. SEMANTIC: User preferences, facts, or general knowledge about the agent's
   environment that might be useful long-term.
   Example: "User prefers dark mode UI" or "User works as a data scientist"

For each memory, return a JSON object with the following fields:
- type: str -- The memory type, either "episodic" or "semantic"
- text: str -- The actual information to store
- topics: list[str] -- Relevant topics for this memory
- entities: list[str] -- Named entities mentioned
- event_date: str | null -- For episodic memories, the date/time when the event occurred (ISO 8601 format), null for semantic memories

IMPORTANT RULES:
1. Only extract information that might be genuinely useful for future interactions.
2. Do not extract procedural knowledge or instructions.
3. If given `user_id`, focus on user-specific information, preferences, and facts.
4. Return an empty list if no useful memories can be extracted.

Message: {message}

Return format:
{{
    "memories": [
        {{
            "type": "episodic",
            "text": "...",
            "topics": ["..."],
            "entities": ["..."],
            "event_date": "2024-01-15T14:30:00Z"
        }},
        {{
            "type": "semantic",
            "text": "...",
            "topics": ["..."],
            "entities": ["..."],
            "event_date": null
        }}
    ]
}}

Extracted memories:
"""


logger = logging.getLogger(__name__)

# Size guards — prevent mega-memory creation (added 2026-03-10)
MAX_MEMORY_INPUT_CHARS = 500  # Skip merging memories larger than this
MAX_MEMORY_OUTPUT_CHARS = 1000  # Cap merged output at this length
MAX_ENTITY_COUNT = 30  # Skip memories with more entities than this

# Debounce configuration for thread-aware extraction (trailing-edge)
# We use a "pending extraction" key to track when extraction should run
EXTRACTION_PENDING_KEY_PREFIX = "extraction_pending"
EXTRACTION_DEBOUNCE_KEY_PREFIX = "extraction_debounce"


async def should_extract_session_thread(session_id: str, redis: Redis) -> bool:
    """
    Check if extraction should proceed based on debounce.

    This checks if the post-extraction debounce key exists. After successful
    extraction, we set this key to prevent immediate re-extraction even if
    new messages arrive.

    Args:
        session_id: The session ID to check
        redis: Redis client

    Returns:
        True if extraction should proceed, False if debounced
    """
    debounce_key = f"{EXTRACTION_DEBOUNCE_KEY_PREFIX}:{session_id}"

    # Check if debounce key exists (set after successful extraction)
    exists = await redis.exists(debounce_key)
    if not exists:
        logger.info(f"Extraction allowed for session {session_id} (no debounce key)")
        return True

    remaining_ttl = await redis.ttl(debounce_key)
    logger.info(
        f"Skipping thread-aware extraction for session {session_id} (debounced, {remaining_ttl}s remaining)"
    )
    return False


async def set_extraction_debounce(session_id: str, redis: Redis) -> None:
    """
    Set the debounce key after successful extraction.

    This should be called after extraction completes successfully to prevent
    re-extraction for the debounce period.

    Args:
        session_id: The session ID to set debounce for
        redis: Redis client
    """
    debounce_key = f"{EXTRACTION_DEBOUNCE_KEY_PREFIX}:{session_id}"
    debounce_ttl = settings.extraction_debounce_seconds
    await redis.setex(debounce_key, debounce_ttl, "extracted")
    logger.info(f"Set extraction debounce for session {session_id} ({debounce_ttl}s)")


async def schedule_trailing_extraction(
    session_id: str,
    namespace: str | None,
    user_id: str | None,
    redis: Redis,
    source_user: str | None = None,
    source_channel: str | None = None,
    visibility: str | None = None,
) -> None:
    """
    Schedule a trailing-edge debounced extraction.

    This implements trailing-edge debounce:
    1. Store a "pending extraction" timestamp in Redis
    2. Schedule a Docket task to run after the debounce period
    3. Each new message resets the timer by updating the pending timestamp
    4. The scheduled task checks if it's still valid before running

    Args:
        session_id: The session ID to extract for
        namespace: Optional namespace
        user_id: Optional user ID
        redis: Redis client
        source_user: Attribution source user to propagate to extracted memories
        source_channel: Attribution source channel to propagate to extracted memories
        visibility: Visibility scope to propagate to extracted memories
    """
    from datetime import timedelta

    pending_key = f"{EXTRACTION_PENDING_KEY_PREFIX}:{session_id}"
    debounce_seconds = settings.extraction_debounce_seconds

    # Calculate when extraction should run (trailing edge)
    extraction_time = datetime.now(UTC) + timedelta(seconds=debounce_seconds)
    extraction_timestamp = extraction_time.isoformat()

    # Store the pending extraction timestamp
    # TTL is 2x debounce to ensure cleanup even if task fails
    await redis.setex(pending_key, debounce_seconds * 2, extraction_timestamp)
    logger.info(
        f"Scheduled trailing extraction for session {session_id} at {extraction_timestamp}"
    )

    # Schedule the Docket task if enabled
    if settings.use_docket:
        from docket import Docket

        try:
            async with Docket(
                name=settings.docket_name,
                url=settings.redis_url,
            ) as docket:
                # Schedule with a unique key per session
                # If there's already a pending task for this session, the new one
                # will check the timestamp and skip if superseded
                task_key = f"extraction:{session_id}:{extraction_timestamp}"
                await docket.add(
                    run_delayed_extraction,
                    when=extraction_time,
                    key=task_key,
                )(
                    session_id=session_id,
                    namespace=namespace,
                    user_id=user_id,
                    scheduled_timestamp=extraction_timestamp,
                    source_user=source_user,
                    source_channel=source_channel,
                    visibility=visibility,
                )
                logger.debug(f"Docket task scheduled with key {task_key}")
        except Exception as e:
            logger.error(f"Failed to schedule Docket extraction task: {e}")
            # Fall back to immediate execution on next request
    else:
        # For asyncio backend, use asyncio.create_task with delay
        import asyncio

        async def delayed_extraction():
            try:
                await asyncio.sleep(debounce_seconds)
                await run_delayed_extraction(
                    session_id=session_id,
                    namespace=namespace,
                    user_id=user_id,
                    scheduled_timestamp=extraction_timestamp,
                    source_user=source_user,
                    source_channel=source_channel,
                    visibility=visibility,
                )
            finally:
                # Remove task from tracking set when done
                _pending_extraction_tasks.discard(asyncio.current_task())

        # Track the task to prevent garbage collection
        task = asyncio.create_task(delayed_extraction())
        _pending_extraction_tasks.add(task)
        logger.debug(
            f"Scheduled asyncio extraction task for session {session_id} "
            f"(delay: {debounce_seconds}s)"
        )


async def run_delayed_extraction(
    session_id: str,
    namespace: str | None = None,
    user_id: str | None = None,
    scheduled_timestamp: str | None = None,
    source_user: str | None = None,
    source_channel: str | None = None,
    visibility: str | None = None,
) -> int:
    """
    Run the delayed extraction if this task is still valid.

    This is called by Docket after the debounce period. It checks if the
    scheduled_timestamp matches the current pending timestamp - if not,
    this task was superseded by a newer one and should skip.

    Args:
        session_id: The session ID to extract for
        namespace: Optional namespace
        user_id: Optional user ID
        scheduled_timestamp: The timestamp when this extraction was scheduled
        source_user: Attribution source user to propagate to extracted memories
        source_channel: Attribution source channel to propagate to extracted memories
        visibility: Visibility scope to propagate to extracted memories

    Returns:
        Number of memories extracted, or 0 if skipped
    """
    from agent_memory_server.utils.redis import get_redis_conn
    from agent_memory_server.working_memory import (
        get_working_memory,
        set_working_memory,
    )

    redis = await get_redis_conn()
    pending_key = f"{EXTRACTION_PENDING_KEY_PREFIX}:{session_id}"

    # Check if this task is still valid (not superseded by a newer schedule)
    current_pending = await redis.get(pending_key)
    if current_pending:
        current_pending = (
            current_pending.decode("utf-8")
            if isinstance(current_pending, bytes)
            else current_pending
        )

    if scheduled_timestamp and current_pending != scheduled_timestamp:
        logger.info(
            f"Skipping extraction for session {session_id} - superseded by newer schedule "
            f"(scheduled: {scheduled_timestamp}, current: {current_pending})"
        )
        return 0

    # Check if we're still within the post-extraction debounce
    if not await should_extract_session_thread(session_id, redis):
        return 0

    # Get working memory to extract from
    working_memory = await get_working_memory(
        session_id=session_id, namespace=namespace, user_id=user_id
    )

    if not working_memory or not working_memory.messages:
        logger.info(f"No working memory to extract for session {session_id}")
        await redis.delete(pending_key)
        return 0

    # Check for unextracted messages
    unextracted_messages = [
        msg for msg in working_memory.messages if msg.discrete_memory_extracted == "f"
    ]

    if not unextracted_messages:
        logger.info(f"No unextracted messages for session {session_id}")
        await redis.delete(pending_key)
        return 0

    logger.info(
        f"Running trailing-edge extraction for session {session_id} "
        f"({len(unextracted_messages)} unextracted messages)"
    )

    try:
        extracted_memories = await extract_memories_from_session_thread(
            session_id=session_id,
            namespace=namespace,
            user_id=user_id,
            source_user=source_user,
            source_channel=source_channel,
            visibility=visibility,
        )

        # Mark all messages as extracted
        for message in working_memory.messages:
            message.discrete_memory_extracted = "t"

        # Persist the updated working memory
        await set_working_memory(
            working_memory=working_memory,
            redis_client=redis,
        )

        # Set post-extraction debounce
        await set_extraction_debounce(session_id, redis)

        # Clear the pending key
        await redis.delete(pending_key)

        # Index the extracted memories
        if extracted_memories:
            await index_long_term_memories(
                extracted_memories,
                deduplicate=True,
            )
            logger.info(
                f"Trailing extraction completed for session {session_id}: "
                f"{len(extracted_memories)} memories extracted and indexed"
            )

        return len(extracted_memories)

    except Exception as e:
        logger.error(f"Error in trailing extraction for session {session_id}: {e}")
        # Clear the pending key to allow retry on next message
        await redis.delete(pending_key)
        return 0


async def extract_memories_from_session_thread(
    session_id: str,
    namespace: str | None = None,
    user_id: str | None = None,
    source_user: str | None = None,
    source_channel: str | None = None,
    visibility: str | None = None,
) -> list[MemoryRecord]:
    """
    Extract memories from the entire conversation thread in working memory.

    This provides full conversational context for proper contextual grounding,
    allowing pronouns and references to be resolved across the entire thread.

    Args:
        session_id: The session ID to extract memories from
        namespace: Optional namespace for the memories
        user_id: Optional user ID for the memories
        source_user: Attribution source user to set on extracted memories
        source_channel: Attribution source channel to set on extracted memories
        visibility: Visibility scope to set on extracted memories (default: "everyone")

    Returns:
        List of extracted memory records with proper contextual grounding
    """
    from agent_memory_server.working_memory import get_working_memory

    # Get the complete working memory thread
    working_memory = await get_working_memory(
        session_id=session_id, namespace=namespace, user_id=user_id
    )

    if not working_memory or not working_memory.messages:
        logger.info(f"No working memory messages found for session {session_id}")
        return []

    # If source_user is None, try to infer from the session ID by matching
    # the peer ID against family.json identities. This handles the case where
    # the gateway stores working memory without attribution.
    from agent_memory_server.extraction import (
        resolve_user_display_name,
        resolve_user_from_session_id,
    )

    if source_user is None:
        inferred = resolve_user_from_session_id(session_id)
        if inferred:
            source_user = inferred
            logger.info(f"Inferred source_user={inferred} from session ID {session_id}")

    # Resolve display name for use in conversation labels and extraction prompt.
    # This ensures the LLM sees "[Chris Baker]: message" instead of "[USER]: message"
    # and produces "Chris Baker prefers..." instead of "User prefers..."
    resolved_name = resolve_user_display_name(source_user)

    # Build full conversation context with resolved names as labels
    conversation_messages = []
    for msg in working_memory.messages:
        if hasattr(msg, "role") and msg.role:
            role_lower = msg.role.lower()
            if role_lower in ("user", "human"):
                role_prefix = f"[{resolved_name}]: "
            elif role_lower in ("assistant", "ai"):
                role_prefix = "[Pat]: "
            else:
                role_prefix = f"[{msg.role.upper()}]: "
        else:
            role_prefix = ""
        conversation_messages.append(f"{role_prefix}{msg.content}")

    full_conversation = "\n".join(conversation_messages)

    logger.info(
        f"Extracting memories from {len(working_memory.messages)} messages "
        f"in session {session_id} (resolved_name={resolved_name})"
    )
    logger.debug(
        f"Full conversation context length: {len(full_conversation)} characters"
    )

    # Use the new memory strategy system for extraction
    from agent_memory_server.memory_strategies import get_memory_strategy

    try:
        # Get the discrete memory strategy for contextual grounding
        strategy = get_memory_strategy("discrete")

        # Extract memories using the strategy
        memories_data = await strategy.extract_memories(
            full_conversation, source_user_name=resolved_name
        )

        logger.info(
            f"Extracted {len(memories_data)} memories from session thread {session_id}"
        )

        # Convert to MemoryRecord objects with attribution
        resolved_visibility = visibility if visibility is not None else "everyone"
        extracted_memories = []
        for memory_data in memories_data:
            memory = MemoryRecord(
                id=str(ULID()),
                text=memory_data["text"],
                memory_type=memory_data.get("type", "semantic"),
                # F0 (LAB-395/LAB-397): epistemic kind from the extractor (coerced;
                # off-vocab/missing → None = 'fact') + the model-paraphrase
                # confidence tier (distinct from unscored first-hand writes).
                kind=coerce_extracted_kind(memory_data.get("kind")),
                confidence=settings.extraction_confidence,
                topics=memory_data.get("topics", []),
                entities=memory_data.get("entities", []),
                session_id=session_id,
                namespace=namespace,
                user_id=user_id,
                discrete_memory_extracted="t",  # Mark as extracted
                source_user=source_user,
                source_channel=source_channel,
                visibility=resolved_visibility,
            )
            extracted_memories.append(memory)

        return extracted_memories

    except Exception as e:
        logger.error(f"Error extracting memories from session thread {session_id}: {e}")
        return []


async def extract_memory_structure(
    memory: MemoryRecord,
    timeout: Timeout = Timeout(timedelta(minutes=settings.llm_task_timeout_minutes)),
):
    redis = await get_redis_conn()

    # Process messages for topic/entity extraction
    topics, entities = await handle_extraction(memory.text)

    merged_topics = memory.topics + topics if memory.topics else topics
    merged_entities = memory.entities + entities if memory.entities else entities

    # Quality filters: enforce controlled topic vocabulary and clean entities
    merged_topics = enforce_topics(merged_topics)
    merged_entities = clean_entities(merged_entities)

    # Safety guard: never overwrite existing good topics/entities with empty results.
    # If the memory already had topics and the merge produced nothing (e.g., LLM
    # extraction failed or enforce_topics dropped everything), keep the originals.
    if not merged_topics and memory.topics:
        merged_topics = enforce_topics(memory.topics)
    if not merged_entities and memory.entities:
        merged_entities = clean_entities(memory.entities)

    # Convert lists to pipe-separated strings for TAG fields
    # Issue #156 fix: langchain-redis uses pipe (|) as the default TAG separator
    topics_joined = "|".join(merged_topics) if merged_topics else ""
    entities_joined = "|".join(merged_entities) if merged_entities else ""

    await redis.hset(
        Keys.memory_key(memory.id),
        mapping={"topics": topics_joined, "entities": entities_joined},
    )  # type: ignore


async def merge_memories_with_llm(
    memories: list[MemoryRecord],
) -> MemoryRecord | None:
    """
    Use an LLM to merge similar or duplicate memories.

    Args:
        memories: List of MemoryRecord objects to merge

    Returns None if the LLM returns empty content (caller should keep originals).

    Returns:
        A merged memory
    """
    # If there's only one memory, just return it
    if len(memories) == 1:
        return memories[0]

    user_ids = {memory.user_id for memory in memories if memory.user_id}

    if len(user_ids) > 1:
        raise ValueError("Cannot merge memories with different user IDs")

    # Create a unified set of topics and entities
    all_topics = set()
    all_entities = set()

    for memory in memories:
        if memory.topics:
            all_topics.update(memory.topics)

        if memory.entities:
            all_entities.update(memory.entities)

    # Get the memory texts for LLM prompt
    memory_texts = [m.text for m in memories]

    # Construct the LLM prompt
    instruction = """You are a memory merging assistant. Merge the following memories into a single, coherent memory.

Rules:
1. Maximum 1000 characters. If the combined content exceeds this, summarize to the essential facts.
2. Never use 'User' to refer to a person — always use their actual name.
3. Do not include technical documentation, SQL schemas, or API references.
4. Keep the merged memory focused on a SINGLE topic. If the input memories cover different subjects, keep only the most specific or personal facts.
5. Output plain text only. No markdown formatting, no headers like '**Merged Memory:**' or '### Overview'."""

    memory_list = "\n".join([f"{i}: {text}" for i, text in enumerate(memory_texts, 1)])

    prompt = f"""{instruction}

The memories:
{memory_list}

Merged memory:"""

    model_name = settings.resolved_merge_model

    response = await LLMClient.create_chat_completion(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract the merged content — reject empty LLM responses
    merged_text = (response.content or "").strip()
    if not merged_text:
        logger.warning(
            "LLM returned empty merge response, refusing to create empty memory"
        )
        return None

    def coerce_to_float(m: MemoryRecord, key: str) -> float:
        try:
            val = getattr(m, key)
        except AttributeError:
            val = time.time()
        if val is None:
            return time.time()
        if isinstance(val, datetime):
            return float(val.timestamp())
        return float(val)

    # Use the oldest creation timestamp
    created_at = min(coerce_to_float(m, "created_at") for m in memories)

    # Use the most recent last_accessed timestamp
    last_accessed = max(coerce_to_float(m, "last_accessed") for m in memories)

    # Prefer non-empty namespace, user_id, session_id from memories
    namespace = next((m.namespace for m in memories if m.namespace), None)
    user_id = next((m.user_id for m in memories if m.user_id), None)
    session_id = next((m.session_id for m in memories if m.session_id), None)

    # Get the memory type from the first memory
    memory_type = next((m.memory_type for m in memories if m.memory_type), "semantic")

    # --- Attribution field propagation ---
    # source_user: all must agree (or be None). Different users = refuse merge.
    source_users = {
        getattr(m, "source_user", None)
        for m in memories
        if getattr(m, "source_user", None) is not None
    }
    if len(source_users) > 1:
        raise ValueError(
            f"Cannot merge memories with different source_user values: {source_users}"
        )
    merged_source_user = next(iter(source_users), None)

    # source_channel: take first non-None value
    merged_source_channel = next(
        (
            getattr(m, "source_channel", None)
            for m in memories
            if getattr(m, "source_channel", None) is not None
        ),
        None,
    )

    # visibility: most restrictive wins (higher rank = more restrictive)
    visibility_values = [getattr(m, "visibility", "everyone") for m in memories]
    merged_visibility = max(visibility_values, key=lambda v: VISIBILITY_RANK.get(v, 0))

    # stale_after: earliest (most conservative) non-None datetime
    stale_after_values = [
        getattr(m, "stale_after", None)
        for m in memories
        if getattr(m, "stale_after", None) is not None
    ]
    merged_stale_after = min(stale_after_values) if stale_after_values else None

    # Create the merged memory
    merged_memory = MemoryRecord(
        text=merged_text.strip(),
        id=str(ULID()),
        user_id=user_id,
        session_id=session_id,
        namespace=namespace,
        created_at=datetime.fromtimestamp(created_at, UTC),
        last_accessed=datetime.fromtimestamp(last_accessed, UTC),
        updated_at=datetime.now(UTC),
        topics=enforce_topics(list(all_topics)) if all_topics else None,
        entities=clean_entities(list(all_entities)) if all_entities else None,
        memory_type=MemoryTypeEnum(memory_type),
        discrete_memory_extracted="t",
        source_user=merged_source_user,
        source_channel=merged_source_channel,
        visibility=merged_visibility,
        stale_after=merged_stale_after,
    )

    # Size guard: reject oversized merged output (mega-memory prevention)
    merged_text_len = len(merged_memory.text)
    merged_entity_count = len(merged_memory.entities) if merged_memory.entities else 0
    if merged_text_len > MAX_MEMORY_OUTPUT_CHARS:
        logger.warning(
            f"Merge rejected: output {merged_text_len} chars exceeds "
            f"{MAX_MEMORY_OUTPUT_CHARS} char limit. Keeping first memory."
        )
        return memories[0]
    if merged_entity_count > MAX_ENTITY_COUNT:
        logger.warning(
            f"Merge rejected: {merged_entity_count} entities exceeds "
            f"{MAX_ENTITY_COUNT} entity limit. Keeping first memory."
        )
        return memories[0]

    # Generate a new hash for the merged memory
    merged_memory.memory_hash = generate_memory_hash(merged_memory)

    return merged_memory


async def compact_long_term_memories(
    limit: int = 1000,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    redis_client: Redis | None = None,
    vector_distance_threshold: float = 0.2,
    compact_hash_duplicates: bool = True,
    # Disabled 2026-03-10: semantic merge was destructive. Hash dedup is safe.
    compact_semantic_duplicates: bool = False,
    perpetual: Perpetual = Perpetual(
        # automatic=False when compaction_every_minutes is 0 (disabled).
        # The task can still be triggered manually via the /compact endpoint.
        every=timedelta(minutes=max(settings.compaction_every_minutes, 1)),
        automatic=settings.compaction_every_minutes > 0,
    ),
    timeout: Timeout = Timeout(timedelta(minutes=settings.llm_task_timeout_minutes)),
) -> int:
    """
    Compact long-term memories by merging duplicates and semantically similar memories.

    This function can identify and merge two types of duplicate memories:
    1. Hash-based duplicates: Memories with identical content (using memory_hash)
    2. Semantic duplicates: Memories with similar meaning but different text
       (DISABLED 2026-03-10 — semantic merge was destructive)

    Returns the count of remaining memories after compaction.
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    logger.info(
        f"Starting memory compaction: namespace={namespace}, "
        f"user_id={user_id}, session_id={session_id}, "
        f"hash_duplicates={compact_hash_duplicates}, "
        f"semantic_duplicates={compact_semantic_duplicates}"
    )

    # Build filters for memory queries
    filters = []
    if namespace:
        filters.append(f"@namespace:{{{namespace}}}")
    if user_id:
        filters.append(f"@user_id:{{{user_id}}}")
    if session_id:
        filters.append(f"@session_id:{{{session_id}}}")

    filter_str = " ".join(filters) if filters else "*"

    # Track metrics
    memories_merged = 0
    start_time = time.time()

    # Step 1: Compact hash-based duplicates using Redis aggregation
    if compact_hash_duplicates:
        logger.info("Starting hash-based duplicate compaction")
        try:
            index_name = Keys.search_index_name()

            # Create aggregation query to group by memory_hash and find duplicates
            agg_query = [
                "FT.AGGREGATE",
                index_name,
                filter_str,
                "GROUPBY",
                str(1),
                "@memory_hash",
                "REDUCE",
                "COUNT",
                str(0),
                "AS",
                "count",
                "FILTER",
                "@count>1",  # Only groups with more than 1 memory
                "SORTBY",
                str(2),
                "@count",
                "DESC",
                "LIMIT",
                str(0),
                str(limit),
            ]

            # Execute aggregation to find duplicate groups
            duplicate_groups = await redis_client.execute_command(*agg_query)

            if duplicate_groups and duplicate_groups[0] > 0:
                num_groups = duplicate_groups[0]
                logger.info(
                    f"Found {num_groups} groups with hash-based duplicates to process"
                )

                # Process each group of duplicates
                for i in range(1, len(duplicate_groups), 2):
                    try:
                        # Get the hash and count from aggregation results
                        group_data = duplicate_groups[i]
                        memory_hash = None
                        count = 0

                        for j in range(0, len(group_data), 2):
                            if group_data[j] == b"memory_hash":
                                if group_data[j + 1] is not None:
                                    memory_hash = group_data[j + 1].decode()
                            elif group_data[j] == b"count":
                                count = int(group_data[j + 1])

                        if not memory_hash or count <= 1:
                            continue

                        # Find all memories with this hash
                        # Use FT.SEARCH to find the actual memories with this hash
                        # TODO: Use RedisVL index
                        if filters:
                            # Combine hash query with filters using boolean AND
                            query_expr = f"(@memory_hash:{{{memory_hash}}}) ({' '.join(filters)})"
                        else:
                            query_expr = f"@memory_hash:{{{memory_hash}}}"

                        search_results = await redis_client.execute_command(
                            "FT.SEARCH",
                            index_name,
                            f"'{query_expr}'",
                            "RETURN",
                            "6",
                            "id_",
                            "text",
                            "last_accessed",
                            "created_at",
                            "user_id",
                            "session_id",
                            "SORTBY",
                            "last_accessed",
                            "ASC",
                        )

                        if search_results and search_results[0] > 1:
                            search_results[0]

                            # Keep the newest memory (last in sorted results)
                            # and delete the rest
                            memories_to_delete = []

                            # Parse FT.SEARCH results by scanning for keys
                            # with the expected prefix, rather than assuming a
                            # fixed stride. Records may have variable field
                            # counts if a hash is missing a field.
                            key_prefix = settings.redisvl_index_prefix + ":"
                            all_keys = []
                            for elem in search_results[1:]:
                                elem_str = (
                                    elem.decode()
                                    if isinstance(elem, bytes)
                                    else str(elem)
                                    if elem is not None
                                    else ""
                                )
                                if elem_str.startswith(key_prefix):
                                    all_keys.append(elem_str)

                            # Delete all but the last (newest, since sorted ASC)
                            for key_str in all_keys[:-1]:
                                memories_to_delete.append(key_str)

                            # Delete older duplicates
                            if memories_to_delete:
                                pipeline = redis_client.pipeline()
                                for key in memories_to_delete:
                                    pipeline.delete(key)

                                await pipeline.execute()
                                memories_merged += len(memories_to_delete)
                                logger.info(
                                    f"Deleted {len(memories_to_delete)} hash-based duplicates "
                                    f"with hash {memory_hash}"
                                )
                    except Exception as e:
                        logger.error(f"Error processing duplicate group: {e}")
            else:
                logger.info("No hash-based duplicates found")

            logger.info(
                f"Completed hash-based deduplication. Removed {memories_merged} duplicate memories."
            )
        except Exception as e:
            logger.error(f"Error during hash-based duplicate compaction: {e}")

    # Step 2: Compact semantic duplicates using vector search
    semantic_memories_merged = 0
    if compact_semantic_duplicates:
        logger.info("Starting semantic duplicate compaction")
        # Get the correct index name
        index_name = Keys.search_index_name()
        logger.info(f"Using index '{index_name}' for semantic duplicate compaction.")

        # Get all memories using the memory vector database
        try:
            # Convert filters to database format
            namespace_filter = None
            user_id_filter = None
            session_id_filter = None

            if namespace:
                from agent_memory_server.filters import Namespace

                namespace_filter = Namespace(eq=namespace)
            if user_id:
                from agent_memory_server.filters import UserId

                user_id_filter = UserId(eq=user_id)
            if session_id:
                from agent_memory_server.filters import SessionId

                session_id_filter = SessionId(eq=session_id)

            # Use memory vector database to get all memories using filter-only query
            # (no embedding required)
            db = await get_memory_vector_db()
            search_result = await db.list_memories(
                namespace=namespace_filter,
                user_id=user_id_filter,
                session_id=session_id_filter,
                limit=limit,
            )
        except Exception as e:
            logger.error(f"Error searching for memories: {e}")
            search_result = None

        if search_result and search_result.memories:
            logger.info(
                f"Found {search_result.total} memories to check for semantic duplicates"
            )

            # Process memories in batches to avoid overloading
            batch_size = 50
            processed_ids = set()  # Track which memories have been processed

            memories_list = search_result.memories
            for i in range(0, len(memories_list), batch_size):
                batch = memories_list[i : i + batch_size]

                for memory_result in batch:
                    memory_id = memory_result.id

                    # Skip if already processed
                    if memory_id in processed_ids:
                        continue

                    # Convert MemoryRecordResult to MemoryRecord for deduplication
                    memory_obj = MemoryRecord(
                        id=memory_result.id,
                        text=memory_result.text,
                        user_id=memory_result.user_id,
                        session_id=memory_result.session_id,
                        namespace=memory_result.namespace,
                        created_at=memory_result.created_at,
                        last_accessed=memory_result.last_accessed,
                        topics=memory_result.topics or [],
                        entities=memory_result.entities or [],
                        memory_type=memory_result.memory_type,  # type: ignore
                        discrete_memory_extracted=memory_result.discrete_memory_extracted,  # type: ignore
                        source_user=getattr(memory_result, "source_user", None),
                        source_channel=getattr(memory_result, "source_channel", None),
                        visibility=getattr(memory_result, "visibility", "everyone"),
                        stale_after=getattr(memory_result, "stale_after", None),
                    )

                    # Add this memory to processed list BEFORE processing to prevent cycles
                    processed_ids.add(memory_id)

                    # Size guard: skip oversized memories from compaction
                    mem_text_len = len(memory_obj.text) if memory_obj.text else 0
                    mem_entity_count = (
                        len(memory_obj.entities) if memory_obj.entities else 0
                    )
                    if (
                        mem_text_len > MAX_MEMORY_INPUT_CHARS
                        or mem_entity_count > MAX_ENTITY_COUNT
                    ):
                        logger.info(
                            f"Skipping compaction of oversized memory {memory_id}: "
                            f"{mem_text_len} chars, {mem_entity_count} entities"
                        )
                        continue

                    # Check for semantic duplicates
                    (
                        merged_memory,
                        was_merged,
                    ) = await deduplicate_by_semantic_search(
                        memory=memory_obj,
                        redis_client=redis_client,
                        namespace=namespace,
                        user_id=user_id,
                        session_id=session_id,
                        vector_distance_threshold=vector_distance_threshold,
                    )

                    if was_merged:
                        semantic_memories_merged += 1
                        # Delete the original memory using the database
                        await db.delete_memories([memory_id])

                        # Re-index the merged memory
                        if merged_memory:
                            await index_long_term_memories(
                                [merged_memory],
                                redis_client=redis_client,
                                deduplicate=False,  # Already deduplicated
                            )
                            # Mark the merged memory as processed to prevent cycles
                            processed_ids.add(merged_memory.id)
        logger.info(
            f"Completed semantic deduplication. Merged {semantic_memories_merged} memories."
        )

    # Get the count of remaining memories
    total_memories = await count_long_term_memories(
        namespace=namespace,
        user_id=user_id,
        session_id=session_id,
        redis_client=redis_client,
    )

    end_time = time.time()
    total_merged = memories_merged + semantic_memories_merged

    logger.info(
        f"Memory compaction completed in {end_time - start_time:.2f}s. "
        f"Merged {total_merged} memories. "
        f"{total_memories} memories remain."
    )

    return total_memories


# ============================================================================
# Content noise guard — rejects operational noise at the universal write funnel.
# This prevents tweet logs, meta-memories, analytics schemas, monitoring noise,
# and file-operation records from polluting the memory store.
# ============================================================================

_NOISE_TWEET_LOG = re.compile(
    r"Pat posted (?:a tweet|about|an? )|"
    r"@Hi_Its_Pat (?:posted|tweeted)|"
    r"tweet was posted|"
    r"posted (?:a |an )?(?:original )?tweet",
    re.IGNORECASE,
)

_NOISE_META_MEMORY = re.compile(
    r"memory.system.*(?:overhaul|redesign|audit|cleanup|migration|restore)|"
    r"memory.(?:quality|maintenance|compaction|dedup|consolidat).*(?:ran|completed|phase)|"
    r"mega.memory.*(?:split|decompos)|"
    r"(?:disabled|permanently).*dream.cycle|dream.cycle.*(?:disabled|prompt|configured)|"
    r"docket.*(?:compact|worker|disabled)|"
    r"backup.*file.*named.*\.(?:json|raw)|"
    r"(?:orphan|stale).*(?:memory|memories|keys).*(?:found|cleaned|deleted)|"
    r"memories.*were.*(?:cleaned|fixed|purged|deleted|restored)|"
    r"decomposition.*(?:task|workflow).*(?:stored|completed)|"
    r"memory.*hygiene|memory.*remediation|"
    # Reverse word order: "...audit of the memory system" / "review the memory
    # system" (the forward "memory.system.*audit" alternation above misses these).
    r"(?:audit|overhaul|redesign|review|cleanup|reindex).{0,30}memory.system|"
    r"stored a backup file named",
    re.IGNORECASE,
)

# Test-harness artifacts: contract/smoke/e2e test fixtures that leak into the
# corpus as if they were durable facts (LAB-406). "observes the moon" /
# "unique-marker-" are the seeded contract-test fixture markers.
_NOISE_TEST_ARTIFACT = re.compile(
    # TIGHTENED (codex F1, gateway-half parity): require a noise-context suffix or
    # a colon so a durable fact merely mentioning testing ("prefers contract test
    # coverage before deploys") is NOT rejected at the write funnel. unique-marker
    # anchored to its hex id shape.
    r"\b(?:contract|smoke|e2e) test(?:\s+(?:marker|fixture|artifact|message|response)|:)|"
    r"\btest artifact\b|"
    r"\bunique-marker-[0-9a-f-]{8,}|"
    r"\bpat observes the moon\b",
    re.IGNORECASE,
)

# Heartbeat / liveness check spam (LAB-406): "Pat reported HEARTBEAT_OK after a
# heartbeat check." — operational liveness pings, never durable knowledge.
_NOISE_HEARTBEAT = re.compile(
    r"\bHEARTBEAT_OK\b|"
    # Both "heartbeat check" and "reported a heartbeat" only count as noise when
    # an OPERATIONAL actor precedes them — never a bare medical/family phrase.
    # "Grant has a heartbeat check with cardiology on Friday" and "The
    # cardiologist reported a heartbeat during the ultrasound" are durable health
    # facts, NOT noise (codex F1: over-filtering health facts is corruption here).
    r"\b(?:gateway|service|server|monitor|agent|fleet|bot|cron|daemon|liveness)\b.{0,40}"
    r"\bheartbeat[ _-]?check\b|"
    r"\b(?:gateway|service|server|monitor|agent|fleet|bot|cron|daemon|liveness)\b.{0,40}"
    r"\breported\s+(?:a\s+)?heartbeat\b",
    re.IGNORECASE,
)

# Daily biometric log dumps (LAB-406): dated WHOOP recovery/HRV/RHR/strain
# readings (the ~218 "Lindsey's WHOOP 2026-03-20 — Recovery 52%" daily logs).
# Anchored on WHOOP + a numeric metric reading so durable health facts that
# happen to mention heart rate / training zones (no WHOOP) are NOT caught.
_NOISE_DAILY_BIOMETRIC = re.compile(
    r"\bWHOOP\b[^.]{0,60}?"
    r"\b(?:recovery|strain|hrv|rhr|resting heart rate|sleep performance)\b"
    r"[^.]{0,20}?\d",
    re.IGNORECASE,
)

_NOISE_ANALYTICS_SCHEMA = re.compile(
    r"(?:BigQuery|Dataform).*(?:table|column|schema|partition)|"
    r"(?:table|schema).*(?:partition(?:ed)?|cluster(?:ed)?|BigQuery|Dataform)|"
    r"(?:jitsu_events|google_ads|facebook_capi|meta_ads|mailcoach|marketing_funnel|"
    r"email_deliverability|property_listing|all_internal|users_unified).*"
    r"(?:table|column|schema|partition|affected|complex|optimization|source.*dependenc)|"
    r"interconnected tables.*(?:manage|track|analyze)|"
    r"The merged memory covers.*(?:advertising|email marketing|SEO metrics)|"
    # Catch remaining analytics table references with data-dictionary context
    r"(?:affected table|most complex table|source table).*"
    r"(?:jitsu_events|facebook_capi|meta_ads|users_unified|marketing_funnel)",
    re.IGNORECASE,
)

_NOISE_MONITORING = re.compile(
    r"no (?:activity|events?) (?:found|detected|recorded)|"
    r"no raw.scanner|no individual file|"
    r"(?:PDF|document|file) renamed from (?:scan_|document_)|"
    r"all (?:docker |homelab )?services (?:are )?healthy(?!\s*\w)|"
    # LAB-406: was over-anchored as ^...$, so "Nightly backup completed
    # successfully: …" (the real shape) slipped through. Match the phrase
    # anywhere — a "backup completed successfully" line is pure ops noise
    # regardless of surrounding words.
    r"backup completed successfully",
    re.IGNORECASE,
)


def _is_noise_content(text: str) -> bool:
    """Check if memory text is operational noise that should not be stored.

    Returns True for:
    - Tweet-by-tweet event logs (ephemeral, not durable knowledge)
    - Meta-memories about the memory system itself
    - Analytics/BigQuery schema dumps
    - Monitoring status reports and file-operation records
    - Test-harness artifacts (contract/smoke/e2e fixtures) (LAB-406)
    - Heartbeat / liveness-check spam (LAB-406)
    - Dated WHOOP daily-biometric log dumps (LAB-406)
    """
    return bool(
        _NOISE_TWEET_LOG.search(text)
        or _NOISE_META_MEMORY.search(text)
        or _NOISE_ANALYTICS_SCHEMA.search(text)
        or _NOISE_MONITORING.search(text)
        or _NOISE_TEST_ARTIFACT.search(text)
        or _NOISE_HEARTBEAT.search(text)
        or _NOISE_DAILY_BIOMETRIC.search(text)
    )


def _within_validity_window(m: MemoryRecordResult, as_of_ts: float) -> bool:
    """LAB-405: True if record `m`'s validity window contains the instant `as_of_ts`.

    `valid_from <= as_of < valid_to` (end-exclusive). Graceful on legacy records:
    `valid_from` None → no lower bound (always-started); `valid_to` None (open) →
    always currently-valid (no upper bound). Pure + side-effect-free.
    """
    if m.valid_from is not None and as_of_ts < m.valid_from.timestamp():
        return False  # not yet valid at as_of
    if m.valid_to is not None and as_of_ts >= m.valid_to.timestamp():
        return False  # validity ended at/before as_of
    return True


# Source user normalization — maps slugified names back to full names.
# The family registry at ~/.openclaw/family.json is the authoritative source.
# Falls back to title-casing the slug if the registry isn't available.
_FAMILY_NAME_MAP: dict[str, str] | None = None


def _normalize_source_user(source_user: str) -> str:
    """Normalize source_user from slug/short name to full name via family registry.

    Handles all known input formats:
      - Short key:    "chris"        → "Chris Baker"
      - Snake slug:   "chris_baker"  → "Chris Baker"
      - No-sep slug:  "chrisbaker"   → "Chris Baker"
      - Display name: "Chris"        → "Chris Baker"
      - Full name:    "Chris Baker"  → "Chris Baker" (passthrough)

    Uses the same family.json structure and full-name construction as
    extraction.py:_load_family_registry() — display_name + lastName (default "Baker").
    """
    global _FAMILY_NAME_MAP
    if _FAMILY_NAME_MAP is None:
        _FAMILY_NAME_MAP = {}
        try:
            import json
            import os

            family_path = os.path.expanduser("~/.openclaw/family.json")
            with open(family_path) as f:
                family = json.load(f)
            # family.json: {"users": {"chris": {"displayName": "Chris", ...}, ...}}
            users = family.get("users", {})
            if isinstance(users, dict):
                for key, user_obj in users.items():
                    display = user_obj.get("displayName", key.title())
                    last_name = user_obj.get("lastName", "Baker")
                    full_name = f"{display} {last_name}"
                    # Map every known variant to the canonical full name
                    _FAMILY_NAME_MAP[key.lower()] = full_name  # "chris"
                    _FAMILY_NAME_MAP[display.lower()] = full_name  # "chris"
                    _FAMILY_NAME_MAP[full_name.lower()] = full_name  # "chris baker"
                    _FAMILY_NAME_MAP[full_name.lower().replace(" ", "_")] = (
                        full_name  # "chris_baker"
                    )
                    _FAMILY_NAME_MAP[full_name.lower().replace(" ", "")] = (
                        full_name  # "chrisbaker"
                    )
        except Exception:
            pass  # No family registry — fall back to title-case

    # Exact match in map
    lower = source_user.lower().strip()
    if lower in _FAMILY_NAME_MAP:
        return _FAMILY_NAME_MAP[lower]

    # Slug pattern (underscores) → title case
    if "_" in source_user and source_user == source_user.lower():
        return source_user.replace("_", " ").title()

    return source_user


def _coerce_event_date(value: Any) -> datetime | None:
    """Normalize an event_date value to a datetime, or None.

    External scripts writing directly to Redis can store ISO strings like
    "2026-02-22" in the NUMERIC event_date field, which makes Redis Search
    silently EXCLUDE the record from the index (an orphan key). Coerce at the
    write funnel: a datetime passes through, a parseable ISO string is
    converted, and anything unparseable (or None) becomes None rather than
    corrupting the index. Pure + side-effect-free so it is directly testable.
    """
    if value is None or isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


# "User"-as-person reference pattern. Matches "User <verb>" at the start of the
# text or mid-sentence (after ", " / ". "), e.g. "User prefers …" or
# "On March 14, User asked …". Keep this verb list aligned with
# tests/memory/attribution.sh::check_no_user_prefix.
_USER_VERB_PATTERN = (
    r"(?:^|[,.]\s+)User\s+(?:is|was|has|had|does|did|prefers|likes|wants|"
    r"mentioned|asked|enjoys|works|lives|loves|needs|feels|"
    r"believes|thinks|participates|uses|values|gave|ordered|"
    r"bought|tends|keeps|expressed|currently|also|recently|"
    r"reported|said|told|requested|inquired|checked|noted|"
    r"learned|started|stopped|stored|saved|added|removed|created|"
    r"deleted|updated|visited|sent|received|replied|shared|"
    r"confirmed|completed|finished|opened|closed|approved|rejected|"
    r"wrote|read|bought|paid|owes|owns|plans|intends|knows)\b"
)


def _is_user_as_person(text: str) -> bool:
    """True if `text` uses the generic literal "User" as a person reference.

    Extraction paths that fail to resolve the speaker's real name emit "User"
    as the fact subject ("User prefers dark mode"); storing it both
    mis-attributes the fact and bypasses the attribution model, so such
    memories are rejected at the write funnel. The leading "[tag]" envelope
    is stripped before matching. Pure + side-effect-free for direct testing.
    """
    stripped = re.sub(r"^\[.*?\]\s*", "", text)
    return bool(re.search(_USER_VERB_PATTERN, stripped))


async def index_long_term_memories(
    memories: list[MemoryRecord | ExtractedMemoryRecord],
    redis_client: Redis | None = None,
    deduplicate: bool = False,
    vector_distance_threshold: float | None = None,
) -> None:
    """
    Index long-term memories using the pluggable memory vector database.

    Args:
        memories: List of long-term memories to index
        redis_client: Optional Redis client (kept for compatibility, may be unused depending on backend)
        deduplicate: Whether to deduplicate memories before indexing
        vector_distance_threshold: Threshold for semantic similarity.
            If None, uses settings.deduplication_distance_threshold (default 0.35)

    Note:
        Breaking change in v0.x: The default threshold changed from 0.12 to 0.35
        (via settings.deduplication_distance_threshold). The new threshold better
        catches paraphrased content. To restore the old strict behavior, explicitly
        pass vector_distance_threshold=0.12.
    """
    background_tasks = get_background_tasks()

    # Filter out memories with empty text or id before processing
    # Empty text causes OpenAI's embedding API to reject with "'$.input' is invalid"
    valid_memories = []
    for memory in memories:
        if not memory.text:
            logger.warning(
                f"Skipping memory with empty text: id={memory.id}",
            )
            continue
        if not memory.id:
            logger.warning(
                f"Skipping memory with empty id: text={memory.text[:50] if memory.text else ''}...",
            )
            continue

        # Content security pass (ported from wfr-memory-commons dlp.py +
        # content_trust.py): NFC-normalize + strip control/zero-width chars,
        # redact secret shapes (API keys, private-key blocks, tokens) before the
        # text is embedded or stored, and flag (log + telemetry) injection-shaped
        # text. Runs FIRST so downstream guards (noise, size, "User") operate on
        # the cleaned + redacted text. Removal-only for sanitize/redact, so this
        # can only shorten text — never bypasses the size guard below. Each step
        # is independently gated in config; redact/sanitize default ON.
        sec = apply_content_security(
            memory.text,
            sanitize=settings.memory_text_sanitization_enabled,
            redact=settings.memory_secret_redaction_enabled,
            scan=settings.memory_injection_scan_enabled,
        )
        if sec.secrets_redacted:
            # Log the labels (the secret TYPE), never the secret itself.
            logger.warning(
                f"Redacted secret(s) from memory before storage: "
                f"labels={list(sec.secret_labels)}, id={memory.id}"
            )
            record_counter(
                "memory_server.content_security.secrets_redacted",
                value=float(len(sec.secret_labels)),
                attributes={"labels": ",".join(sec.secret_labels)},
            )
        if sec.injection_signals:
            # Flag-and-keep: surface for review, but store the record. The
            # load-bearing injection defense is the gateway's read-time handling.
            # Log the SANITIZED/secret-redacted text (sec.text), never the raw input
            # (LAB-65): the raw memory.text matched an injection signal and may carry
            # attacker-controlled payloads or unredacted secrets — emitting it to logs is a
            # re-injection / secret-leak vector. sec.text has secrets → [REDACTED] and
            # control/bidi chars stripped.
            logger.warning(
                f"Memory text matched injection signal(s) (flagged, not rejected): "
                f"signals={list(sec.injection_signals)}, id={memory.id}, "
                f"redacted_text={sec.text[:80]}..."
            )
            record_counter(
                "memory_server.content_security.injection_flagged",
                value=1.0,
                attributes={"signals": ",".join(sec.injection_signals)},
            )
        if sec.changed:
            memory = memory.model_copy()
            memory.text = sec.text

        # Content noise guard: reject memories that are operational noise,
        # not genuine user knowledge. This catches all write paths (API,
        # manager, extraction, promotion, plugin).
        if _is_noise_content(memory.text):
            logger.info(
                f"Skipping noise memory: id={memory.id}, text={memory.text[:80]}...",
            )
            continue

        # Size guard: truncate oversized memories at ingestion to prevent bloat.
        # MAX_MEMORY_OUTPUT_CHARS (1000) is the maximum allowed text length.
        # This catches all write paths: API, manager, extraction, promotion, plugin.
        # Oversized text is truncated at the last sentence boundary within the limit.
        # Work on a shallow copy so callers' MemoryRecord objects are not mutated.
        text_len = len(memory.text)
        if text_len > MAX_MEMORY_OUTPUT_CHARS:
            memory = memory.model_copy()
            truncated = memory.text[:MAX_MEMORY_OUTPUT_CHARS]
            # Try to truncate at a sentence boundary for cleaner text
            last_period = truncated.rfind(". ")
            last_newline = truncated.rfind("\n")
            best_break = max(last_period, last_newline)
            if best_break > MAX_MEMORY_OUTPUT_CHARS // 2:
                truncated = truncated[: best_break + 1].rstrip()
            else:
                truncated = truncated.rstrip()
            memory.text = truncated
            logger.warning(
                f"Truncated oversized memory from {text_len} to {len(truncated)} chars: "
                f"id={memory.id}, text={truncated[:80]}...",
            )

        # Event date guard: ensure event_date is a proper datetime, not a string.
        # External scripts writing directly to Redis can store ISO strings like
        # "2026-02-22" in the NUMERIC event_date field, causing Redis Search to
        # silently exclude the record from the index (orphan key).
        if memory.event_date is not None and not isinstance(
            memory.event_date, datetime
        ):
            original_event_date = memory.event_date
            coerced = _coerce_event_date(original_event_date)
            memory = memory.model_copy()
            memory.event_date = coerced
            if coerced is not None:
                logger.warning(
                    f"Converted string event_date to datetime for memory {memory.id}: "
                    f"{coerced}"
                )
            else:
                logger.warning(
                    f"Invalid event_date for memory {memory.id}, setting to None: "
                    f"{original_event_date!r}"
                )

        # "User" text guard: reject memories that use "User" as a person name.
        # This catches all extraction paths that fail to resolve the actual name.
        # Two patterns:
        # 1. Text starts with "User <verb>" (after optional tag)
        # 2. "User <verb>" appears anywhere in the text as a mid-sentence person
        #    reference (e.g., "On March 14, User asked...")
        if _is_user_as_person(memory.text):
            logger.warning(
                f"Rejecting memory with 'User' as person reference "
                f"(source_user={getattr(memory, 'source_user', None)}): "
                f"id={memory.id}, text={memory.text[:80]}..."
            )
            continue

        # Source user normalization: convert slugs (e.g., "chris_baker") to
        # full names ("Chris Baker"). The session-memory-bridge and other callers
        # sometimes pass slugified names. Normalize at ingestion so all stored
        # records have consistent human-readable source_user values.
        if hasattr(memory, "source_user") and memory.source_user:
            normalized = _normalize_source_user(memory.source_user)
            if normalized != memory.source_user:
                memory = memory.model_copy()
                memory.source_user = normalized

        # Provenance temporal stamping (ported from wfr-memory-commons): when the
        # writer doesn't supply observed_at / valid_from, default both to
        # created_at so every record has a populated validity window + last-
        # observed timestamp. valid_to stays None ("still valid"); superseded_by
        # stays None — both are server-managed by the supersede endpoint.
        if memory.observed_at is None or memory.valid_from is None:
            memory = memory.model_copy()
            if memory.observed_at is None:
                memory.observed_at = memory.created_at
            if memory.valid_from is None:
                memory.valid_from = memory.created_at

        valid_memories.append(memory)

    if not valid_memories:
        logger.info("No valid memories to index (all had empty text or id)")
        return

    # Process memories for deduplication if requested
    processed_memories = []
    if deduplicate:
        # Get Redis client for deduplication operations (still needed for existing dedup logic)
        redis = redis_client or await get_redis_conn()

        for memory in valid_memories:
            current_memory = memory
            was_deduplicated = False

            # Check for id-based duplicates
            if not was_deduplicated:
                deduped_memory, was_overwrite = await deduplicate_by_id(
                    memory=current_memory,
                    redis_client=redis,
                )
                if was_overwrite:
                    # This overwrote an existing memory with the same ID
                    current_memory = deduped_memory or current_memory
                    logger.info(f"Overwrote memory with ID {memory.id}")
                else:
                    current_memory = deduped_memory or current_memory

            # Check for hash-based duplicates
            if not was_deduplicated:
                deduped_memory, was_dup = await deduplicate_by_hash(
                    memory=current_memory,
                    redis_client=redis,
                )
                if was_dup:
                    # This is a duplicate, skip it
                    was_deduplicated = True
                else:
                    current_memory = deduped_memory or current_memory

            # Check for semantic duplicates
            if not was_deduplicated:
                deduped_memory, was_merged = await deduplicate_by_semantic_search(
                    memory=current_memory,
                    redis_client=redis,
                    vector_distance_threshold=vector_distance_threshold,
                )
                if was_merged:
                    current_memory = deduped_memory or current_memory

            # Add the memory to be indexed if not a pure duplicate
            if not was_deduplicated:
                processed_memories.append(current_memory)
    else:
        processed_memories = valid_memories

    # If all memories were duplicates, we're done
    if not processed_memories:
        logger.info("All memories were duplicates, nothing to index")
        return

    # Get the memory vector database and add memories
    db = await get_memory_vector_db()

    # Add memories to the database
    try:
        ids = await db.add_memories(processed_memories)
        logger.info(f"Indexed {len(processed_memories)} memories with IDs: {ids}")
    except Exception as e:
        logger.error(f"Error indexing memories: {e}")
        raise

    # Schedule background tasks for topic/entity extraction.
    # Skip memories that already have topics AND entities populated —
    # callers like Finley's slack-scanner pre-normalize these fields
    # and background re-extraction would overwrite them with lower-quality
    # LLM-generated alternatives.
    for memory in processed_memories:
        has_topics = memory.topics and len(memory.topics) > 0
        has_entities = memory.entities and len(memory.entities) > 0
        if has_topics and has_entities:
            logger.debug(
                f"Skipping extract_memory_structure for {memory.id} — "
                f"already has {len(memory.topics)} topics + {len(memory.entities)} entities"
            )
            continue
        background_tasks.add_task(extract_memory_structure, memory)

    if settings.enable_discrete_memory_extraction:
        needs_extraction = [
            memory
            for memory in processed_memories
            if memory.discrete_memory_extracted == "f"
        ]
        # Extract discrete memories from the indexed messages and persist
        # them as separate long-term memory records. This process also
        # runs deduplication if requested.
        background_tasks.add_task(
            extract_memories_with_strategy,
            memories=needs_extraction,
            deduplicate=deduplicate,
        )


async def _observe_recall_egress(record_count: int, *, exempt: bool = False) -> None:
    """Feed a recall's returned-record count into the global egress volume guard
    (C5; see utils/egress_guard.py) and, on a flagged window, log + emit
    telemetry. DETECT-ONLY — never blocks recall; fail-open on any Redis error
    (record_and_check swallows RedisError, and this wrapper swallows anything
    else so the guard can never break the recall hot path). Called at every
    recall return site (the filter-only listing AND the semantic-search path),
    since the bulk-drain vector applies to both.

    ``exempt`` — when True, the call is a no-op (no counter increment, no
    classify). This is how trusted INTERNAL full-corpus enumeration is kept out
    of the shared window. The signal is the existing ``bypass_recall_filters``
    flag (LAB-281). In this single-tenant loopback deployment that flag is set
    by the curator's nightly full-corpus backup, which legitimately paginates the
    entire corpus (~10k+ rows/run) through the search API in a 60s burst
    (LAB-388) — and is trusted to be the only caller that sets it. Without this
    exemption that one trusted caller poisoned the global counter to ~5x the soft
    cap every night, sustaining a false ``flagged`` state that would drown a real
    bulk-exfil signal. The exemption is deliberately keyed on the trusted-caller
    flag (approach (a)) — NOT a blanket cap raise — so the real-exfil detection
    floor (a non-exempt caller offset-walking the corpus WITHOUT the flag) is
    unchanged: such a caller still increments the shared window and still flags
    above the cap. CAVEAT: ``bypass_recall_filters`` is a wire-settable
    ``SearchRequest`` field, so the exemption is a deployment convention, not an
    enforced invariant — a caller that reaches the search endpoint can set it and
    opt out of the (detect-only) guard. Acceptable while detect-only/fail-open;
    if the guard is ever flipped to ENFORCING, gate the exemption on a
    server-trusted signal (internal-call marker / operator token, FORK.md #17)
    rather than the wire-settable flag."""
    if exempt:
        return
    try:
        config = egress_config_from_settings()
        if not config.enabled or record_count <= 0:
            return
        redis = await get_redis_conn()
        verdict = await egress_record_and_check(redis, record_count, config=config)
        if verdict.outcome == "flagged":
            logger.warning(
                f"[search_long_term_memories] egress guard FLAGGED — {verdict.reason} "
                f"(this request returned {verdict.this_request} records)"
            )
            record_counter(
                "memory_server.egress_guard.flagged",
                value=1.0,
                attributes={
                    "window_total": verdict.window_total,
                    "window_seconds": config.window_seconds,
                    "max_records": config.max_records,
                },
            )
    except Exception as exc:  # noqa: BLE001 — defense-in-depth must never break recall
        logger.warning(
            f"[search_long_term_memories] egress guard error (ignored): {exc}"
        )


async def search_long_term_memories(
    text: str,
    session_id: SessionId | None = None,
    user_id: UserId | None = None,
    namespace: Namespace | None = None,
    created_at: CreatedAt | None = None,
    last_accessed: LastAccessed | None = None,
    topics: Topics | None = None,
    entities: Entities | None = None,
    distance_threshold: float | None = None,
    memory_type: MemoryType | None = None,
    event_date: EventDate | None = None,
    memory_hash: MemoryHash | None = None,
    source_user: SourceUser | None = None,
    source_channel: SourceChannel | None = None,
    visibility: VisibilityFilter | None = None,
    stale_after: StaleAfter | None = None,
    kind: Kind | None = None,
    min_confidence: MinConfidence | None = None,
    include_superseded: bool = False,
    as_of: datetime | None = None,
    server_side_recency: bool | None = None,
    recency_params: dict | None = None,
    limit: int = 10,
    offset: int = 0,
    optimize_query: bool = False,
    bypass_recall_filters: bool = False,
) -> MemoryRecordResults:
    """
    Search for long-term memories using the pluggable memory vector database.

    Args:
        text: Query for vector search - will be used for semantic similarity matching
        session_id: Optional session ID filter
        user_id: Optional user ID filter
        namespace: Optional namespace filter
        created_at: Optional created at filter
        last_accessed: Optional last accessed filter
        topics: Optional topics filter
        entities: Optional entities filter
        distance_threshold: Optional similarity threshold
        memory_type: Optional memory type filter
        event_date: Optional event date filter
        memory_hash: Optional memory hash filter
        source_user: Optional source user filter
        source_channel: Optional source channel filter
        visibility: Optional visibility scope filter
        stale_after: Optional stale-after timestamp filter
        limit: Maximum number of results
        offset: Offset for pagination
        optimize_query: Whether to optimize the query for vector search using a fast model (default: False)

    Returns:
        MemoryRecordResults containing matching memories
    """
    # LAB-405 (codex F2): when as_of is set, the validity-window post-filter can
    # trim the raw page below `limit`, so over-fetch a bounded multiple from the
    # DB and truncate back to `limit` AFTER filtering — a valid-at-as_of record
    # just past the raw `limit` is then still considered, and an all-invalid first
    # page no longer collapses to total=0. Deep offset pagination under as_of stays
    # best-effort: a complete fix needs valid_from indexed (a reindex, out of scope
    # per FORK.md #28); valid_from is store-and-return only.
    db_limit = limit if as_of is None else min(max(limit, limit * 5), 200)

    # If no query text is provided, perform a filter-only listing (no semantic search).
    # This enables patterns like: "return all memories for this user/namespace".
    if not (text or "").strip():
        db = await get_memory_vector_db()
        listing = await db.list_memories(
            session_id=session_id,
            user_id=user_id,
            namespace=namespace,
            created_at=created_at,
            last_accessed=last_accessed,
            topics=topics,
            entities=entities,
            memory_type=memory_type,
            event_date=event_date,
            memory_hash=memory_hash,
            source_user=source_user,
            source_channel=source_channel,
            visibility=visibility,
            stale_after=stale_after,
            limit=db_limit,
            offset=offset,
        )
        # LAB-405 (codex F2): the filter-only listing path returns BEFORE the
        # vector-path as_of post-filter below, so time-travel recall must be
        # applied here too — otherwise a metadata-only "list all" recall with
        # as_of would leak future/expired records. Over-fetched db_limit is
        # window-filtered then truncated back to `limit`.
        if as_of is not None and listing.memories:
            as_of_ts = as_of.timestamp()
            listing.memories = [
                m for m in listing.memories if _within_validity_window(m, as_of_ts)
            ][:limit]
            listing.total = len(listing.memories)
        await _observe_recall_egress(
            len(listing.memories), exempt=bypass_recall_filters
        )
        return listing

    # Search-query length clamp. A pathologically long recall query (a whole
    # pasted document, an over-stuffed conversation window) is clamped before
    # embedding so it degrades to a partial semantic search rather than erroring
    # or wasting an oversized embedding call. Adapted from wfr-finley
    # clampSearchText. Applied to the ORIGINAL text so both the optimized and
    # fallback paths below operate on the bounded query.
    _query_cap = settings.max_search_query_chars
    if _query_cap and len(text) > _query_cap:
        logger.info(
            f"[search_long_term_memories] clamping query from {len(text)} to {_query_cap} chars"
        )
        text = text[:_query_cap]

    # Optimize query for vector search if requested.
    search_query = text
    optimized_applied = False
    if optimize_query and text:
        search_query = await optimize_query_for_vector_search(text)
        optimized_applied = True

    # Debug: Log search input
    optimized_display = repr(search_query) if optimized_applied else "N/A"
    logger.debug(
        f"[search_long_term_memories] INPUT - query: {text!r}, "
        f"optimized_query: {optimized_display}, "
        f"session_id: {session_id}, user_id: {user_id}, namespace: {namespace}, "
        f"distance_threshold: {distance_threshold}, limit: {limit}"
    )

    # Get the memory vector database
    db = await get_memory_vector_db()

    # Delegate search to the database
    results = await db.search_memories(
        query=search_query,
        session_id=session_id,
        user_id=user_id,
        namespace=namespace,
        created_at=created_at,
        last_accessed=last_accessed,
        topics=topics,
        entities=entities,
        memory_type=memory_type,
        event_date=event_date,
        memory_hash=memory_hash,
        source_user=source_user,
        source_channel=source_channel,
        visibility=visibility,
        stale_after=stale_after,
        kind=kind,
        min_confidence=min_confidence,
        distance_threshold=distance_threshold,
        server_side_recency=server_side_recency,
        recency_params=recency_params,
        limit=db_limit,
        offset=offset,
    )

    # If an optimized query with a strict distance threshold returns no results,
    # retry once with the original query to preserve recall.
    try:
        if (
            optimized_applied
            and distance_threshold is not None
            and results.total == 0
            and search_query != text
        ):
            results = await db.search_memories(
                query=text,
                session_id=session_id,
                user_id=user_id,
                namespace=namespace,
                created_at=created_at,
                last_accessed=last_accessed,
                topics=topics,
                entities=entities,
                memory_type=memory_type,
                event_date=event_date,
                memory_hash=memory_hash,
                source_user=source_user,
                source_channel=source_channel,
                visibility=visibility,
                stale_after=stale_after,
                kind=kind,
                min_confidence=min_confidence,
                distance_threshold=distance_threshold,
                server_side_recency=server_side_recency,
                recency_params=recency_params,
                limit=db_limit,
                offset=offset,
            )
    except Exception as e:
        logger.warning("Optimized-query fallback search failed: %s", e)

    # Recall relevance gate (deterministic; see utils/relevance.py). Trims the
    # weak-distance tail that shares no salient term with the query. Default-OFF;
    # shadow mode logs what would drop without dropping. Applied against the
    # ORIGINAL query text (`text`) — the user's words, not the LLM-optimized
    # rewrite — since the gate measures lexical anchoring to what was asked.
    # bypass_recall_filters: full-corpus enumeration (curator backup, LAB-281). The
    # gate keys off the global setting and would otherwise trim every page of a
    # text-anchored "list all" export down to the salient-term subset (~93/16976),
    # so an opt-in per-request bypass lets a trusted caller read the whole corpus.
    if (
        not bypass_recall_filters
        and settings.recall_relevance_gate_enabled
        and results.memories
    ):
        outcome = apply_relevance_gate(
            [(m.text, m.dist) for m in results.memories],
            text,
            enabled=settings.recall_relevance_gate_enabled,
            shadow=settings.recall_relevance_gate_shadow,
            distance_floor=settings.recall_relevance_gate_distance_floor,
        )
        if outcome.dropped_count:
            _verb = "would drop (shadow)" if outcome.shadow else "dropped"
            _floor = settings.recall_relevance_gate_distance_floor
            logger.info(
                f"[search_long_term_memories] relevance gate {_verb} "
                f"{outcome.dropped_count}/{len(results.memories)} weak-tail results "
                f"(floor={_floor:.3f}, query={text[:80]!r})"
            )
            # Emit to SigNoz so the gate's drop rate is observable — the over-
            # trimming risk FORK.md flags (a gate that silently eats conceptual
            # recall is invisible otherwise). `shadow` distinguishes would-drop
            # from actually-dropped so a shadow evaluation can be compared.
            record_counter(
                "memory_server.recall_relevance_gate.dropped",
                value=float(outcome.dropped_count),
                attributes={"shadow": str(outcome.shadow).lower()},
            )
        if not outcome.shadow and outcome.dropped_count:
            kept = set(outcome.kept_indices)
            results.memories = [m for i, m in enumerate(results.memories) if i in kept]
            results.total = len(results.memories)

    # Time-travel recall (LAB-405). When `as_of` is set, the validity window is the
    # authoritative filter: keep records valid at that instant
    # (`valid_from <= as_of < valid_to`) and SKIP the default superseded-hide below —
    # a record superseded AFTER as_of was still valid then (its valid_to is later than
    # as_of) and must appear. Done as a post-filter (NOT a Redis predicate) so legacy
    # records lacking valid_from/valid_to degrade gracefully:
    #   - valid_from None  → no lower bound (always-started)
    #   - valid_to   None  → open-ended, always currently-valid (no upper bound)
    if as_of is not None and results.memories:
        as_of_ts = as_of.timestamp()
        before = len(results.memories)
        # Window-filter the over-fetched candidate pool (db_limit), then truncate
        # back to the caller's `limit` (codex F2).
        results.memories = [
            m for m in results.memories if _within_validity_window(m, as_of_ts)
        ][:limit]
        excluded = before - len(results.memories)
        if excluded:
            results.total = len(results.memories)
            logger.debug(
                f"[search_long_term_memories] as_of={as_of.isoformat()} "
                f"excluded/truncated {excluded} record(s) (db_limit={db_limit})"
            )
    # Supersede-hiding (versioning; ported from wfr-memory-commons). By default,
    # records that have been replaced by a newer version (superseded_by set) are
    # hidden from recall. Done as a post-filter — NOT a Redis query predicate —
    # so records written before this field existed (no superseded_by on the hash)
    # are correctly treated as "not superseded" and always kept. include_superseded
    # surfaces them (e.g. for history/audit views). Skipped under as_of (above).
    elif not include_superseded and results.memories:
        before = len(results.memories)
        results.memories = [m for m in results.memories if not m.superseded_by]
        hidden = before - len(results.memories)
        if hidden:
            results.total = len(results.memories)
            logger.debug(
                f"[search_long_term_memories] hid {hidden} superseded record(s)"
            )

    # Debug: Log search output
    memory_previews = [
        f"{m.id}: {m.text[:80]}..." if len(m.text) > 80 else f"{m.id}: {m.text}"
        for m in results.memories[:5]  # Show first 5
    ]
    logger.debug(
        f"[search_long_term_memories] OUTPUT - {results.total} results: {memory_previews}"
    )

    # Egress volume guard (C5) — count records actually returned (post relevance
    # gate + supersede hiding) into the global fixed-window counter. Detect-only.
    # bypass_recall_filters marks a trusted internal full-corpus enumeration
    # (curator backup, LAB-281) and exempts it from the shared window (LAB-388).
    await _observe_recall_egress(len(results.memories), exempt=bypass_recall_filters)

    return results


async def count_long_term_memories(
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    redis_client: Redis | None = None,
) -> int:
    """
    Count the total number of long-term memories matching the given filters.

    Uses the pluggable memory vector database instead of direct Redis calls.

    Args:
        namespace: Optional namespace filter
        user_id: Optional user ID filter
        session_id: Optional session ID filter
        redis_client: Optional Redis client (for compatibility)

    Returns:
        Total count of memories matching filters
    """
    # Get the memory vector database
    db = await get_memory_vector_db()

    # Delegate to the database
    return await db.count_memories(
        namespace=namespace,
        user_id=user_id,
        session_id=session_id,
    )


async def deduplicate_by_hash(
    memory: MemoryRecord,
    redis_client: Redis | None = None,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> tuple[MemoryRecord | None, bool]:
    """
    Check if a memory has hash-based duplicates and handle accordingly.

    Memories have a hash generated from their text and metadata. If we
    see the exact-same memory again, we ignore it.

    Args:
        memory: The memory to check for duplicates
        redis_client: Optional Redis client
        namespace: Optional namespace filter
        user_id: Optional user ID filter
        session_id: Optional session ID filter

    Returns:
        Tuple of (memory to save (if any), was_duplicate)
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    # Generate hash for the memory
    memory_hash = generate_memory_hash(memory)

    # Use memory vector database to search for memories with the same hash
    # Build filter objects
    namespace_filter = None
    if namespace or memory.namespace:
        namespace_filter = Namespace(eq=namespace or memory.namespace)

    user_id_filter = None
    if user_id or memory.user_id:
        user_id_filter = UserId(eq=user_id or memory.user_id)

    session_id_filter = None
    if session_id or memory.session_id:
        session_id_filter = SessionId(eq=session_id or memory.session_id)

    # Create memory hash filter
    memory_hash_filter = MemoryHash(eq=memory_hash)

    # Use memory vector database to search for memories with the same hash
    db = await get_memory_vector_db()

    # Search for existing memories with the same hash using filter-only query
    # (no embedding required)
    results = await db.list_memories(
        session_id=session_id_filter,
        user_id=user_id_filter,
        namespace=namespace_filter,
        memory_hash=memory_hash_filter,
        limit=1,  # We only need to know if one exists
    )

    if results.memories and len(results.memories) > 0:
        # Found existing memory with the same hash
        logger.info(f"Found existing memory with hash {memory_hash}")

        # Update the last_accessed timestamp of the existing memory
        existing_memory = results.memories[0]
        if existing_memory.id:
            # Use the memory key format to update last_accessed
            existing_key = Keys.memory_key(existing_memory.id)
            await redis_client.hset(
                existing_key,
                mapping={"last_accessed": str(int(datetime.now(UTC).timestamp()))},
            )  # type: ignore

            # Don't save this memory, it's a duplicate
            return None, True
    # No duplicates found, return the original memory
    return memory, False


async def deduplicate_by_id(
    memory: MemoryRecord,
    redis_client: Redis | None = None,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> tuple[MemoryRecord | None, bool]:
    """
    Check if a memory with the same ID exists and deduplicate if found.

    When two memories have the same ID, the most recent memory replaces the
    oldest memory. (They are not merged.)

    Args:
        memory: The memory to check for ID duplicates
        redis_client: Optional Redis client
        namespace: Optional namespace filter
        user_id: Optional user ID filter
        session_id: Optional session ID filter

    Returns:
        Tuple of (memory to save (potentially updated), was_overwrite)
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    # If no id, can't deduplicate by id
    if not memory.id:
        return memory, False

    # Use memory vector database to search for memories with the same id
    # Build filter objects
    namespace_filter = None
    if namespace or memory.namespace:
        from agent_memory_server.filters import Namespace

        namespace_filter = Namespace(eq=namespace or memory.namespace)

    user_id_filter = None
    if user_id or memory.user_id:
        from agent_memory_server.filters import UserId

        user_id_filter = UserId(eq=user_id or memory.user_id)

    session_id_filter = None
    if session_id or memory.session_id:
        from agent_memory_server.filters import SessionId

        session_id_filter = SessionId(eq=session_id or memory.session_id)

    # Create id filter
    from agent_memory_server.filters import Id

    id_filter = Id(eq=memory.id)

    # Use memory vector database to search for memories with the same id
    db = await get_memory_vector_db()

    # Search for existing memories with the same id using filter-only query
    # (no embedding required)
    results = await db.list_memories(
        session_id=session_id_filter,
        user_id=user_id_filter,
        namespace=namespace_filter,
        id=id_filter,
        limit=1,  # We only need to know if one exists
    )

    if results.memories and len(results.memories) > 0:
        # Found existing memory with the same id
        existing_memory = results.memories[0]
        logger.info(f"Found existing memory with id {memory.id}, will overwrite")

        # If the existing memory was already persisted, preserve that timestamp
        if existing_memory.persisted_at:
            memory.persisted_at = existing_memory.persisted_at

        # Delete the existing memory using the database
        if existing_memory.id:
            await db.delete_memories([existing_memory.id])

        # Return the memory to be saved (overwriting the existing one)
        return memory, True

    # No existing memory with this id found
    return memory, False


async def detect_similar_memories(
    memory: MemoryRecord,
    distance_threshold: float = 0.2,
) -> list[dict]:
    """
    Find existing memories that are semantically similar to the given memory.

    Unlike deduplicate_by_semantic_search, this function does NOT merge or delete
    anything. It only reports what it finds, letting the caller decide what to do.

    Added 2026-03-10 as part of the memory redesign: detect conflicts instead of
    auto-merging them.

    Args:
        memory: The memory to check for similar existing records
        distance_threshold: Maximum cosine distance to consider "similar" (default 0.2)

    Returns:
        List of dicts with keys: id, text, dist, topics, entities
    """
    if not memory.text:
        return []

    db = await get_memory_vector_db()

    namespace_filter = None
    user_id_filter = None
    if memory.namespace:
        namespace_filter = Namespace(eq=memory.namespace)
    if memory.user_id:
        user_id_filter = UserId(eq=memory.user_id)

    try:
        search_result = await db.search_memories(
            query=memory.text,
            namespace=namespace_filter,
            user_id=user_id_filter,
            distance_threshold=distance_threshold,
            limit=5,
        )
    except Exception as e:
        logger.warning(f"Conflict detection search failed for memory {memory.id}: {e}")
        return []

    if not search_result or not search_result.memories:
        return []

    # Filter out the memory itself
    similar = []
    for m in search_result.memories:
        if m.id == memory.id:
            continue
        similar.append(
            {
                "id": m.id,
                "text": m.text[:500] if m.text else "",
                "dist": m.dist,
                "topics": m.topics,
                "entities": m.entities,
            }
        )

    return similar


async def deduplicate_by_semantic_search(
    memory: MemoryRecord,
    redis_client: Redis | None = None,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    vector_distance_threshold: float | None = None,
) -> tuple[MemoryRecord | None, bool]:
    """
    Check if a memory has semantic duplicates and merge if found.

    Unlike deduplicate_by_id, this function does not overwrite any existing
    memories. Instead, all semantically similar duplicates are merged.

    Uses vector similarity search to find semantically similar memories.
    The distance threshold determines how similar memories must be to be
    considered duplicates. A threshold of 0.35 works well for catching
    paraphrased content while avoiding false positives.

    Args:
        memory: The memory to check for semantic duplicates
        redis_client: Optional Redis client
        namespace: Optional namespace filter
        user_id: Optional user ID filter
        session_id: Optional session ID filter
        vector_distance_threshold: Distance threshold for semantic similarity.
            If None, uses settings.deduplication_distance_threshold (default 0.35)

    Returns:
        Tuple of (memory to save (potentially merged), was_merged)
    """
    # Master switch: skip all semantic dedup if disabled in config.
    # Disabled 2026-03-10 — LLM-based merge was destructive and non-deterministic.
    # Hash-based dedup (layers 1 and 2) remain active in index_long_term_memories().
    if not settings.semantic_dedup_enabled:
        logger.debug(
            "Semantic dedup disabled (settings.semantic_dedup_enabled=False), skipping",
        )
        return memory, False

    # Skip semantic deduplication for memories with empty text
    # OpenAI's embedding API rejects empty strings with "'$.input' is invalid"
    if not memory.text:
        logger.debug(
            "Skipping semantic deduplication for memory with empty text",
            memory_id=memory.id,
        )
        return memory, False

    if not redis_client:
        redis_client = await get_redis_conn()

    # Use memory vector database to find semantically similar memories
    db = await get_memory_vector_db()

    # Get threshold from settings if not provided
    if vector_distance_threshold is None:
        vector_distance_threshold = settings.deduplication_distance_threshold

    # Convert filters to database format
    namespace_filter = None
    user_id_filter = None
    session_id_filter = None

    if namespace or memory.namespace:
        namespace_filter = Namespace(eq=namespace or memory.namespace)
    if user_id or memory.user_id:
        user_id_filter = UserId(eq=user_id or memory.user_id)
    # Only filter by session_id if explicitly provided — do NOT fall back to
    # memory.session_id.  Semantic dedup must search across sessions so that
    # compaction can merge identical memories created in different sessions.
    if session_id:
        session_id_filter = SessionId(eq=session_id)

    # Use the memory vector database for semantic search
    # TODO: Paginate through results?
    search_result = await db.search_memories(
        query=memory.text,  # Use memory text for semantic search
        namespace=namespace_filter,
        user_id=user_id_filter,
        session_id=session_id_filter,
        distance_threshold=vector_distance_threshold,
        limit=10,
    )

    vector_search_result = search_result.memories if search_result else []

    # Filter out the memory itself from the search results (avoid self-duplication)
    vector_search_result = [m for m in vector_search_result if m.id != memory.id]

    # Size guard: skip merge if the incoming memory is already oversized
    input_text_len = len(memory.text) if memory.text else 0
    input_entity_count = len(memory.entities) if memory.entities else 0
    if input_text_len > MAX_MEMORY_INPUT_CHARS or input_entity_count > MAX_ENTITY_COUNT:
        logger.info(
            f"Skipping semantic dedup for oversized memory: "
            f"{input_text_len} chars, {input_entity_count} entities"
        )
        return memory, False

    # Size guard: filter out oversized similar memories from merge candidates
    vector_search_result = [
        m
        for m in vector_search_result
        if (len(m.text) if m.text else 0) <= MAX_MEMORY_INPUT_CHARS
        and (len(m.entities) if m.entities else 0) <= MAX_ENTITY_COUNT
    ]

    if vector_search_result and len(vector_search_result) > 0:
        # Found semantically similar memories
        similar_memory_ids = [m.id for m in vector_search_result]

        # Merge the memories
        merged_memory = await merge_memories_with_llm(
            [memory] + vector_search_result,
        )

        # If merge was rejected (returned original memory), skip deletion
        if merged_memory.id == memory.id:
            logger.info("Merge was rejected by size guard, keeping all memories as-is")
            return memory, False

        # Delete the similar memories using the database
        if similar_memory_ids:
            await db.delete_memories(similar_memory_ids)

        logger.info(
            f"Merged new memory with {len(similar_memory_ids)} semantic duplicates"
        )
        return merged_memory, True

    # No similar memories found or error occurred
    return memory, False


async def promote_working_memory_to_long_term(
    session_id: str,
    namespace: str | None = None,
    user_id: str | None = None,
    redis_client: Redis | None = None,
) -> int:
    """
    Promote eligible working memory records to long-term storage.

    This function:
    1. Identifies memory records with no persisted_at from working memory
    2. For message records, runs extraction to generate semantic/episodic memories
    3. Uses id to detect and replace duplicates in long-term memory
    4. Persists the record and stamps it with persisted_at = now()
    5. Updates the working memory session store to reflect new timestamps

    Args:
        session_id: The session ID to promote memories from
        namespace: Optional namespace for the session
        user_id: Optional user ID for the session
        redis_client: Optional Redis client to use

    Returns:
        Number of memories promoted to long-term storage
    """

    from agent_memory_server import working_memory
    from agent_memory_server.utils.redis import get_redis_conn

    redis = redis_client or await get_redis_conn()

    # Get current working memory
    current_working_memory = await working_memory.get_working_memory(
        session_id=session_id,
        namespace=namespace,
        user_id=user_id,
        redis_client=redis,
    )

    if not current_working_memory:
        logger.debug(f"No working memory found for session {session_id}")
        return 0

    logger.info("Promoting memories to long-term storage...")

    promoted_count = 0
    updated_memories = []

    # Derive session-level attribution from existing working memory records.
    # This propagates attribution context to child memories extracted from
    # conversation messages in this session.
    wm_memory_records = [
        m for m in current_working_memory.memories if isinstance(m, MemoryRecord)
    ]
    session_source_user, session_source_channel, session_visibility = (
        _resolve_parent_attribution(wm_memory_records)
        if wm_memory_records
        else (None, None, "everyone")
    )

    # If attribution wasn't found on working memory records (gateway doesn't
    # set source_user), try to infer from the session ID peer ID.
    if session_source_user is None:
        from agent_memory_server.extraction import resolve_user_from_session_id

        inferred = resolve_user_from_session_id(session_id)
        if inferred:
            session_source_user = inferred
            logger.info(
                f"Inferred session_source_user={inferred} from session ID "
                f"{session_id} (working memory had no attribution)"
            )
        # Also try to infer channel from session ID
        if session_source_channel is None and session_id:
            parts = session_id.split(":")
            # Pattern: agent:main:discord:direct:peerId
            if len(parts) >= 3:
                session_source_channel = parts[2]

    # Thread-aware discrete memory extraction with trailing-edge debouncing
    # Instead of extracting immediately, we schedule extraction to run after
    # a period of inactivity. Each new message resets the timer.
    unextracted_messages = [
        message
        for message in current_working_memory.messages
        if message.discrete_memory_extracted == "f"
    ]

    if settings.enable_discrete_memory_extraction and unextracted_messages:
        # Check if we're not in post-extraction debounce
        if await should_extract_session_thread(session_id, redis):
            if settings.use_docket:
                # When using Docket, schedule via the queue for worker processing
                await schedule_trailing_extraction(
                    session_id=session_id,
                    namespace=namespace,
                    user_id=user_id,
                    redis=redis,
                    source_user=session_source_user,
                    source_channel=session_source_channel,
                    visibility=session_visibility,
                )
            else:
                # When running in-process (no worker), run extraction inline
                # with a short debounce sleep. This is already in a background
                # task so the sleep won't block the HTTP response.
                import asyncio

                debounce = settings.extraction_debounce_seconds
                logger.info(
                    f"Running inline extraction for session {session_id} "
                    f"(debounce: {debounce}s)"
                )
                await asyncio.sleep(debounce)
                extraction_count = await run_delayed_extraction(
                    session_id=session_id,
                    namespace=namespace,
                    user_id=user_id,
                    source_user=session_source_user,
                    source_channel=session_source_channel,
                    visibility=session_visibility,
                )
                logger.info(
                    f"Inline extraction completed for session {session_id}: "
                    f"{extraction_count} memories extracted"
                )
        else:
            logger.info(
                f"Skipping extraction scheduling for session {session_id} - in post-extraction debounce"
            )

    # Process existing memories for promotion (extracted memories are handled by the delayed task)
    all_memories_to_process = list(current_working_memory.memories)

    for memory in all_memories_to_process:
        if memory.persisted_at is None:
            # This memory needs to be promoted

            # Apply session-level attribution to promoted memories that lack it.
            # The gateway plugin doesn't pass source_user/visibility on working
            # memory records, so we backfill from the session-resolved values.
            if not memory.source_user and session_source_user:
                memory.source_user = session_source_user
            if not memory.source_channel and session_source_channel:
                memory.source_channel = session_source_channel
            if (
                not memory.visibility or memory.visibility == "everyone"
            ) and session_visibility != "everyone":
                memory.visibility = session_visibility

            # Check for id-based duplicates and handle accordingly
            deduped_memory, was_overwrite = await deduplicate_by_id(
                memory=memory,
                redis_client=redis,
            )

            # Set persisted_at timestamp
            current_memory = deduped_memory or memory
            current_memory.persisted_at = datetime.now(UTC)

            # Set extraction strategy configuration from working memory
            current_memory.extraction_strategy = (
                current_working_memory.long_term_memory_strategy.strategy
            )
            current_memory.extraction_strategy_config = (
                current_working_memory.long_term_memory_strategy.config
            )

            # Index the memory in long-term storage
            # Fix for Issue #110 - this path previously bypassed deduplication
            await index_long_term_memories(
                [current_memory],
                redis_client=redis,
                deduplicate=True,  # Enable hash and semantic deduplication
            )

            promoted_count += 1
            updated_memories.append(current_memory)

            if was_overwrite:
                logger.info(f"Overwrote existing memory with id {memory.id}")
            else:
                logger.info(f"Promoted new memory with id {memory.id}")
        else:
            # This memory is already persisted, keep as-is
            updated_memories.append(memory)

    count_persisted_messages = 0
    message_records_to_index = []

    # Process unpersisted messages if configured to do so
    if settings.index_all_messages_in_long_term_memory:
        updated_messages = []
        for msg in current_working_memory.messages:
            if msg.persisted_at is None:
                # Skip messages with empty or None content
                if not msg.content or not msg.content.strip():
                    logger.warning(f"Skipping message with empty content: {msg.id}")
                    updated_messages.append(msg)
                    continue

                # Generate ID if not present (backward compatibility)
                if not msg.id:
                    msg.id = str(ULID())

                memory_record = MemoryRecord(
                    id=msg.id,
                    session_id=session_id,
                    text=f"{msg.role}: {msg.content}",
                    namespace=namespace,
                    user_id=current_working_memory.user_id,
                    persisted_at=None,
                    created_at=msg.created_at,
                    memory_type=MemoryTypeEnum.MESSAGE,
                    source_user=session_source_user,
                    source_channel=session_source_channel,
                    visibility=session_visibility,
                )

                # Apply same deduplication logic as structured memories
                deduped_memory, was_overwrite = await deduplicate_by_id(
                    memory=memory_record,
                    redis_client=redis,
                )

                # Set persisted_at timestamp
                current_memory = deduped_memory or memory_record
                current_memory.persisted_at = datetime.now(UTC)

                # Set extraction strategy configuration from working memory
                current_memory.extraction_strategy = "message"

                # Collect memory record for batch indexing
                message_records_to_index.append(current_memory)

                # Update message with persisted_at timestamp
                msg.persisted_at = current_memory.persisted_at
                promoted_count += 1

                if was_overwrite:
                    logger.info(
                        f"Overwrote existing long-term message memory with ID {msg.id}"
                    )
                else:
                    logger.info(
                        f"Promoted new long-term message memory with ID {msg.id}"
                    )

            updated_messages.append(msg)

        # Batch index all new memory records for messages
        # Fix for Issue #110 - this path previously bypassed deduplication
        if message_records_to_index:
            count_persisted_messages = len(message_records_to_index)
            await index_long_term_memories(
                message_records_to_index,
                redis_client=redis,
                deduplicate=True,  # Enable hash and semantic deduplication
            )
    else:
        count_persisted_messages = 0
        updated_messages = current_working_memory.messages

    # Update working memory with the new persisted_at timestamps
    # Note: Extraction now happens asynchronously via trailing-edge debounce
    if promoted_count > 0 or count_persisted_messages > 0:
        updated_working_memory = current_working_memory.model_copy()
        updated_working_memory.memories = updated_memories
        updated_working_memory.messages = updated_messages
        updated_working_memory.updated_at = datetime.now(UTC)

        await working_memory.set_working_memory(
            working_memory=updated_working_memory,
            redis_client=redis,
        )

        logger.info(
            f"Successfully promoted {promoted_count} memories and {len(message_records_to_index)} messages to long-term storage"
        )

    return promoted_count


def _reference_protection_active(caller_trust_level: TrustLevel | None) -> bool:
    """Return True iff the reference-protection guard should run for this call.

    The guard runs only when ALL hold: it is enabled in config, an operator
    token is actually configured (otherwise no record can be OPERATOR-tier — the
    dormant default), a caller tier was supplied (an internal/library call passes
    None and is trusted), and that caller is not already OPERATOR (which outranks
    every record, so the per-record fetch would be wasted work).

    Gating on ``memory_operator_token`` keeps normal agent deletes free of the
    extra per-id fetch until an operator deliberately opts in.
    """
    return (
        settings.memory_reference_protection_enabled
        and bool(settings.memory_operator_token)
        and caller_trust_level is not None
        and caller_trust_level is not TrustLevel.OPERATOR
    )


async def delete_long_term_memories(
    ids: list[str],
    *,
    caller_trust_level: TrustLevel | None = None,
) -> int:
    """
    Delete long-term memories by ID.

    Args:
        ids: Memory IDs to delete.
        caller_trust_level: Server-resolved trust tier of the caller (C1). When
            provided and the reference-protection guard is active, each target is
            fetched and the whole operation is refused (no record deleted) if any
            target out-ranks the caller — raising :class:`ReferenceProtectedError`.
            ``None`` (the default, for internal/library callers) skips the guard.
    """
    if _reference_protection_active(caller_trust_level):
        blocked: list[str] = []
        for memory_id in ids:
            target = await get_long_term_memory_by_id(memory_id)
            # A missing target is not protected — let the delete proceed (it is a
            # no-op for that id) so the guard never masks a 404-shaped result.
            if target is not None and is_reference_protected_mutation(
                caller=caller_trust_level, record=target
            ):
                blocked.append(memory_id)
        if blocked:
            record_counter(
                "memory_server.reference_protection.blocked",
                value=float(len(blocked)),
                attributes={"op": "delete", "caller": caller_trust_level.value},
            )
            logger.warning(
                f"Reference-protected delete refused: caller="
                f"{caller_trust_level.value}, blocked_ids={blocked}"
            )
            raise ReferenceProtectedError(blocked, caller_trust_level)

    db = await get_memory_vector_db()
    return await db.delete_memories(ids)


async def delete_invalid_memories(
    redis_client: Redis | None = None,
) -> int:
    """
    Delete corrupted memory records from Redis.

    Corrupted records are those that:
    1. Have empty key ID components (keys like "prefix:" or "prefix: ")
    2. Are missing required fields (like 'text')

    These records cannot be properly searched or deleted by ID, so they need
    to be cleaned up directly via Redis key operations.

    Args:
        redis_client: Optional Redis client

    Returns:
        Number of memories deleted
    """
    if not redis_client:
        redis_client = await get_redis_conn()

    prefix = settings.redisvl_index_prefix

    deleted_count = 0
    cursor = 0

    # Scan for all memory keys
    while True:
        cursor, keys = await redis_client.scan(
            cursor=cursor, match=f"{prefix}:*", count=1000
        )

        for key in keys:
            key_str = key.decode("utf-8") if isinstance(key, bytes) else key
            should_delete = False

            # Check 1: Empty ID in key name
            id_part = key_str[len(prefix) + 1 :]  # +1 for the colon
            if not id_part or not id_part.strip():
                should_delete = True
                logger.info(f"Found invalid memory with empty key ID: {key_str}")
            else:
                # Check 2: Missing required 'text' field
                data = await redis_client.hgetall(key)
                text_field = data.get(b"text") or data.get("text")
                if text_field is None:
                    should_delete = True
                    logger.info(f"Found invalid memory missing 'text' field: {key_str}")

            if should_delete:
                await redis_client.delete(key)
                deleted_count += 1

        if cursor == 0:
            break

    if deleted_count > 0:
        logger.info(f"Deleted {deleted_count} corrupted memory records")

    return deleted_count


async def get_long_term_memory_by_id(memory_id: str) -> MemoryRecord | None:
    """
    Get a single long-term memory by its ID.

    Args:
        memory_id: The ID of the memory to retrieve

    Returns:
        MemoryRecord if found, None if not found
    """
    from agent_memory_server.filters import Id

    db = await get_memory_vector_db()

    # Search for the memory by ID using filter-only query (no embedding required)
    results = await db.list_memories(
        limit=1,
        id=Id(eq=memory_id),
    )

    if results.memories:
        return results.memories[0]
    return None


async def update_long_term_memory(
    memory_id: str,
    updates: dict[str, Any],
) -> MemoryRecord | None:
    """
    Update a long-term memory by ID.

    Args:
        memory_id: The ID of the memory to update
        updates: Dictionary of fields to update

    Returns:
        Updated MemoryRecord if found and updated, None if not found

    Raises:
        ValueError: If the update contains invalid fields
    """
    # First, get the existing memory
    existing_memory = await get_long_term_memory_by_id(memory_id)
    if not existing_memory:
        return None

    # Valid fields that can be updated
    updatable_fields = {
        "text",
        "topics",
        "entities",
        "memory_type",
        "namespace",
        "user_id",
        "session_id",
        "event_date",
        "source_user",
        "source_channel",
        "visibility",
        "stale_after",
        # Client-settable provenance (valid_to / superseded_by are server-managed
        # via supersede_memory, deliberately excluded here).
        "kind",
        "confidence",
        "derived_from",
        "observed_at",
        "valid_from",
    }

    # Validate update fields
    invalid_fields = set(updates.keys()) - updatable_fields
    if invalid_fields:
        raise ValueError(
            f"Cannot update fields: {invalid_fields}. Valid fields: {updatable_fields}"
        )

    # Create updated memory record using efficient model_copy and hash helper
    base_updates = {**updates, "updated_at": datetime.now(UTC)}
    update_dict = update_memory_hash_if_text_changed(existing_memory, base_updates)
    updated_memory = existing_memory.model_copy(update=update_dict)

    # Update in the database
    db = await get_memory_vector_db()
    await db.update_memories([updated_memory])

    return updated_memory


# Outcome statuses for supersede_memory (mirrors further-memory's contract).
SUPERSEDE_OK = "superseded"
SUPERSEDE_IDEMPOTENT = "idempotent"
SUPERSEDE_TARGET_MISSING = "target_missing"
SUPERSEDE_REPLACEMENT_MISSING = "replacement_missing"
SUPERSEDE_SELF = "self_supersede"
SUPERSEDE_CONFLICT = "conflict"
SUPERSEDE_PROTECTED = "reference_protected"


async def supersede_memory(
    target_id: str,
    replacement_id: str,
    *,
    require_replacement_exists: bool = True,
    force: bool = False,
    caller_trust_level: TrustLevel | None = None,
) -> tuple[str, MemoryRecord | None]:
    """Mark a memory as superseded by a newer version (non-destructive versioning).

    Ported from wfr-memory-commons' ``POST /v1/memories/{id}/supersede``. This is
    the "link, don't merge" alternative to LLM merge that the fork's design
    forbids: the old record is RETAINED in Redis, stamped with
    ``superseded_by = replacement_id`` and ``valid_to = now``, and hidden from
    default recall (``include_superseded=True`` surfaces it). The replacement is
    a separate, already-stored record.

    Args:
        target_id: ID of the record being superseded (the old version).
        replacement_id: ID of the newer record that replaces it.
        require_replacement_exists: When True (default), 404 if the replacement
            is not present. Set False to allow forward-references.
        force: When True, overwrite an existing supersede link (the default
            refuses, returning ``conflict``, to keep versioning monotonic).
        caller_trust_level: Server-resolved trust tier of the caller (C1). When
            the reference-protection guard is active and the caller ranks below
            the target, returns ``SUPERSEDE_PROTECTED`` and does not mutate.
            ``None`` (internal callers) skips the guard.

    Returns:
        ``(status, updated_record_or_None)`` where status is one of the
        ``SUPERSEDE_*`` constants.
    """
    if target_id == replacement_id:
        return SUPERSEDE_SELF, None

    target = await get_long_term_memory_by_id(target_id)
    if not target:
        return SUPERSEDE_TARGET_MISSING, None

    # Reference-record write protection (C1): a lower-trust caller cannot
    # supersede a higher-trust (canonical) record. Checked after the target is
    # fetched (so a missing target still 404s, not 403). Dormant unless an
    # operator token is configured + a caller tier was resolved.
    if _reference_protection_active(caller_trust_level) and (
        is_reference_protected_mutation(caller=caller_trust_level, record=target)
    ):
        record_counter(
            "memory_server.reference_protection.blocked",
            value=1.0,
            attributes={"op": "supersede", "caller": caller_trust_level.value},
        )
        logger.warning(
            f"Reference-protected supersede refused: caller="
            f"{caller_trust_level.value}, target_id={target_id}"
        )
        return SUPERSEDE_PROTECTED, target

    if require_replacement_exists:
        replacement = await get_long_term_memory_by_id(replacement_id)
        if not replacement:
            return SUPERSEDE_REPLACEMENT_MISSING, None

    # Already superseded? Idempotent if pointing at the same replacement;
    # conflict otherwise (unless force overrides).
    if target.superseded_by:
        if target.superseded_by == replacement_id:
            return SUPERSEDE_IDEMPOTENT, target
        if not force:
            return SUPERSEDE_CONFLICT, target

    now = datetime.now(UTC)
    updated = target.model_copy(
        update={
            "superseded_by": replacement_id,
            "valid_to": now,
            "updated_at": now,
        }
    )
    db = await get_memory_vector_db()
    await db.update_memories([updated])
    logger.info(
        f"Superseded memory {target_id} -> {replacement_id} (valid_to={now.isoformat()})"
    )
    return SUPERSEDE_OK, updated


def _is_numeric(value: Any) -> bool:
    """Check if a value is numeric (int, float, or other number type)."""
    return isinstance(value, numbers.Number)


def _parse_stale_after(value: Any) -> datetime | None:
    """Coerce a stale_after value to a timezone-aware datetime.

    Accepts:
      - ``datetime`` (returned as-is, promoted to UTC if naive)
      - ``float`` / ``int`` (UNIX timestamp → UTC datetime)
      - ``str`` (ISO-8601 → UTC datetime)
      - ``None`` (pass-through)
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, int | float):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


def select_ids_for_forgetting(
    results: Iterable[MemoryRecordResult],
    *,
    policy: dict,
    now: datetime,
    pinned_ids: set[str] | None = None,
) -> list[str]:
    """Select IDs for deletion based on TTL, inactivity, stale_after, and budget policies.

    Policy keys:
      - max_age_days: float | None
      - max_inactive_days: float | None
      - budget: int | None (keep top N by recency score)
      - memory_type_allowlist: set[str] | list[str] | None (only consider these types for deletion)
      - hard_age_multiplier: float (default 12.0) - multiplier for max_age_days to determine extremely old items
    """
    pinned_ids = pinned_ids or set()
    max_age_days = policy.get("max_age_days")
    max_inactive_days = policy.get("max_inactive_days")
    hard_age_multiplier = float(policy.get("hard_age_multiplier", 12.0))
    budget = policy.get("budget")
    allowlist = policy.get("memory_type_allowlist")
    if allowlist is not None and not isinstance(allowlist, set):
        allowlist = set(allowlist)

    stale_cleanup = settings.stale_after_cleanup_enabled

    to_delete: set[str] = set()
    eligible_for_budget: list[MemoryRecordResult] = []

    for mem in results:
        if not mem.id or mem.id in pinned_ids or getattr(mem, "pinned", False):
            continue

        # If allowlist provided, only consider those types for deletion
        mem_type_value = (
            mem.memory_type.value
            if isinstance(mem.memory_type, MemoryTypeEnum)
            else mem.memory_type
        )
        if allowlist is not None and mem_type_value not in allowlist:
            # Not eligible for deletion under current policy
            continue

        # stale_after policy: delete memories past their stale_after datetime
        if stale_cleanup:
            stale_dt = _parse_stale_after(getattr(mem, "stale_after", None))
            if stale_dt is not None and now >= stale_dt:
                to_delete.add(mem.id)
                continue

        age_days = _days_between(now, mem.created_at)
        inactive_days = _days_between(now, mem.last_accessed)

        # Combined TTL/inactivity policy:
        # - If both thresholds are set, prefer not to delete recently accessed
        #   items unless they are extremely old.
        # - Extremely old: age > max_age_days * hard_age_multiplier (default 12x)
        if _is_numeric(max_age_days) and _is_numeric(max_inactive_days):
            if age_days > float(max_age_days) * hard_age_multiplier:
                to_delete.add(mem.id)
                continue
            if age_days > float(max_age_days) and inactive_days > float(
                max_inactive_days
            ):
                to_delete.add(mem.id)
                continue
        else:
            ttl_hit = _is_numeric(max_age_days) and age_days > float(max_age_days)
            inactivity_hit = _is_numeric(max_inactive_days) and (
                inactive_days > float(max_inactive_days)
            )
            if ttl_hit or inactivity_hit:
                to_delete.add(mem.id)
                continue

        # Eligible for budget consideration
        eligible_for_budget.append(mem)

    # Budget-based pruning (keep top N by recency among eligible)
    if isinstance(budget, int) and budget >= 0 and budget < len(eligible_for_budget):
        params = {
            "semantic_weight": 0.0,  # budget considers only recency
            "recency_weight": 1.0,
            "freshness_weight": 0.6,
            "novelty_weight": 0.4,
            "half_life_last_access_days": 14.0,
            "half_life_created_days": 30.0,
        }
        ranked = rerank_with_recency(eligible_for_budget, now=now, params=params)
        keep_ids = {mem.id for mem in ranked[:budget]}
        for mem in eligible_for_budget:
            if mem.id not in keep_ids:
                to_delete.add(mem.id)

    return list(to_delete)


async def update_last_accessed(
    ids: list[str],
    *,
    redis_client: Redis | None = None,
    min_interval_seconds: int = 900,
) -> int:
    """Rate-limited update of last_accessed for a list of memory IDs.

    Returns the number of records updated.
    """
    if not ids:
        return 0

    redis = redis_client or await get_redis_conn()
    now_ts = int(datetime.now(UTC).timestamp())

    # Batch read existing last_accessed
    keys = [Keys.memory_key(mid) for mid in ids]
    pipeline = redis.pipeline()
    for key in keys:
        pipeline.hget(key, "last_accessed")
    current_vals = await pipeline.execute()

    # Decide which to update and whether to increment access_count
    to_update: list[tuple[str, int]] = []
    incr_keys: list[str] = []
    for key, val in zip(keys, current_vals, strict=False):
        try:
            last_ts = int(val) if val is not None else 0
        except (TypeError, ValueError):
            last_ts = 0
        if now_ts - last_ts >= min_interval_seconds:
            to_update.append((key, now_ts))
            incr_keys.append(key)

    if not to_update:
        return 0

    pipeline2 = redis.pipeline()
    for key, ts in to_update:
        pipeline2.hset(key, mapping={"last_accessed": str(ts)})
        pipeline2.hincrby(key, "access_count", 1)
    await pipeline2.execute()
    return len(to_update)


async def forget_long_term_memories(
    policy: dict,
    *,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = 1000,
    dry_run: bool = True,
    pinned_ids: list[str] | None = None,
) -> dict:
    """Select and delete long-term memories according to policy.

    Uses the memory vector database to fetch candidates (empty query + filters),
    then applies `select_ids_for_forgetting` locally and deletes matching memories.
    """
    db = await get_memory_vector_db()

    # Build filters
    namespace_filter = Namespace(eq=namespace) if namespace else None
    user_id_filter = UserId(eq=user_id) if user_id else None
    session_id_filter = SessionId(eq=session_id) if session_id else None

    # Fetch candidates using filter-only query (no embedding required)
    results = await db.list_memories(
        namespace=namespace_filter,
        user_id=user_id_filter,
        session_id=session_id_filter,
        limit=limit,
    )

    now = datetime.now(UTC)
    candidate_results = results.memories or []

    # Select IDs for deletion using policy
    to_delete_ids = select_ids_for_forgetting(
        candidate_results,
        policy=policy,
        now=now,
        pinned_ids=set(pinned_ids) if pinned_ids else None,
    )

    deleted = 0
    if to_delete_ids and not dry_run:
        deleted = await db.delete_memories(to_delete_ids)

    return {
        "scanned": len(candidate_results),
        "deleted": deleted if not dry_run else len(to_delete_ids),
        "deleted_ids": to_delete_ids,
        "dry_run": dry_run,
    }


async def periodic_forget_long_term_memories(
    *,
    namespace: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
    limit: int = 1000,
    dry_run: bool = False,
    perpetual: Perpetual = Perpetual(
        every=timedelta(minutes=settings.forgetting_every_minutes), automatic=True
    ),
) -> dict:
    """Periodic forgetting using defaults from settings.

    This function can be registered with Docket and will run automatically
    according to the `perpetual` schedule when a worker is active.
    """
    # Build default policy from settings
    policy: dict[str, object] = {
        "max_age_days": settings.forgetting_max_age_days,
        "max_inactive_days": settings.forgetting_max_inactive_days,
        "budget": settings.forgetting_budget_keep_top_n,
        "memory_type_allowlist": None,
    }

    # If feature disabled, no-op
    if not settings.forgetting_enabled:
        logger.info("Forgetting is disabled; skipping periodic run")
        return {"scanned": 0, "deleted": 0, "deleted_ids": [], "dry_run": True}

    return await forget_long_term_memories(
        policy,
        namespace=namespace,
        user_id=user_id,
        session_id=session_id,
        limit=limit,
        dry_run=dry_run,
    )

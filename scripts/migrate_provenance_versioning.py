#!/usr/bin/env python3
"""Migration: activate provenance & versioning fields on the live memory index.

Ported alongside the wfr-memory-commons provenance/versioning feature. This is a
DELIBERATE, count-safeguarded, one-time migration — it is NOT auto-run on
startup (the homelab has ~16.8k live records; an unattended index rebuild is not
worth the risk).

What it does, in order:

  1. Snapshot the current record count (safety baseline).
  2. Backfill MISSING provenance/versioning hash fields on existing records so
     opt-in filters behave correctly over legacy data:
       - confidence_idx → CONFIDENCE_UNSCORED_SENTINEL (2.0) when absent, so a
         min_confidence floor never silently drops a pre-existing (unscored) record.
       - valid_to_ts    → VALID_TO_SENTINEL (far future) when absent ("still valid").
       - observed_at / valid_from → created_at when absent (populate the window).
     The backfill is idempotent — it only writes fields that are missing, so
     re-running is safe and a no-op once complete.
  3. Re-verify the record count is unchanged (abort if it dropped — mirrors the
     memory-maintenance count safeguard).
  4. Rebuild the search index (FT.DROPINDEX + FT.CREATE via RedisVL
     overwrite=True; documents are preserved) so the new TAG/NUMERIC fields
     (kind, superseded_by, confidence_idx, valid_to_ts) become queryable. Until
     this step runs, the kind / min_confidence QUERY FILTERS are inert — but
     storage, field round-trip, and supersede-hiding already work without it
     (FT.SEARCH RETURN reads hash fields regardless of indexing).

DRY-RUN by default. Pass --execute to apply.

IMPORTANT — run with the SAME embedding env as the memory server, or the index
rebuild recreates the vector field at the wrong dimension (the schema falls back
to text-embedding-3-small=1536) and orphans every record. The deployment uses
ollama/nomic-embed-text (768). A dim-safety guard aborts the rebuild if the
schema dim != the live index dim (incident 2026-06-19), but set the env anyway.

Usage (deployed runtime venv):
  cd ~/Developer/homelab/openclaw/agent-memory-server-fork
  EMBEDDING_MODEL=ollama/nomic-embed-text REDISVL_VECTOR_DIMENSIONS=768 \\
  OLLAMA_API_BASE=http://localhost:11434 \\
  REDIS_URL="redis://:<password>@localhost:6379" \\
    ../agent-memory-server/venv/bin/python scripts/migrate_provenance_versioning.py
  # then, to apply, append --execute to the same command.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from agent_memory_server.config import settings
from agent_memory_server.memory_vector_db import (
    CONFIDENCE_UNSCORED_SENTINEL,
    VALID_TO_SENTINEL,
)
from agent_memory_server.memory_vector_db_factory import (
    _get_embedding_dimensions,
    get_memory_vector_db,
)
from agent_memory_server.utils.redis import get_redis_conn


async def _existing_index_vector_dim(redis, index_name: str) -> int | None:
    """Return the vector DIM of the live index, or None if absent/unparseable.

    Guards against a catastrophic dimension change: if the migration runs without
    the deployment's EMBEDDING_MODEL set, the schema falls back to a default model
    (text-embedding-3-small=1536) and a rebuild would recreate the index at the
    wrong dim, orphaning every record (hash_indexing_failures = all). Incident
    2026-06-19.
    """
    try:
        raw = await redis.execute_command("FT.INFO", index_name)
    except Exception:
        return None
    d = {}
    for i in range(0, len(raw) - 1, 2):
        k = raw[i].decode() if isinstance(raw[i], bytes) else raw[i]
        d[k] = raw[i + 1]
    for attr in d.get("attributes", []):
        parts = [p.decode() if isinstance(p, bytes) else p for p in attr]
        for j, x in enumerate(parts):
            if str(x).upper() == "DIM":
                try:
                    return int(parts[j + 1])
                except (ValueError, IndexError, TypeError):
                    return None
    return None


# Fields backfilled when missing: hash field -> ("constant", value) or
# ("copy_from", source_field).
_BACKFILL = {
    "confidence_idx": ("constant", CONFIDENCE_UNSCORED_SENTINEL),
    "valid_to_ts": ("constant", VALID_TO_SENTINEL),
    "observed_at": ("copy_from", "created_at"),
    "valid_from": ("copy_from", "created_at"),
}

# Abort if the post-backfill count drops by more than this fraction of the
# baseline (mirrors memory-maintenance.py's 5% guard).
_MAX_COUNT_DROP_FRACTION = 0.05


async def _scan_keys(redis, prefix: str):
    """Yield all record hash keys under the index prefix."""
    pattern = f"{prefix}:*"
    cursor = 0
    while True:
        cursor, batch = await redis.scan(cursor=cursor, match=pattern, count=500)
        for key in batch:
            yield key
        if cursor == 0:
            break


async def _count_keys(redis, prefix: str) -> int:
    n = 0
    async for _ in _scan_keys(redis, prefix):
        n += 1
    return n


async def run(execute: bool, force_dim_change: bool = False) -> int:
    prefix = settings.redisvl_index_prefix
    index_name = settings.redisvl_index_name
    redis = await get_redis_conn()

    baseline = await _count_keys(redis, prefix)
    print(f"[migrate] index={index_name} prefix={prefix}:* records={baseline}")
    if baseline == 0:
        print("[migrate] no records found — nothing to do")
        return 0

    # --- Phase 1: backfill missing fields -----------------------------------
    scanned = 0
    would_write = 0
    written = 0
    per_field: dict[str, int] = dict.fromkeys(_BACKFILL, 0)

    def _dk(b):
        # Field NAMES are always UTF-8; safe to decode.
        return b.decode() if isinstance(b, bytes) else b

    def _dv(b):
        # Field VALUES may be non-UTF-8 binary — the `vector` field is a packed
        # float32 blob. We only ever read ASCII timestamp/text values here, so
        # decode defensively (errors="replace") and never touch the vector.
        return b.decode("utf-8", "replace") if isinstance(b, bytes) else b

    async for key in _scan_keys(redis, prefix):
        scanned += 1
        raw = await redis.hgetall(key)
        # Map decoded field name -> RAW value. Keys decoded (presence check);
        # values left raw so the binary vector blob is never UTF-8-decoded.
        present = {_dk(k): v for k, v in raw.items()}
        to_set: dict[str, float | str] = {}
        for field, (mode, ref) in _BACKFILL.items():
            if field in present:
                continue
            if mode == "constant":
                to_set[field] = ref
            else:  # copy_from another field (e.g. created_at — ASCII)
                src_raw = present.get(ref)
                if src_raw is None:
                    continue  # no source value (rare/corrupt record) — skip
                to_set[field] = _dv(src_raw)
            per_field[field] += 1
        if not to_set:
            continue
        would_write += 1
        if execute:
            await redis.hset(key, mapping=to_set)
            written += 1

    print(
        f"[migrate] scanned={scanned} records_needing_backfill={would_write} "
        f"per_field={per_field}"
    )

    if not execute:
        print("[migrate] DRY-RUN — no writes. Re-run with --execute to apply.")
        print("[migrate] (the index rebuild also only runs under --execute)")
        return 0

    print(f"[migrate] backfilled {written} records")

    # --- Phase 2: count safeguard -------------------------------------------
    after = await _count_keys(redis, prefix)
    if after < baseline * (1 - _MAX_COUNT_DROP_FRACTION):
        print(
            f"[migrate] ABORT: record count dropped {baseline} -> {after} "
            f"(> {_MAX_COUNT_DROP_FRACTION:.0%}); NOT rebuilding index",
            file=sys.stderr,
        )
        return 2
    print(f"[migrate] count safeguard OK ({baseline} -> {after})")

    # --- Phase 3: rebuild index ---------------------------------------------
    db = await get_memory_vector_db()
    index = getattr(db, "index", None)
    if index is None:
        print(
            "[migrate] backend has no rebuildable index — skipping rebuild",
            file=sys.stderr,
        )
        return 0
    # Dimension-safety guard (incident 2026-06-19): refuse to rebuild if the
    # schema's vector dim would differ from the live index — a wrong
    # EMBEDDING_MODEL silently recreates the index at the wrong dim and orphans
    # every record. The rebuild MUST run with the same embedding env as the
    # memory server (EMBEDDING_MODEL=ollama/nomic-embed-text → 768).
    existing_dim = await _existing_index_vector_dim(redis, index_name)
    schema_dim = _get_embedding_dimensions()
    print(f"[migrate] vector dim: existing={existing_dim} schema={schema_dim}")
    if existing_dim is not None and existing_dim != schema_dim and not force_dim_change:
        print(
            f"[migrate] ABORT: schema vector dim ({schema_dim}) != live index dim "
            f"({existing_dim}). Rebuilding would orphan ALL records. Re-run with the "
            f"deployment embedding env, e.g.:\n"
            f"    EMBEDDING_MODEL=ollama/nomic-embed-text REDISVL_VECTOR_DIMENSIONS={existing_dim} "
            f"REDIS_URL=... <venv>/python scripts/migrate_provenance_versioning.py --execute\n"
            f"(or pass --force-dim-change if a dimension change is truly intended).",
            file=sys.stderr,
        )
        return 3

    print(f"[migrate] rebuilding index '{index.name}' (overwrite=True, data preserved)")
    await index.create(overwrite=True)
    print("[migrate] index rebuilt — kind / min_confidence filters now active")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Apply the backfill + rebuild (default is dry-run)",
    )
    parser.add_argument(
        "--force-dim-change",
        action="store_true",
        help=(
            "Allow the index rebuild even when the schema vector dim differs "
            "from the live index (DANGEROUS — orphans records unless the data "
            "is also re-embedded). Default refuses the dim change."
        ),
    )
    args = parser.parse_args()
    return asyncio.run(run(args.execute, force_dim_change=args.force_dim_change))


if __name__ == "__main__":
    raise SystemExit(main())

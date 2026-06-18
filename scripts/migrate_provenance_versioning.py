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

Usage (deployed runtime venv):
  cd ~/Developer/homelab/openclaw/agent-memory-server-fork
  REDIS_URL="redis://:<password>@localhost:6379" \\
    ../agent-memory-server/venv/bin/python scripts/migrate_provenance_versioning.py
  # then, to apply:
  REDIS_URL=... ../agent-memory-server/venv/bin/python \\
    scripts/migrate_provenance_versioning.py --execute
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
from agent_memory_server.memory_vector_db_factory import get_memory_vector_db
from agent_memory_server.utils.redis import get_redis_conn


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


async def run(execute: bool) -> int:
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

    def _dec(v):
        return v.decode() if isinstance(v, bytes) else v

    async for key in _scan_keys(redis, prefix):
        scanned += 1
        raw = await redis.hgetall(key)
        # Normalize bytes keys/values to str (get_redis_conn may or may not
        # set decode_responses).
        existing = {_dec(k): _dec(v) for k, v in raw.items()}
        to_set: dict[str, float | str] = {}
        for field, (mode, ref) in _BACKFILL.items():
            if field in existing:
                continue
            if mode == "constant":
                to_set[field] = ref
            else:  # copy_from another field (e.g. created_at)
                src_val = existing.get(ref)
                if src_val is None:
                    continue  # no source value (rare/corrupt record) — skip
                to_set[field] = src_val
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
    args = parser.parse_args()
    return asyncio.run(run(args.execute))


if __name__ == "__main__":
    raise SystemExit(main())

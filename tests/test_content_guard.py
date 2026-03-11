"""Tests for the content noise guard in index_long_term_memories().

Verifies that operational noise — tweet logs, meta-memories, analytics
schemas, and monitoring reports — is rejected at the universal write funnel,
preventing it from polluting the memory store.
"""

import pytest

from agent_memory_server.long_term_memory import _is_noise_content


class TestTweetLogNoise:
    """Verify tweet-by-tweet event logs are caught."""

    def test_pat_posted_about(self):
        assert _is_noise_content(
            "On March 9, 2026 at 11:40 AM EDT, @Hi_Its_Pat posted three "
            "original tweets about Everest fossils, National Meatball Day"
        )

    def test_tweet_was_posted(self):
        assert _is_noise_content(
            "On March 9, 2026 at 9:19 AM ET, a tweet was posted to "
            "@Hi_Its_Pat about NFL Free Agency"
        )

    def test_pat_tweeted(self):
        assert _is_noise_content(
            "@Hi_Its_Pat tweeted about the stealth bear market"
        )

    def test_posted_original_tweet(self):
        assert _is_noise_content(
            "Pat posted an original tweet about Mario Day using the "
            "MAR10 calendar joke"
        )

    def test_twitter_strategy_not_noise(self):
        """Twitter strategy/config is valuable — should NOT be caught."""
        assert not _is_noise_content(
            "@Hi_Its_Pat is restricted from replying due to 403 errors. "
            "Original tweets on trending topics are the best strategy."
        )


class TestMetaMemoryNoise:
    """Verify meta-memories about memory system operations are caught."""

    def test_memory_maintenance_ran(self):
        assert _is_noise_content(
            "On March 10, 2026, memory maintenance ran and cleaned "
            "12 duplicate records in phase 2."
        )

    def test_dream_cycle_disabled(self):
        assert _is_noise_content(
            "Chris Baker's household permanently disabled the 2:00 AM "
            "dream cycle in March 2026."
        )

    def test_memory_system_overhaul(self):
        assert _is_noise_content(
            "The memory system overhaul in March 2026 reduced entity "
            "count from 3,606 to under 30 per record."
        )

    def test_mega_memory_decomposition(self):
        assert _is_noise_content(
            "The mega-memory decomposition workflow stored replacement "
            "memories and deleted the original."
        )

    def test_backup_file_stored(self):
        assert _is_noise_content(
            "Chris Baker stored a backup file named "
            "01KKBFXH6VPZ10H46AV873MR6R.json containing the original."
        )

    def test_memories_were_cleaned(self):
        assert _is_noise_content(
            "On March 10, 20 empty memories were cleaned from Redis."
        )

    def test_docket_worker_disabled(self):
        assert _is_noise_content(
            "The docket worker was permanently disabled to prevent "
            "compaction from destroying memories."
        )

    def test_orphan_memories_found(self):
        assert _is_noise_content(
            "5 orphan memory keys were found and cleaned during "
            "nightly maintenance."
        )

    def test_memory_hygiene(self):
        assert _is_noise_content(
            "Memory hygiene is an emerging concern in the OpenClaw "
            "community as of March 2026."
        )

    def test_actual_memory_content_not_noise(self):
        """Real family knowledge should NOT be caught."""
        assert not _is_noise_content(
            "Grant Baker has LADD syndrome, a rare genetic disorder "
            "affecting tear ducts and salivary glands."
        )

    def test_memory_preference_not_noise(self):
        """User preferences about memory behavior are borderline but should pass."""
        assert not _is_noise_content(
            "Chris Baker prefers that Pat ask before storing sensitive "
            "medical information."
        )


class TestAnalyticsSchemaNoise:
    """Verify BigQuery/Dataform schema dumps are caught."""

    def test_table_with_columns(self):
        assert _is_noise_content(
            "The 'facebook_capi_events' table manages event structure "
            "validation with columns for event_name, event_time."
        )

    def test_bigquery_partitioned(self):
        assert _is_noise_content(
            "The 'jitsu_events_parsed' table is partitioned by DATE "
            "and clustered by user_id in BigQuery."
        )

    def test_interconnected_tables(self):
        assert _is_noise_content(
            "The database structure includes three interconnected "
            "tables designed to manage and track user interactions."
        )

    def test_merged_advertising_memory(self):
        assert _is_noise_content(
            "The merged memory covers advertising campaign performance, "
            "email marketing analytics, and SEO metrics."
        )

    def test_meta_ads_schema(self):
        assert _is_noise_content(
            "The meta_ads_performance table includes fields for "
            "report_date, ad_id, campaign_name, spend, impressions."
        )

    def test_users_unified_table(self):
        assert _is_noise_content(
            "The users_unified table has 131 columns and 15,000 rows, "
            "refreshed daily at 2 AM ET."
        )

    def test_property_listing_schema(self):
        assert _is_noise_content(
            "The property_listing_parsed table includes an AI insights "
            "flag which is partitioned by DATE(saved_at)."
        )

    def test_legitimate_data_insight_not_noise(self):
        """Insights about data quality are valuable — should NOT be caught."""
        assert not _is_noise_content(
            "Further's conversion rate dropped 15% last week due to "
            "a broken CTA on the mortgage calculator page."
        )


class TestMonitoringNoise:
    """Verify monitoring status and file-operation noise is caught."""

    def test_no_activity_found(self):
        assert _is_noise_content(
            "Kids Activity Monitor: no activity found for Christian, "
            "Lindalee, or Grant."
        )

    def test_no_raw_scanner(self):
        assert _is_noise_content(
            "No raw-scanner PDFs were found or renamed."
        )

    def test_file_renamed(self):
        assert _is_noise_content(
            "PDF renamed from scan_02182026_180705.pdf to "
            "Grant-Baker-IEP-Progress-2026.pdf"
        )

    def test_all_services_healthy(self):
        assert _is_noise_content(
            "Fleet check: all services healthy, 12 containers running."
        )

    def test_backup_completed(self):
        assert _is_noise_content(
            "Nightly backup completed successfully: Redis RDB + configs "
            "synced to QNAP."
        )

    def test_actual_monitoring_instruction_not_noise(self):
        """Standing instructions about monitoring should NOT be caught."""
        assert not _is_noise_content(
            "Instruction: Time context rules for kids network monitoring. "
            "School nights (Sun-Thu): activity after 9 PM ET is notable."
        )


class TestNoiseGuardEdgeCases:
    """Edge cases and false positive prevention."""

    def test_short_text_not_noise(self):
        assert not _is_noise_content("Grant has an iPad.")

    def test_empty_text_not_noise(self):
        """Empty text is handled by the empty-text guard, not the noise guard."""
        assert not _is_noise_content("")

    def test_family_health_not_noise(self):
        assert not _is_noise_content(
            "Lindsey Baker's max heart rate is 181 bpm (measured). "
            "Her UT1 training zone is 136-154 bpm."
        )

    def test_infrastructure_fact_not_noise(self):
        assert not _is_noise_content(
            "The Mac Mini M4 Pro runs the OpenClaw gateway natively "
            "on port 18789 with Redis on port 6379."
        )

    def test_standing_instruction_not_noise(self):
        assert not _is_noise_content(
            "Chris Baker wants silence-is-good-news: only notify "
            "parents when kids' device activity warrants attention."
        )

    def test_genealogy_not_noise(self):
        assert not _is_noise_content(
            "The Baker/Searcy family Fohr connection runs through "
            "Christine Braren Searcy (Nonnie)."
        )

    def test_work_preference_not_noise(self):
        assert not _is_noise_content(
            "Chris Baker prefers 2-space indentation, single quotes, "
            "and trailing commas in TypeScript."
        )

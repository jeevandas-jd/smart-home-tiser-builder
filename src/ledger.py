"""
This handles our tracking database. If the pipeline hits an API ceiling or crashes at episode 342,
running it again will immediately pick up right at 342 without re-running previous instances.

"""
# src/ledger.py

import sqlite3
import os
from contextlib import contextmanager
from datetime import datetime


class ExecutionLedger:
    def __init__(self, db_path="state.db"):
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.commit()
            conn.close()

    def _init_db(self):
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
            CREATE TABLE IF NOT EXISTS ledger (
                episode_id TEXT PRIMARY KEY,

                status TEXT NOT NULL DEFAULT 'pending',

                attempts INTEGER DEFAULT 0,

                provider TEXT,
                model_name TEXT,

                output_path TEXT,
                checksum TEXT,

                validation_score REAL,

                generation_latency REAL,

                last_error TEXT,

                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """)

    def register_episodes(self, raw_folder_path):

        if not os.path.exists(raw_folder_path):
            os.makedirs(raw_folder_path)
            return

        files = [
            f for f in os.listdir(raw_folder_path)
            if f.endswith(".json")
        ]

        with self._connect() as conn:
            cursor = conn.cursor()

            for f in files:
                episode_id = os.path.splitext(f)[0]

                cursor.execute("""
                INSERT OR IGNORE INTO ledger (
                    episode_id,
                    status,
                    attempts
                )
                VALUES (?, 'pending', 0)
                """, (episode_id,))

    def acquire_next_episode(self):

        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
            SELECT episode_id
            FROM ledger
            WHERE status IN (
                'pending',
                'failed_generation',
                'failed_validation'
            )
            AND attempts < 3
            ORDER BY created_at ASC
            LIMIT 1
            """)

            row = cursor.fetchone()

            if not row:
                return None

            episode_id = row[0]

            cursor.execute("""
            UPDATE ledger
            SET status = 'in_progress',
                updated_at = CURRENT_TIMESTAMP
            WHERE episode_id = ?
            """, (episode_id,))

            return episode_id

    def mark_generation_failure(
        self,
        episode_id,
        error_msg
    ):

        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
            UPDATE ledger
            SET status = 'failed_generation',
                attempts = attempts + 1,
                last_error = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE episode_id = ?
            """, (error_msg, episode_id))

    def mark_validation_failure(
        self,
        episode_id,
        error_msg
    ):

        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
            UPDATE ledger
            SET status = 'failed_validation',
                attempts = attempts + 1,
                last_error = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE episode_id = ?
            """, (error_msg, episode_id))

    def mark_completed(
        self,
        episode_id,
        output_path,
        provider=None,
        model_name=None,
        validation_score=None,
        generation_latency=None
    ):

        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute("""
            UPDATE ledger
            SET status = 'completed',

                output_path = ?,

                provider = ?,
                model_name = ?,

                validation_score = ?,
                generation_latency = ?,

                updated_at = CURRENT_TIMESTAMP

            WHERE episode_id = ?
            """, (
                output_path,
                provider,
                model_name,
                validation_score,
                generation_latency,
                episode_id
            ))

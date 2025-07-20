import logging
from contextlib import contextmanager
from typing import List

import psycopg2
from psycopg2 import pool

from epi_monitor.config.settings import SETTINGS

logger = logging.getLogger(__name__)

# Global connection pool, initialized once at startup
db_pool = None


def initialize_db_pool():
    """Initializes the PostgreSQL connection pool."""
    global db_pool
    try:
        db_pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1, maxconn=5, dsn=SETTINGS.DATABASE_URL
        )
        logger.info("Database connection pool initialized successfully.")
    except psycopg2.OperationalError as e:
        logger.critical(f"Failed to initialize database pool: {e}")
        db_pool = None


def close_db_pool():
    """Closes all connections in the pool."""
    global db_pool
    if db_pool:
        db_pool.closeall()
        logger.info("Database connection pool closed.")


@contextmanager
def get_db_connection():
    """
    Provides a database connection from the pool using a context manager.
    Ensures the connection is returned to the pool automatically.
    """
    if not db_pool:
        raise ConnectionError("Database pool is not initialized or connection failed.")

    conn = None
    try:
        conn = db_pool.getconn()
        yield conn
    finally:
        if conn:
            db_pool.putconn(conn)


def create_events_table():
    """Creates the 'events' table if it doesn't exist."""
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS events (
                        id SERIAL PRIMARY KEY,
                        track_id INTEGER NOT NULL,
                        present_epis TEXT[],
                        missing_epis TEXT[],
                        event_timestamp TIMESTAMP WITH TIME ZONE DEFAULT (NOW() AT TIME ZONE 'UTC'),
                        image_path VARCHAR(255),
                        notification_status VARCHAR(20),
                        camera_id VARCHAR(50)
                    );
                    """
                )
                conn.commit()
                logger.info("Table 'events' is ready.")
    except (psycopg2.Error, ConnectionError) as e:
        logger.error(f"Error creating table: {e}")


def insert_event(
    track_id: int,
    present_epis: List[str],
    missing_epis: List[str],
    image_path: str,
    notification_status: str,
    camera_id: str,
):
    """
    Inserts a non-compliance event into the database using parameterized queries
    to prevent SQL injection.
    """
    sql = """
        INSERT INTO events (track_id, present_epis, missing_epis, image_path, notification_status, camera_id)
        VALUES (%s, %s, %s, %s, %s, %s);
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    sql,
                    (
                        track_id,
                        present_epis,
                        missing_epis,
                        image_path,
                        notification_status,
                        camera_id,
                    ),
                )
                conn.commit()
                logger.info(f"Successfully inserted event for track ID {track_id}.")
    except (psycopg2.Error, ConnectionError) as e:
        logger.error(f"Database insert failed for track ID {track_id}: {e}")
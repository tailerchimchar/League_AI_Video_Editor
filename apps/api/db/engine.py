"""Async database engine using asyncpg connection pool."""

import os

import asyncpg

_pool: asyncpg.Pool | None = None


async def init_db() -> asyncpg.Pool:
    """Create the connection pool. Called once at app startup."""
    global _pool
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    _pool = await asyncpg.create_pool(database_url, min_size=2, max_size=10)
    return _pool


async def close_db() -> None:
    """Close the connection pool. Called at app shutdown."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def get_pool() -> asyncpg.Pool:
    """Return the current connection pool. Raises if not initialized."""
    if _pool is None:
        raise RuntimeError("Database pool not initialized. Call init_db() first.")
    return _pool

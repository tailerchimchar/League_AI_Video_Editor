"""Simple SQL migration runner."""

import asyncio
import glob
import os
import sys
from pathlib import Path

# Add api root to path so we can import db module
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncpg
from dotenv import load_dotenv

_root = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_root / ".env")


async def run_migrations():
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL not set")
        sys.exit(1)

    conn = await asyncpg.connect(database_url)

    # Track applied migrations
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS _migrations (
            filename TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    applied = {row["filename"] for row in await conn.fetch("SELECT filename FROM _migrations")}

    migrations_dir = Path(__file__).parent
    sql_files = sorted(migrations_dir.glob("*.sql"))

    for sql_file in sql_files:
        if sql_file.name in applied:
            print(f"  SKIP  {sql_file.name} (already applied)")
            continue

        print(f"  APPLY {sql_file.name}")
        sql = sql_file.read_text(encoding="utf-8")
        await conn.execute(sql)
        await conn.execute("INSERT INTO _migrations (filename) VALUES ($1)", sql_file.name)

    await conn.close()
    print("Migrations complete.")


if __name__ == "__main__":
    asyncio.run(run_migrations())

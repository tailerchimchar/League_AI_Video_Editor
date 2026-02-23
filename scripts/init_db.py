"""Initialize the database by running all migrations."""

import asyncio
import sys
from pathlib import Path

# Add api to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "apps" / "api"))

from migrations.run import run_migrations


if __name__ == "__main__":
    print("Running database migrations...")
    asyncio.run(run_migrations())

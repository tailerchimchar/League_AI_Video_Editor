"""Search Roboflow Universe for compatible League of Legends detection datasets.

Queries the Roboflow API to find datasets with overlapping class schemas
and shows compatibility analysis with our 16-class system.

Usage:
    python scripts/search_roboflow_datasets.py

Requires ROBOFLOW_API_KEY in .env or environment.
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Our 16-class schema
OUR_CLASSES = {
    "enemy_cannon_minion", "enemy_champion", "enemy_inhibitor",
    "enemy_melee_minion", "enemy_nexus", "enemy_ranged_minion",
    "enemy_tower", "freindly_cannon_minion", "freindly_champion",
    "freindly_melee_minion", "freindly_ranged_minion",
    "freindly_super_minion", "friendly_inhibitor", "friendly_nexus",
    "friendly_tower", "played_champion",
}

# Keywords to search for on Roboflow Universe
SEARCH_QUERIES = [
    "league of legends detection",
    "league of legends object detection",
    "LoL minion champion",
    "league gameplay detection",
]


def load_api_key() -> str:
    """Load Roboflow API key from .env or environment."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key

    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("ROBOFLOW_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    print("ERROR: ROBOFLOW_API_KEY not found in environment or .env")
    print("Get your key from https://app.roboflow.com/settings/api")
    sys.exit(1)


def search_universe(api_key: str):
    """Search Roboflow Universe for compatible datasets."""
    try:
        from roboflow import Roboflow
    except ImportError:
        print("ERROR: roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)

    print("Searching Roboflow Universe for League of Legends datasets...\n")
    print("=" * 70)

    seen_projects = set()

    for query in SEARCH_QUERIES:
        print(f"\nQuery: '{query}'")
        print("-" * 50)

        try:
            results = rf.search(query)
        except Exception as e:
            print(f"  Search failed: {e}")
            continue

        if not results:
            print("  No results found")
            continue

        for result in results:
            project_id = getattr(result, "id", None) or str(result)
            if project_id in seen_projects:
                continue
            seen_projects.add(project_id)

            name = getattr(result, "name", project_id)
            universe_url = getattr(result, "universe_url", "")
            num_images = getattr(result, "images", "?")
            classes = getattr(result, "classes", {})

            if isinstance(classes, dict):
                class_names = set(classes.keys())
            elif isinstance(classes, list):
                class_names = set(classes)
            else:
                class_names = set()

            # Compute compatibility
            overlap = class_names & OUR_CLASSES
            extra = class_names - OUR_CLASSES
            missing = OUR_CLASSES - class_names

            # Compatibility score
            compat = len(overlap) / len(OUR_CLASSES) * 100 if OUR_CLASSES else 0

            print(f"\n  {name}")
            if universe_url:
                print(f"  URL: {universe_url}")
            print(f"  Images: {num_images}")
            print(f"  Classes ({len(class_names)}): {', '.join(sorted(class_names)[:10])}")
            if len(class_names) > 10:
                print(f"    ... and {len(class_names) - 10} more")
            print(f"  Compatibility: {compat:.0f}% ({len(overlap)}/{len(OUR_CLASSES)} classes)")
            if overlap:
                print(f"  Matching: {', '.join(sorted(overlap))}")
            if extra:
                print(f"  Extra classes: {', '.join(sorted(extra)[:8])}")

    print(f"\n{'='*70}")
    print(f"Found {len(seen_projects)} unique projects")
    print("\nTo use a dataset, note its workspace/project ID and use merge_datasets.py")


def main():
    api_key = load_api_key()
    search_universe(api_key)


if __name__ == "__main__":
    main()

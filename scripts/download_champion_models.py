#!/usr/bin/env python3
"""Download champion 3D models (GLB) for ALL skins from modelviewer.lol CDN.

Usage:
    python scripts/download_champion_models.py              # download all A-Z, all skins
    python scripts/download_champion_models.py --force      # re-download all
    python scripts/download_champion_models.py --batch A-F  # only champions A through F
    python scripts/download_champion_models.py --batch S    # only champions starting with S

Downloads ~2000+ GLB files (all skins) to D:/LoLVideoAI/GLBs/{ChampionName}/.
Each champion gets its own folder with {ChampionName}_skin{num}.glb files.
These are used by render_topdown_sprites.py to generate top-down sprite PNGs.

Champions are processed in alphabetical batches (A, B, C...) with a short
delay between downloads to avoid hammering the CDN.
"""

import argparse
import json
import string
import sys
import time
from collections import defaultdict
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "data" / "champions" / "manifest.json"

# GLB models are large — store on D:/ drive to keep the repo drive clean
MODELS_DIR = Path("D:/LoLVideoAI/GLBs")

VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
CHAMPION_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
CHAMPION_DETAIL_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion/{key}.json"
MODEL_URL = "https://cdn.modelviewer.lol/lol/models/{champion_id_lower}/{skin_id}/model-lite.glb"

# Delay between individual downloads (seconds) to be polite to CDN
DOWNLOAD_DELAY = 0.3
# Delay between letter batches
BATCH_DELAY = 1.0


def parse_batch_range(batch_str: str) -> set[str]:
    """Parse a batch argument like 'A-F' or 'S' into a set of uppercase letters."""
    batch_str = batch_str.upper().strip()

    if len(batch_str) == 1 and batch_str in string.ascii_uppercase:
        return {batch_str}

    if len(batch_str) == 3 and batch_str[1] == "-":
        start, end = batch_str[0], batch_str[2]
        if start in string.ascii_uppercase and end in string.ascii_uppercase:
            return {chr(c) for c in range(ord(start), ord(end) + 1)}

    print(f"ERROR: Invalid batch range '{batch_str}'. Use a single letter (S) or range (A-F).")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download champion 3D models (all skins) from modelviewer.lol")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument("--batch", type=str, default=None,
                        help="Only download champions starting with these letters (e.g. 'A-F' or 'S')")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"GLB output directory: {MODELS_DIR}")

    # Load manifest if available (for champion key -> numeric key mapping + skin list)
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text())
    else:
        manifest = None

    # If no manifest, fetch champion list from Data Dragon
    if manifest is None:
        print("No manifest found, fetching champion data from Data Dragon...")
        resp = requests.get(VERSIONS_URL, timeout=10)
        resp.raise_for_status()
        version = resp.json()[0]
        print(f"  Using version: {version}")

        resp = requests.get(CHAMPION_URL.format(version=version), timeout=10)
        resp.raise_for_status()
        champions = resp.json()["data"]

        # Build manifest with skin data from detailed endpoints
        manifest = {}
        for key, info in sorted(champions.items()):
            manifest[key] = {
                "name": info["name"],
                "key": info["key"],
            }
            # Fetch detailed skin info
            try:
                detail_resp = requests.get(
                    CHAMPION_DETAIL_URL.format(version=version, key=key), timeout=10
                )
                detail_resp.raise_for_status()
                detail = detail_resp.json()["data"][key]
                manifest[key]["skins"] = [
                    {"num": s["num"], "name": s["name"] if s["name"] != "default" else "Default"}
                    for s in detail.get("skins", [])
                ]
            except Exception as e:
                print(f"  WARNING: Failed to fetch skin list for {key}: {e}")
                manifest[key]["skins"] = [{"num": 0, "name": "Default"}]

        print(f"  Found {len(manifest)} champions")
    else:
        print(f"Using existing manifest with {len(manifest)} champions")

    # Filter by batch if specified
    letter_filter = None
    if args.batch:
        letter_filter = parse_batch_range(args.batch)
        print(f"Batch filter: {', '.join(sorted(letter_filter))}")

    # Group champions by first letter for batched downloading
    batches: dict[str, list[str]] = defaultdict(list)
    for champ_key in sorted(manifest):
        first_letter = champ_key[0].upper()
        if letter_filter and first_letter not in letter_filter:
            continue
        batches[first_letter].append(champ_key)

    # Count total skins to download
    total_champs = sum(len(v) for v in batches.values())
    total_skins = 0
    for champs in batches.values():
        for champ_key in champs:
            skins = manifest[champ_key].get("skins", [{"num": 0}])
            total_skins += len(skins)

    print(f"\n{total_champs} champions, {total_skins} skins to process across {len(batches)} letter batches\n")

    downloaded = 0
    skipped = 0
    failed = 0
    total_bytes = 0

    for letter in sorted(batches):
        champs = batches[letter]
        batch_dl = 0
        batch_skip = 0
        batch_fail = 0

        # Count skins in this batch
        batch_skin_count = sum(
            len(manifest[ck].get("skins", [{"num": 0}])) for ck in champs
        )
        print(f"--- Batch [{letter}] ({len(champs)} champions, {batch_skin_count} skins) ---")

        for champ_key in champs:
            info = manifest[champ_key]
            numeric_key = info.get("key")
            if numeric_key is None:
                print(f"  WARNING: No numeric key for {champ_key}, skipping")
                batch_fail += 1
                continue

            skins = info.get("skins", [{"num": 0, "name": "Default"}])
            champ_dir = MODELS_DIR / champ_key
            champ_dir.mkdir(parents=True, exist_ok=True)

            champ_dl = 0
            champ_skip = 0
            champ_fail = 0

            for skin in skins:
                skin_num = skin["num"]
                skin_name = skin.get("name", f"skin{skin_num}")
                skin_id = int(numeric_key) * 1000 + skin_num

                filename = f"{champ_key}_skin{skin_num}.glb"
                model_path = champ_dir / filename

                if model_path.exists() and not args.force:
                    champ_skip += 1
                    continue

                champion_id_lower = champ_key.lower()
                url = MODEL_URL.format(champion_id_lower=champion_id_lower, skin_id=skin_id)

                try:
                    r = requests.get(url, timeout=30)
                    r.raise_for_status()
                    size_bytes = len(r.content)
                    size_mb = size_bytes / (1024 * 1024)
                    model_path.write_bytes(r.content)
                    champ_dl += 1
                    total_bytes += size_bytes
                    time.sleep(DOWNLOAD_DELAY)
                except Exception as e:
                    champ_fail += 1

            status_parts = []
            if champ_dl > 0:
                status_parts.append(f"{champ_dl} new")
            if champ_skip > 0:
                status_parts.append(f"{champ_skip} existing")
            if champ_fail > 0:
                status_parts.append(f"{champ_fail} failed")
            status = ", ".join(status_parts) if status_parts else "nothing"

            print(f"  {champ_key} ({len(skins)} skins): {status}")

            batch_dl += champ_dl
            batch_skip += champ_skip
            batch_fail += champ_fail

        downloaded += batch_dl
        skipped += batch_skip
        failed += batch_fail

        batch_mb = "?"  # just use running total
        print(f"  Batch [{letter}] done: {batch_dl} downloaded, {batch_skip} existing, {batch_fail} failed\n")

        # Pause between batches
        if letter != sorted(batches)[-1] and batch_dl > 0:
            time.sleep(BATCH_DELAY)

    total_mb = total_bytes / (1024 * 1024)
    total_gb = total_mb / 1024
    print(f"Done!")
    print(f"  Models dir: {MODELS_DIR}")
    if total_gb >= 1.0:
        print(f"  Total: {downloaded} skins downloaded ({total_gb:.2f} GB), {skipped} existing, {failed} failed")
    else:
        print(f"  Total: {downloaded} skins downloaded ({total_mb:.1f} MB), {skipped} existing, {failed} failed")


if __name__ == "__main__":
    main()

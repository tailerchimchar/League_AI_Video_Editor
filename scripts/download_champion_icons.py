#!/usr/bin/env python3
"""Download champion icons and skin loading screens from Riot's Data Dragon CDN.

Usage:
    python scripts/download_champion_icons.py           # default icons only
    python scripts/download_champion_icons.py --skins   # also download all skin loading screens

Downloads ~172 champion icons (120x120 PNG) to data/champions/icons/ and
optionally ~1700 skin loading screen images to data/champions/skins/.
Saves manifest.json with champion/skin metadata.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "champions"
ICONS_DIR = DATA_DIR / "icons"
SKINS_DIR = DATA_DIR / "skins"
MANIFEST_PATH = DATA_DIR / "manifest.json"

VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
CHAMPION_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
CHAMPION_DETAIL_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion/{key}.json"
ICON_URL = "https://ddragon.leagueoflegends.com/cdn/{version}/img/champion/{key}.png"
LOADING_URL = "https://ddragon.leagueoflegends.com/cdn/img/champion/loading/{key}_{num}.jpg"


def download_icons(version: str, champions: dict, manifest: dict) -> None:
    """Download default champion square icons."""
    ICONS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0

    for key in sorted(champions):
        icon_path = ICONS_DIR / f"{key}.png"
        if icon_path.exists():
            skipped += 1
            continue

        url = ICON_URL.format(version=version, key=key)
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            icon_path.write_bytes(r.content)
            downloaded += 1
            if downloaded % 20 == 0:
                print(f"  Downloaded {downloaded} icons...")
        except Exception as e:
            print(f"  WARNING: Failed to download icon {key}: {e}")

    print(f"  Icons: {downloaded} new, {skipped} existing")


def download_skins(version: str, champions: dict, manifest: dict) -> None:
    """Download loading screen images for every skin of every champion."""
    SKINS_DIR.mkdir(parents=True, exist_ok=True)
    total_downloaded = 0
    total_skipped = 0
    total_skins = 0

    for key in sorted(champions):
        # Fetch detailed champion data (includes skin list)
        detail_url = CHAMPION_DETAIL_URL.format(version=version, key=key)
        try:
            resp = requests.get(detail_url, timeout=10)
            resp.raise_for_status()
            detail = resp.json()
        except Exception as e:
            print(f"  WARNING: Failed to fetch details for {key}: {e}")
            continue

        champ_data = detail["data"][key]
        skins = champ_data.get("skins", [])

        # Store skin info in manifest
        skin_list = []
        for skin in skins:
            skin_num = skin["num"]
            skin_name = skin["name"] if skin["name"] != "default" else "Default"
            skin_list.append({"num": skin_num, "name": skin_name})

            # Download loading screen image
            filename = f"{key}_{skin_num}.jpg"
            skin_path = SKINS_DIR / filename
            total_skins += 1

            if skin_path.exists():
                total_skipped += 1
                continue

            url = LOADING_URL.format(key=key, num=skin_num)
            try:
                r = requests.get(url, timeout=10)
                r.raise_for_status()
                skin_path.write_bytes(r.content)
                total_downloaded += 1
                if total_downloaded % 50 == 0:
                    print(f"  Downloaded {total_downloaded} skin images...")
            except Exception as e:
                # Some skin numbers might not have loading screens
                pass

        manifest[key]["skins"] = skin_list

    print(f"  Skins: {total_downloaded} new, {total_skipped} existing ({total_skins} total)")


def main():
    parser = argparse.ArgumentParser(description="Download champion data from Data Dragon")
    parser.add_argument("--skins", action="store_true",
                        help="Also download loading screen images for all skins")
    args = parser.parse_args()

    # Get latest version
    print("Fetching latest Data Dragon version...")
    resp = requests.get(VERSIONS_URL, timeout=10)
    resp.raise_for_status()
    version = resp.json()[0]
    print(f"  Using version: {version}")

    # Get champion list
    print("Fetching champion list...")
    resp = requests.get(CHAMPION_URL.format(version=version), timeout=10)
    resp.raise_for_status()
    champions = resp.json()["data"]
    print(f"  Found {len(champions)} champions")

    # Build manifest
    manifest: dict[str, dict] = {}
    for key, info in sorted(champions.items()):
        manifest[key] = {
            "name": info["name"],
            "title": info["title"],
            "id": info["id"],
            "key": info["key"],
        }

    # Download icons
    print("\nDownloading champion icons...")
    download_icons(version, champions, manifest)

    # Download skins if requested
    if args.skins:
        print("\nDownloading skin loading screens (this may take a few minutes)...")
        download_skins(version, champions, manifest)

    # Save manifest
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    print(f"\nDone!")
    print(f"  Manifest: {MANIFEST_PATH}")
    if args.skins:
        print(f"  Skins:    {SKINS_DIR}")


if __name__ == "__main__":
    main()

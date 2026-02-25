"""Champion identification via HUD portrait matching (Phase 2A).

Extracts the player's champion portrait from the bottom HUD and the 4 ally
portraits above the minimap, then matches them against Riot's official
champion data using spatial color histograms + template matching.

Improvements over v1:
- Spatial histograms: 2x2 grid captures color distribution across portrait
- Multi-metric scoring: histogram correlation (0.6) + template matching (0.4)
- Reduced reference noise: default skin + top 5 popular skins per champion

Enemies are identified via YOLO detection only (Phase B for specific names).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

from extractor.config import (
    PLAYER_PORTRAIT_REGION,
    ALLY_PORTRAIT_SLOTS,
    BASE_WIDTH,
    BASE_HEIGHT,
)

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_ICONS_DIR = _REPO_ROOT / "data" / "champions" / "icons"
_SKINS_DIR = _REPO_ROOT / "data" / "champions" / "skins"
_SPRITES_DIR = _REPO_ROOT / "data" / "champions" / "sprites"
_MANIFEST_PATH = _REPO_ROOT / "data" / "champions" / "manifest.json"

# Below this confidence, report "unknown" — set to 0 to always pick the best match
MATCH_CONFIDENCE_THRESHOLD = 0.0

# Maximum skins per champion to load (default + top N popular).
# Skin 0 is always the default; we take the first MAX_SKINS_PER_CHAMP skins
# which Riot orders by release (popular/classic skins come first).
MAX_SKINS_PER_CHAMP = 6

# Standard size for template matching (portrait crops are resized to this)
_TEMPLATE_SIZE = (48, 48)


class ChampionIdentifier:
    """Identifies champions from HUD portraits using spatial histograms + template matching."""

    def __init__(self, icons_only: bool = True):
        self._available = False
        # Each entry: (champion_key, label, spatial_hist, template_img)
        self._references: list[tuple[str, str, np.ndarray, np.ndarray]] = []
        self._champion_names: dict[str, str] = {}
        self._load_references(icons_only=icons_only)

    def _load_references(self, icons_only: bool = True) -> None:
        """Load reference images and precompute spatial histograms + templates.

        Args:
            icons_only: If True, skip skin loading screen images and only use
                the ~172 default champion icons. Icons are square face images
                that are much closer to HUD portraits than cropped loading
                screens, resulting in far more accurate matching.
        """
        if not _MANIFEST_PATH.exists():
            logger.warning(
                "Champion manifest not found — run scripts/download_champion_icons.py"
            )
            return

        manifest = json.loads(_MANIFEST_PATH.read_text())

        for key, info in manifest.items():
            self._champion_names[key] = info.get("name", key)

        # Load skin loading screen images (limited per champion)
        skin_count = 0
        if not icons_only and _SKINS_DIR.exists():
            for key, info in manifest.items():
                skins = info.get("skins", [])
                loaded_for_champ = 0
                for skin in skins:
                    if loaded_for_champ >= MAX_SKINS_PER_CHAMP:
                        break

                    skin_num = skin["num"]
                    skin_path = _SKINS_DIR / f"{key}_{skin_num}.jpg"
                    if not skin_path.exists():
                        continue

                    img = cv2.imread(str(skin_path))
                    if img is None:
                        continue

                    # Crop face region from loading screen (~308x560)
                    h, w = img.shape[:2]
                    face_y = int(h * 0.02)
                    face_h = int(h * 0.35)
                    face_x = int(w * 0.15)
                    face_w = int(w * 0.70)
                    face_crop = img[face_y:face_y + face_h, face_x:face_x + face_w]

                    if face_crop.size == 0:
                        continue

                    spatial_hist = self._compute_spatial_hist(face_crop)
                    template = cv2.resize(face_crop, _TEMPLATE_SIZE)
                    label = f"{key}_skin{skin_num}"
                    self._references.append((key, label, spatial_hist, template))
                    skin_count += 1
                    loaded_for_champ += 1

        if skin_count > 0:
            logger.info(
                "Champion identifier loaded %d skin references (%d champions)",
                skin_count, len(self._champion_names),
            )
        elif not icons_only:
            logger.info("No skin images found, using default icons only")

        # Load default square icons as references
        icon_count = 0
        if _ICONS_DIR.exists():
            for key in manifest:
                icon_path = _ICONS_DIR / f"{key}.png"
                if not icon_path.exists():
                    continue
                img = cv2.imread(str(icon_path))
                if img is None:
                    continue
                spatial_hist = self._compute_spatial_hist(img)
                template = cv2.resize(img, _TEMPLATE_SIZE)
                self._references.append((key, f"{key}_icon", spatial_hist, template))
                icon_count += 1

        # Load top-down sprites rendered from 3D models (default + all skins)
        sprite_count = 0
        if _SPRITES_DIR.exists():
            for key in manifest:
                # Gather all sprites: {Key}.png, {Key}_skin{N}.png, {Key}_mobafire_*.png
                sprite_paths = []
                default_sprite = _SPRITES_DIR / f"{key}.png"
                if default_sprite.exists():
                    sprite_paths.append((default_sprite, f"{key}_sprite"))
                for skin_sprite in sorted(_SPRITES_DIR.glob(f"{key}_skin*.png")):
                    sprite_paths.append((skin_sprite, f"{skin_sprite.stem}_sprite"))
                for mf_sprite in sorted(_SPRITES_DIR.glob(f"{key}_mobafire_*.png")):
                    sprite_paths.append((mf_sprite, f"{mf_sprite.stem}_sprite"))

                for sprite_path, label in sprite_paths:
                    img = cv2.imread(str(sprite_path), cv2.IMREAD_UNCHANGED)
                    if img is None:
                        continue
                    # Handle alpha channel: composite onto black background
                    if len(img.shape) == 3 and img.shape[2] == 4:
                        alpha = img[:, :, 3:] / 255.0
                        img = (img[:, :, :3] * alpha).astype(np.uint8)
                    spatial_hist = self._compute_spatial_hist(img)
                    template = cv2.resize(img, _TEMPLATE_SIZE)
                    self._references.append((key, label, spatial_hist, template))
                    sprite_count += 1

        total = skin_count + icon_count + sprite_count
        if total > 0:
            self._available = True
            logger.info(
                "Champion identifier ready: %d references (%d skins + %d icons + %d sprites, icons_only=%s)",
                total, skin_count, icon_count, sprite_count, icons_only,
            )
        else:
            logger.warning("No champion references loaded")

    @staticmethod
    def _compute_hist(img: np.ndarray) -> np.ndarray:
        """Compute a normalized HSV histogram for a single image region."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [36, 40], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist

    @staticmethod
    def _compute_spatial_hist(img: np.ndarray) -> np.ndarray:
        """Compute 2x2 spatial HSV histograms concatenated into one vector.

        Splits the image into a 2x2 grid and computes a per-quadrant HSV
        histogram, then concatenates them. This captures spatial color
        distribution (e.g., hair color top, armor color bottom).
        """
        h, w = img.shape[:2]
        mid_h, mid_w = h // 2, w // 2

        quadrants = [
            img[0:mid_h, 0:mid_w],       # top-left
            img[0:mid_h, mid_w:w],        # top-right
            img[mid_h:h, 0:mid_w],        # bottom-left
            img[mid_h:h, mid_w:w],        # bottom-right
        ]

        hists = []
        for quad in quadrants:
            if quad.size == 0:
                hists.append(np.zeros((36, 40), dtype=np.float32))
                continue
            hsv = cv2.cvtColor(quad, cv2.COLOR_BGR2HSV)
            qh = cv2.calcHist([hsv], [0, 1], None, [36, 40], [0, 180, 0, 256])
            cv2.normalize(qh, qh)
            hists.append(qh)

        return np.concatenate([h.flatten() for h in hists])

    @property
    def available(self) -> bool:
        return self._available

    def identify_from_portraits(
        self, frame: np.ndarray, width: int, height: int
    ) -> dict:
        """Identify player + 4 allies from HUD portraits.

        Returns:
            Dict with "player" and "allies" keys.
        """
        if not self._available:
            return {"player": None, "allies": []}

        scale_x = width / BASE_WIDTH
        scale_y = height / BASE_HEIGHT

        # Identify player
        player_crop = self._extract_region(
            frame, PLAYER_PORTRAIT_REGION, scale_x, scale_y
        )
        player = self._match_portrait(player_crop) if player_crop is not None else None

        # Identify allies
        allies = []
        for slot in ALLY_PORTRAIT_SLOTS:
            ally_crop = self._extract_region(frame, slot, scale_x, scale_y)
            if ally_crop is not None:
                allies.append(self._match_portrait(ally_crop))
            else:
                allies.append({"key": "unknown", "name": "Unknown", "confidence": 0.0})

        return {"player": player, "allies": allies}

    def _extract_region(
        self,
        frame: np.ndarray,
        region: dict[str, int],
        scale_x: float,
        scale_y: float,
    ) -> np.ndarray | None:
        """Extract and validate a region from the frame."""
        fh, fw = frame.shape[:2]
        rx = int(region["x"] * scale_x)
        ry = int(region["y"] * scale_y)
        rw = int(region["w"] * scale_x)
        rh = int(region["h"] * scale_y)

        rx = max(0, min(rx, fw - 1))
        ry = max(0, min(ry, fh - 1))
        rw = min(rw, fw - rx)
        rh = min(rh, fh - ry)

        crop = frame[ry : ry + rh, rx : rx + rw]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        return crop

    def _match_portrait(self, portrait_crop: np.ndarray) -> dict:
        """Match a portrait crop against all references using multi-metric scoring.

        Combines:
        - Spatial histogram correlation (weight 0.6): captures color layout
        - Template matching via normalized cross-correlation (weight 0.4): captures texture
        """
        crop_spatial_hist = self._compute_spatial_hist(portrait_crop)
        crop_template = cv2.resize(portrait_crop, _TEMPLATE_SIZE)

        # Score each reference, keep best per champion
        best_per_champ: dict[str, float] = {}

        for champ_key, label, ref_spatial_hist, ref_template in self._references:
            # 1) Spatial histogram correlation
            # Compare flattened spatial histograms using correlation
            hist_score = cv2.compareHist(
                crop_spatial_hist.reshape(-1).astype(np.float32),
                ref_spatial_hist.reshape(-1).astype(np.float32),
                cv2.HISTCMP_CORREL,
            )
            hist_score = max(0.0, hist_score)

            # 2) Template matching (normalized cross-correlation)
            result = cv2.matchTemplate(
                crop_template, ref_template, cv2.TM_CCOEFF_NORMED
            )
            tmpl_score = max(0.0, float(result[0, 0]))

            # Combined score
            score = 0.6 * hist_score + 0.4 * tmpl_score

            if champ_key not in best_per_champ or score > best_per_champ[champ_key]:
                best_per_champ[champ_key] = score

        if not best_per_champ:
            return {"key": "unknown", "name": "Unknown", "confidence": 0.0}

        # Find the champion with the highest score
        best_key = max(best_per_champ, key=best_per_champ.get)
        best_score = best_per_champ[best_key]

        if best_score < MATCH_CONFIDENCE_THRESHOLD:
            return {"key": "unknown", "name": "Unknown", "confidence": round(best_score, 3)}

        return {
            "key": best_key,
            "name": self._champion_names.get(best_key, best_key),
            "confidence": round(best_score, 3),
        }

    def match_detection(
        self,
        frame: np.ndarray,
        bbox: list[float],
        candidate_keys: list[str],
    ) -> dict | None:
        """Match a YOLO detection crop against a narrowed set of candidate champions.

        This is the reusable matching method for any role — allies (1-of-4),
        enemies (1-of-5), or player verification (1-of-1). Crops the body
        region from the bbox and scores against only the candidate references.

        Args:
            frame: Full video frame (BGR numpy array).
            bbox: Detection bounding box [x1, y1, x2, y2].
            candidate_keys: Champion keys to match against (e.g., ["lux", "thresh"]).

        Returns:
            {"key": str, "name": str, "score": float} or None if no confident match.
        """
        if not self._available or not candidate_keys:
            return None

        # Extract body crop: skip top 10% (health bar/name) and bottom 5% (feet)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        det_h = y2 - y1
        det_w = x2 - x1
        if det_h < 10 or det_w < 10:
            return None

        crop_y1 = y1 + int(det_h * 0.10)
        crop_y2 = y2 - int(det_h * 0.05)

        fh, fw = frame.shape[:2]
        crop_y1 = max(0, min(crop_y1, fh - 1))
        crop_y2 = max(crop_y1 + 1, min(crop_y2, fh))
        crop_x1 = max(0, min(x1, fw - 1))
        crop_x2 = max(crop_x1 + 1, min(x2, fw))

        body_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        if body_crop.size == 0 or body_crop.shape[0] < 8 or body_crop.shape[1] < 8:
            return None

        # Compute features for the detection crop
        crop_hist = self._compute_spatial_hist(body_crop)
        crop_tmpl = cv2.resize(body_crop, _TEMPLATE_SIZE)

        # Filter references to only candidate keys
        candidate_set = set(candidate_keys)

        # Score against candidate references only (same scoring as _match_portrait
        # but with heavier color weight — in-game models vary more by color)
        best_per_champ: dict[str, float] = {}
        for champ_key, label, ref_hist, ref_tmpl in self._references:
            if champ_key not in candidate_set:
                continue

            hist_score = cv2.compareHist(
                crop_hist.reshape(-1).astype(np.float32),
                ref_hist.reshape(-1).astype(np.float32),
                cv2.HISTCMP_CORREL,
            )
            hist_score = max(0.0, hist_score)

            result = cv2.matchTemplate(crop_tmpl, ref_tmpl, cv2.TM_CCOEFF_NORMED)
            tmpl_score = max(0.0, float(result[0, 0]))

            # Heavier color weight for in-game detection vs portrait matching
            score = 0.7 * hist_score + 0.3 * tmpl_score

            if champ_key not in best_per_champ or score > best_per_champ[champ_key]:
                best_per_champ[champ_key] = score

        if not best_per_champ:
            return None

        ranked = sorted(best_per_champ.items(), key=lambda kv: kv[1], reverse=True)
        best_key, best_score = ranked[0]

        # Relative threshold: best must beat second-best by 15% (easy for small candidate sets)
        if len(ranked) >= 2:
            second_score = ranked[1][1]
            if second_score > 0 and (best_score - second_score) / second_score < 0.15:
                return None

        if best_score < 0.1:
            return None

        return {
            "key": best_key,
            "name": self._champion_names.get(best_key, best_key),
            "score": round(best_score, 4),
        }

    def identify_from_portraits_multi(
        self, frames: list[np.ndarray], width: int, height: int, n_samples: int = 5
    ) -> dict:
        """Multi-frame portrait identification with voting for improved accuracy.

        Samples up to n_samples frames evenly, runs identify_from_portraits on each,
        accumulates confidence per ally slot, and picks the highest cumulative score
        with deduplication (no two slots can be the same champion).

        Args:
            frames: List of frame numpy arrays (BGR).
            width: Video width.
            height: Video height.
            n_samples: Max frames to sample (evenly spaced from the list).

        Returns:
            Same format as identify_from_portraits: {"player": {...}, "allies": [...]}.
        """
        if not self._available or not frames:
            return {"player": None, "allies": []}

        # Sample frames evenly
        n = min(n_samples, len(frames))
        if n <= 0:
            return {"player": None, "allies": []}

        step = max(1, len(frames) // n)
        sampled = [frames[i * step] for i in range(n) if i * step < len(frames)]

        # Run portrait ID on each sampled frame
        results = []
        for frame in sampled:
            try:
                result = self.identify_from_portraits(frame, width, height)
                results.append(result)
            except Exception:
                logger.debug("Portrait ID failed on one sample frame", exc_info=True)

        if not results:
            return {"player": None, "allies": []}

        # Accumulate player votes: champion_key -> total_confidence
        player_votes: dict[str, float] = {}
        player_names: dict[str, str] = {}
        for r in results:
            p = r.get("player")
            if p and p["key"] != "unknown":
                player_votes[p["key"]] = player_votes.get(p["key"], 0) + p["confidence"]
                player_names[p["key"]] = p["name"]

        # Best player
        best_player = None
        if player_votes:
            best_key = max(player_votes, key=player_votes.get)
            best_player = {
                "key": best_key,
                "name": player_names[best_key],
                "confidence": round(player_votes[best_key] / len(results), 3),
            }

        # Accumulate per-slot ally votes
        n_slots = max((len(r.get("allies", [])) for r in results), default=0)
        slot_votes: list[dict[str, float]] = [{} for _ in range(n_slots)]
        slot_names: list[dict[str, str]] = [{} for _ in range(n_slots)]

        for r in results:
            allies = r.get("allies", [])
            for slot_idx, ally in enumerate(allies):
                if slot_idx >= n_slots:
                    break
                if ally["key"] != "unknown":
                    slot_votes[slot_idx][ally["key"]] = (
                        slot_votes[slot_idx].get(ally["key"], 0) + ally["confidence"]
                    )
                    slot_names[slot_idx][ally["key"]] = ally["name"]

        # Pick best per slot with deduplication
        used_keys: set[str] = set()
        if best_player:
            used_keys.add(best_player["key"])

        allies = []
        for slot_idx in range(n_slots):
            votes = slot_votes[slot_idx]
            names = slot_names[slot_idx]
            # Sort by cumulative confidence, skip already-used keys
            ranked = sorted(votes.items(), key=lambda kv: kv[1], reverse=True)
            chosen = None
            for key, total_conf in ranked:
                if key not in used_keys:
                    chosen = {
                        "key": key,
                        "name": names[key],
                        "confidence": round(total_conf / len(results), 3),
                    }
                    used_keys.add(key)
                    break
            allies.append(chosen or {"key": "unknown", "name": "Unknown", "confidence": 0.0})

        return {"player": best_player, "allies": allies}

    def get_champion_name(self, key: str) -> str:
        """Look up display name for a champion key."""
        return self._champion_names.get(key, key)

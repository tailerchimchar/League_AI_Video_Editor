"""Crop region configurations and extraction constants.

Crop coordinates are defined for 1920x1080 base resolution and scaled
proportionally for other resolutions. These are approximate starting points
and should be calibrated against actual League screenshots.
"""

# Base resolution for crop definitions
BASE_WIDTH = 1920
BASE_HEIGHT = 1080

# Crop regions for 1080p League of Legends HUD.
# Calibrated for Outplayed-style recordings with stats overlay in top-right.
CROP_REGIONS_1080P = {
    "minimap":       {"x": 1630, "y": 805,  "w": 280, "h": 275},
    "game_timer":    {"x": 1830, "y": 2,    "w": 85,  "h": 28},
    "player_stats":  {"x": 390,  "y": 945,  "w": 150, "h": 115},
    "player_level":  {"x": 694,  "y": 1043, "w": 28,  "h": 20},
    "cs_gold":       {"x": 1780, "y": 4,    "w": 55,  "h": 22},
    "kda":           {"x": 1660, "y": 2,    "w": 80,  "h": 28},
    "items":         {"x": 620,  "y": 915,  "w": 170, "h": 40},
    "scoreboard":    {"x": 1455, "y": 2,    "w": 200, "h": 28},
    "ability_bar":   {"x": 680,  "y": 960,  "w": 210, "h": 50},
    "health_bar":    {"x": 850,  "y": 1030, "w": 140, "h": 20},
    "resources_bar": {"x": 860,  "y": 1050, "w": 100, "h": 20},
    "summoner_spells": {"x": 695, "y": 970, "w": 50, "h": 50},
}

# ---------------------------------------------------------------------------
# Champion portrait positions (Phase 2A)
# ---------------------------------------------------------------------------
# Calibrated for Outplayed recordings at 1920x1080.
# Player portrait: large circular icon at bottom-center HUD.
# Ally portraits: 4 circular icons in a row above the minimap (right side).
# Enemies: identified via YOLO detection only (Phase B for specific IDs).
#
# Use scripts/calibrate_scoreboard.py to fine-tune these values.
# ---------------------------------------------------------------------------

# Player's own champion portrait (bottom-center HUD, face region only)
PLAYER_PORTRAIT_REGION = {"x": 650, "y": 1005, "w": 60, "h": 52}

# 4 ally champion portraits above the minimap (face regions, avoiding
# the cooldown circle indicator at the top of each frame)
ALLY_PORTRAIT_SLOTS: list[dict[str, int]] = [
    {"x": 1635, "y": 688, "w": 50, "h": 38},  # Ally 1 (leftmost)
    {"x": 1710, "y": 688, "w": 50, "h": 38},  # Ally 2
    {"x": 1784, "y": 688, "w": 50, "h": 38},  # Ally 3
    {"x": 1856, "y": 688, "w": 50, "h": 38},  # Ally 4 (rightmost)
]

# OCR field types that use numeric whitelist
NUMERIC_FIELDS = {"cs_gold", "game_timer", "player_level"}
KDA_FIELDS = {"kda"}
HEALTH_BAR_FIELDS = {"health_bar", "resources_bar"}

# Extraction defaults
DEFAULT_SAMPLE_FPS = 5.0
MAX_FRAMES_PER_EXTRACTION = 600
BATCH_INSERT_SIZE = 20

# Segmenter thresholds
FIGHT_HP_DELTA_THRESHOLD = 0.05  # 5% HP change per sample
FIGHT_MIN_DURATION_MS = 2000     # minimum 2 seconds


def get_crop_regions(width: int, height: int) -> dict[str, dict[str, int]]:
    """Scale 1080p crop regions to the actual video resolution."""
    scale_x = width / BASE_WIDTH
    scale_y = height / BASE_HEIGHT
    return {
        name: {
            "x": int(r["x"] * scale_x),
            "y": int(r["y"] * scale_y),
            "w": int(r["w"] * scale_x),
            "h": int(r["h"] * scale_y),
        }
        for name, r in CROP_REGIONS_1080P.items()
    }

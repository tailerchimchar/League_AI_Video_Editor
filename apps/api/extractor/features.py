"""Derived micro feature computation from OCR deltas between frames."""


def compute_derived_features(
    current_ocr: dict,
    previous_ocr: dict | None,
    dt_ms: int,
) -> dict:
    """Compute derived micro features from OCR deltas between frames.

    Args:
        current_ocr: OCR data for the current frame.
        previous_ocr: OCR data for the previous frame (None if first frame).
        dt_ms: Time delta in milliseconds between frames.

    Returns:
        Dict of derived feature name -> value.
    """
    features: dict = {}

    if previous_ocr is None or dt_ms <= 0:
        return features

    dt_sec = dt_ms / 1000

    # HP delta (proxy for taking/dealing damage)
    cur_hp = current_ocr.get("player_hp_pct")
    prev_hp = previous_ocr.get("player_hp_pct")
    if cur_hp is not None and prev_hp is not None:
        features["hp_delta"] = cur_hp - prev_hp

    # CS delta (CS per minute proxy)
    cur_cs = current_ocr.get("player_cs")
    prev_cs = previous_ocr.get("player_cs")
    if cur_cs is not None and prev_cs is not None:
        cs_diff = cur_cs - prev_cs
        if cs_diff >= 0:  # sanity check — CS shouldn't decrease
            features["cs_delta"] = cs_diff / (dt_sec / 60) if dt_sec > 0 else 0

    # Gold delta
    cur_gold = current_ocr.get("player_gold")
    prev_gold = previous_ocr.get("player_gold")
    if cur_gold is not None and prev_gold is not None:
        features["gold_delta"] = cur_gold - prev_gold

    return features

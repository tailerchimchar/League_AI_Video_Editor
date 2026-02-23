"""Analyze health bar colors from a full League screenshot.

Samples specific pixel regions where health bars are visible and reports HSV values.
"""
import sys
import cv2
import numpy as np


def sample_region(img, hsv, x1, y1, x2, y2, label):
    """Sample a rectangular region and report HSV stats."""
    h_frame, w_frame = img.shape[:2]
    x1 = max(0, min(x1, w_frame))
    x2 = max(0, min(x2, w_frame))
    y1 = max(0, min(y1, h_frame))
    y2 = max(0, min(y2, h_frame))

    roi = hsv[y1:y2, x1:x2]
    bgr_roi = img[y1:y2, x1:x2]

    if roi.size == 0:
        print(f"  {label}: empty region")
        return

    # Filter for saturated, bright pixels (actual health bar color)
    mask = (roi[:,:,1] > 30) & (roi[:,:,2] > 50)
    if not np.any(mask):
        print(f"  {label}: no saturated pixels in ({x1},{y1})-({x2},{y2})")
        return

    pixels = roi[mask]
    bgr_pixels = bgr_roi[mask.reshape(bgr_roi.shape[:2])] if mask.any() else None

    print(f"\n  {label} region ({x1},{y1})-({x2},{y2}): {len(pixels)} colored pixels")
    print(f"    H: min={pixels[:,0].min()}, max={pixels[:,0].max()}, mean={pixels[:,0].mean():.1f}, median={np.median(pixels[:,0]):.1f}")
    print(f"    S: min={pixels[:,1].min()}, max={pixels[:,1].max()}, mean={pixels[:,1].mean():.1f}, median={np.median(pixels[:,1]):.1f}")
    print(f"    V: min={pixels[:,2].min()}, max={pixels[:,2].max()}, mean={pixels[:,2].mean():.1f}, median={np.median(pixels[:,2]):.1f}")
    if bgr_pixels is not None and len(bgr_pixels) > 0:
        print(f"    BGR: B={bgr_pixels[:,0].mean():.0f} G={bgr_pixels[:,1].mean():.0f} R={bgr_pixels[:,2].mean():.0f}")

    # H histogram
    h_vals = pixels[:,0]
    print(f"    H distribution:")
    for lo in range(0, 180, 10):
        hi = lo + 10
        count = np.sum((h_vals >= lo) & (h_vals < hi))
        if count > 0:
            pct = count * 100 // len(h_vals)
            bar = '#' * min(40, max(1, pct * 40 // 100))
            print(f"      H {lo:3d}-{hi:3d}: {count:5d} ({pct:2d}%) {bar}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/analyze_full_screenshot.py <screenshot_path>")
        sys.exit(1)

    path = sys.argv[1]
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: Could not read {path}")
        sys.exit(1)

    h, w = img.shape[:2]
    print(f"Image: {path} ({w}x{h})")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Scale factors from 1920x1080
    sx = w / 1920
    sy = h / 1080

    # Known health bar locations from the screenshots:
    # The player's bottom HUD health bar
    print("\n--- Bottom HUD (player) health bar ---")
    sample_region(img, hsv,
        int(560 * sx), int(680 * sy), int(660 * sx), int(695 * sy),
        "Player HUD HP bar")

    # Sample broad area where health bars appear above characters
    # Scan for green-ish and blue-ish horizontal lines in the upper 2/3 of frame
    print("\n--- Scanning for colored horizontal bars in game area ---")

    # Let's just scan all bright, saturated pixels and categorize by hue
    game_area = hsv[0:int(h*0.85), 0:w]
    mask = (game_area[:,:,1] > 50) & (game_area[:,:,2] > 80)
    if np.any(mask):
        pixels = game_area[mask]
        h_vals = pixels[:,0]
        print(f"\nAll saturated bright pixels in game area: {len(pixels)}")
        print("Hue distribution:")
        for lo in range(0, 180, 5):
            hi = lo + 5
            count = np.sum((h_vals >= lo) & (h_vals < hi))
            if count > 100:  # Only show significant clusters
                pct = count * 100 / len(h_vals)
                bar = '#' * min(50, max(1, int(pct * 10)))
                print(f"  H {lo:3d}-{hi:3d}: {count:6d} ({pct:4.1f}%) {bar}")


if __name__ == "__main__":
    main()

"""Extract HSV values from health bar screenshots to calibrate postprocess.py ranges.

Usage: python scripts/extract_health_bar_colors.py <image_path>

Samples pixels from the center region of the image (where the health bar is)
and reports HSV statistics.
"""
import sys
import cv2
import numpy as np

def analyze_image(path: str, label: str = ""):
    img = cv2.imread(path)
    if img is None:
        print(f"ERROR: Could not read {path}")
        return

    h, w = img.shape[:2]
    print(f"\n{'='*60}")
    print(f"Image: {path} ({w}x{h}) {label}")
    print(f"{'='*60}")

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Sample the middle 60% of the image (avoid edges)
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)
    roi = hsv[y1:y2, x1:x2]

    # Filter out very dark pixels (background, not health bar)
    # V > 40 to exclude dark backgrounds
    mask = roi[:,:,2] > 40
    if not np.any(mask):
        print("No bright pixels found")
        return

    pixels = roi[mask]

    print(f"\nAll bright pixels (V>40): {len(pixels)} pixels")
    print(f"  H: min={pixels[:,0].min()}, max={pixels[:,0].max()}, mean={pixels[:,0].mean():.1f}, median={np.median(pixels[:,0]):.1f}")
    print(f"  S: min={pixels[:,1].min()}, max={pixels[:,1].max()}, mean={pixels[:,1].mean():.1f}, median={np.median(pixels[:,1]):.1f}")
    print(f"  V: min={pixels[:,2].min()}, max={pixels[:,2].max()}, mean={pixels[:,2].mean():.1f}, median={np.median(pixels[:,2]):.1f}")

    # Also filter for saturated pixels (actual colored health bar, not white/gray)
    sat_mask = (roi[:,:,1] > 30) & (roi[:,:,2] > 40)
    if np.any(sat_mask):
        sat_pixels = roi[sat_mask]
        print(f"\nSaturated pixels (S>30, V>40): {len(sat_pixels)} pixels")
        print(f"  H: min={sat_pixels[:,0].min()}, max={sat_pixels[:,0].max()}, mean={sat_pixels[:,0].mean():.1f}, median={np.median(sat_pixels[:,0]):.1f}")
        print(f"  S: min={sat_pixels[:,1].min()}, max={sat_pixels[:,1].max()}, mean={sat_pixels[:,1].mean():.1f}, median={np.median(sat_pixels[:,1]):.1f}")
        print(f"  V: min={sat_pixels[:,2].min()}, max={sat_pixels[:,2].max()}, mean={sat_pixels[:,2].mean():.1f}, median={np.median(sat_pixels[:,2]):.1f}")

        # Show H histogram buckets
        h_vals = sat_pixels[:,0]
        print(f"\n  H distribution:")
        for lo in range(0, 180, 10):
            hi = lo + 10
            count = np.sum((h_vals >= lo) & (h_vals < hi))
            if count > 0:
                bar = '#' * min(50, count * 50 // len(h_vals))
                print(f"    H {lo:3d}-{hi:3d}: {count:5d} ({count*100//len(h_vals):2d}%) {bar}")

    # BGR analysis too
    bgr_roi = img[y1:y2, x1:x2]
    bgr_pixels = bgr_roi[mask]
    print(f"\nBGR values:")
    print(f"  B: mean={bgr_pixels[:,0].mean():.1f}")
    print(f"  G: mean={bgr_pixels[:,1].mean():.1f}")
    print(f"  R: mean={bgr_pixels[:,2].mean():.1f}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/extract_health_bar_colors.py <image_path> [label]")
        sys.exit(1)

    label = sys.argv[2] if len(sys.argv) > 2 else ""
    analyze_image(sys.argv[1], label)

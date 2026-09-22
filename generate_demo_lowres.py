"""
Generate High-Resolution Ground Truth & Low-Resolution Synthetic Test Image Pair
Day 25 - Super-Resolution Image Upscaler & Detail Enhancer
"""

import os
import cv2
import numpy as np

def create_synthetic_super_res_demo(output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)

    hr_path = os.path.join(output_dir, "ground_truth_hr.jpg")
    lr_path = os.path.join(output_dir, "input_lowres.jpg")

    # Generate rich 800x800 high-resolution test image with sharp text, geometric shapes, and fine textures
    h, w = 800, 800
    hr_img = np.zeros((h, w, 3), dtype=np.uint8)

    # Background gradient
    for y in range(h):
        for x in range(w):
            hr_img[y, x] = [
                int(180 + 70 * (y / h)),
                int(200 + 55 * (x / w)),
                int(220 - 50 * ((x+y) / (h+w)))
            ]

    # Draw complex fine line grid pattern
    for i in range(0, w, 20):
        cv2.line(hr_img, (i, 0), (i, h), (100, 100, 100), 1)
        cv2.line(hr_img, (0, i), (w, i), (100, 100, 100), 1)

    # Draw sharp concentric geometric circles and polygons
    cv2.circle(hr_img, (400, 350), 180, (255, 50, 50), 4)
    cv2.circle(hr_img, (400, 350), 120, (50, 200, 50), 3)
    cv2.circle(hr_img, (400, 350), 60, (50, 50, 255), 2)

    # Draw crisp text characters
    cv2.putText(hr_img, "NEURAL SUPER RESOLUTION 4X", (80, 120), cv2.FONT_HERSHEY_DUPLEX, 1.4, (20, 20, 20), 3)
    cv2.putText(hr_img, "Computer Vision AI Benchmark", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 150), 2)
    cv2.putText(hr_img, "Detail & Edge Enhancement Studio", (120, 620), cv2.FONT_HERSHEY_TRIPLEX, 1.1, (10, 100, 10), 2)

    # Add fine checkerboard texture in corner
    for y in range(650, 780, 10):
        for x in range(650, 780, 10):
            if ((x // 10) + (y // 10)) % 2 == 0:
                hr_img[y:y+10, x:x+10] = (255, 255, 255)
            else:
                hr_img[y:y+10, x:x+10] = (0, 0, 0)

    # Save Ground Truth HR Image
    cv2.imwrite(hr_path, hr_img)
    print(f"[SUCCESS] High-Resolution Ground Truth saved: {hr_path} (800x800)")

    # Downsample by 4x to create 200x200 low-resolution degraded input $I_{LR}$
    lr_h, lr_w = 200, 200
    lr_img = cv2.resize(hr_img, (lr_w, lr_h), interpolation=cv2.INTER_AREA)

    # Add subtle Gaussian blur to simulate degradation
    lr_img = cv2.GaussianBlur(lr_img, (3, 3), 0.5)

    cv2.imwrite(lr_path, lr_img)
    print(f"[SUCCESS] Low-Resolution Test Input saved: {lr_path} (200x200)")

    return hr_path, lr_path

if __name__ == "__main__":
    create_synthetic_super_res_demo()

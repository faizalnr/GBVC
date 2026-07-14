import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# Update this path
# ===============================
DATASET_PATH = "dataset_images"

# Histogram accumulators
hist_original = np.zeros(256, dtype=np.float64)
hist_share = np.zeros(256, dtype=np.float64)
hist_reconstructed = np.zeros(256, dtype=np.float64)

image_count = 0

for root, _, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(root, file)

            # Load grayscale image
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # -------------------------------
            # Gradient calculation (3x3)
            # -------------------------------
            padded = np.pad(img, 1, mode='reflect')
            gradient = np.zeros_like(img, dtype=np.float32)

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    neighborhood = padded[i:i+3, j:j+3]
                    center = img[i, j]
                    gradient[i, j] = np.mean(np.abs(neighborhood - center))

            gradient = np.clip(gradient, 0, 255).astype(np.uint8)

            # -------------------------------
            # Share generation (GBVC)
            # -------------------------------
            r1 = np.random.randint(0, 256, img.shape, dtype=np.uint8)
            r2 = np.random.randint(0, 256, img.shape, dtype=np.uint8)

            s1 = (gradient + r1) % 256
            s2 = (gradient + r2) % 256
            s3 = (img.astype(np.int16) - s1.astype(np.int16) - s2.astype(np.int16)) % 256
            s3 = s3.astype(np.uint8)

            # Reconstruction
            reconstructed = (s1.astype(np.int16) + s2.astype(np.int16) + s3.astype(np.int16)) % 256
            reconstructed = reconstructed.astype(np.uint8)

            # -------------------------------
            # Histogram computation
            # -------------------------------
            h_org = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
            h_share = cv2.calcHist([s1], [0], None, [256], [0, 256]).flatten()
            h_rec = cv2.calcHist([reconstructed], [0], None, [256], [0, 256]).flatten()

            # Normalize and accumulate
            hist_original += h_org / np.sum(h_org)
            hist_share += h_share / np.sum(h_share)
            hist_reconstructed += h_rec / np.sum(h_rec)

            image_count += 1

# Average histograms
hist_original /= image_count
hist_share /= image_count
hist_reconstructed /= image_count

# ===============================
# Plot (IEEE-ready)
# ===============================
plt.figure(figsize=(10, 6))
plt.plot(hist_original, label="Original Images", linewidth=2)
plt.plot(hist_share, label="Shares (GBVC)", linestyle="--")
plt.plot(hist_reconstructed, label="Reconstructed Images", linewidth=2)

plt.title("Fig. 6.1. Dataset-Level Histogram Comparison under Statistical Attack")
plt.xlabel("Pixel Intensity")
plt.ylabel("Average Normalized Frequency")
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.show()

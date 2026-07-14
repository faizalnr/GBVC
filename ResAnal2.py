import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# -------------------------------------------------
# Gradient Computation
# -------------------------------------------------
def calculate_gradient(image, window_size=3):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    pad = window_size // 2

    padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    gradient = np.zeros((h, w), dtype=np.float64)

    for i in range(h):
        for j in range(w):
            neighborhood = padded[i:i + window_size, j:j + window_size]
            gradient[i, j] = np.mean(np.abs(gray[i, j] - neighborhood))

    gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX)
    return gradient.astype(np.uint8)

# -------------------------------------------------
# Share Generation
# -------------------------------------------------
def generate_shares(image, gradient, n_shares=3, alpha_weights=None):
    if alpha_weights is None:
        alpha_weights = [0.5] * (n_shares - 1)

    h, w, c = image.shape
    image = image.astype(np.int32)
    gradient = gradient.astype(np.int32)

    shares = [np.zeros((h, w, c), dtype=np.int32) for _ in range(n_shares)]

    for ch in range(c):
        for i in range(h):
            for j in range(w):
                pixel = image[i, j, ch]
                grad = gradient[i, j]
                random_vals = np.random.randint(0, 256, n_shares - 1)

                partial_sum = 0
                for k in range(n_shares - 1):
                    shares[k][i, j, ch] = int(
                        (alpha_weights[k] * grad + random_vals[k]) % 256
                    )
                    partial_sum += shares[k][i, j, ch]

                shares[-1][i, j, ch] = (pixel - partial_sum) % 256

    return [s.astype(np.uint8) for s in shares]

# -------------------------------------------------
# Reconstruction
# -------------------------------------------------
def reconstruct_image(shares):
    reconstructed = np.sum(
        np.stack(shares, axis=0).astype(np.int32), axis=0
    ) % 256
    return reconstructed.astype(np.uint8)

# -------------------------------------------------
# Metrics
# -------------------------------------------------
def calculate_metrics(original, reconstructed):
    original = original.astype(np.float64)
    reconstructed = reconstructed.astype(np.float64)

    mse = np.mean((original - reconstructed) ** 2)
    psnr = np.inf if mse == 0 else 10 * np.log10((255 ** 2) / mse)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_total = 0
    for ch in range(original.shape[2]):
        x = original[:, :, ch]
        y = reconstructed[:, :, ch]

        mu_x, mu_y = x.mean(), y.mean()
        var_x, var_y = x.var(), y.var()
        cov = np.mean((x - mu_x) * (y - mu_y))

        ssim = ((2 * mu_x * mu_y + C1) * (2 * cov + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2))
        ssim_total += ssim

    return mse, psnr, ssim_total / original.shape[2]

# -------------------------------------------------
# Plot and Save Results
# -------------------------------------------------
def save_results(image, shares, reconstructed, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Save images
    cv2.imwrite(os.path.join(out_dir, "input.png"), image)
    for i, s in enumerate(shares):
        cv2.imwrite(os.path.join(out_dir, f"share_{i+1}.png"), s)
    cv2.imwrite(os.path.join(out_dir, "reconstructed.png"), reconstructed)

    # Combined image plot
    titles = ["Input"] + [f"Share {i+1}" for i in range(len(shares))] + ["Reconstructed"]
    images = [image] + shares + [reconstructed]

    plt.figure(figsize=(20, 8))
    for i, img in enumerate(images):
        plt.subplot(1, len(images), i + 1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "combined_results.png"), dpi=300)
    plt.close()

    # Histogram plot
    plt.figure(figsize=(14, 8))

    # First row (center-aligned): Input and Reconstructed
    plt.subplot(2, 3, 2)
    plt.hist(image.ravel(), bins=256, color='gray')
    plt.title("Histogram: Input")

    plt.subplot(2, 3, 3)
    plt.hist(reconstructed.ravel(), bins=256, color='gray')
    plt.title("Histogram: Reconstructed")

    # Second row: Shares
    for i, s in enumerate(shares):
        plt.subplot(2, 3, 4 + i)
        plt.hist(s.ravel(), bins=256, color='gray')
        plt.title(f"Histogram: Share {i+1}")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "histograms.png"), dpi=300)
    plt.close()

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    input_path = "Xray.jpg"
    image = cv2.imread(input_path)

    if image is None:
        print("Error: Image not found.")
        return

    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = base_name

    gradient = calculate_gradient(image)
    shares = generate_shares(image, gradient, n_shares=3)
    reconstructed = reconstruct_image(shares)

    save_results(image, shares, reconstructed, output_dir)

    mse, psnr, ssim = calculate_metrics(image, reconstructed)
    print(f"MSE: {mse:.4f}, PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")

if __name__ == "__main__":
    main()

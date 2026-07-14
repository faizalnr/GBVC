import cv2
import numpy as np

def calculate_gradient(image, window_size=3):
    """
    Calculate the gradient for each pixel based on a neighborhood window.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray_image.shape
    gradient = np.zeros_like(gray_image, dtype=np.float64)
    pad_size = window_size // 2

    # Padding the image for boundary handling
    padded_image = cv2.copyMakeBorder(gray_image, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)

    for i in range(height):
        for j in range(width):
            # Extract the neighborhood
            neighborhood = padded_image[i:i + window_size, j:j + window_size]
            pixel_value = gray_image[i, j]
            # Calculate gradient
            gradient[i, j] = np.sum(np.abs(pixel_value - neighborhood)) / (window_size**2)

    return cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def generate_shares(image, gradient, n_shares=3, alpha_weights=None):
    """
    Generate shares using gradient information.
    """
    if alpha_weights is None:
        alpha_weights = [0.5] * (n_shares - 1)  # Default weights

    height, width, channels = image.shape
    shares = [np.zeros((height, width, channels), dtype=np.uint8) for _ in range(n_shares)]

    for c in range(channels):
        for i in range(height):
            for j in range(width):
                pixel = image[i, j, c]
                grad = gradient[i, j]
                random_values = [np.random.randint(0, 256) for _ in range(n_shares - 1)]
                # Generate n-1 shares
                for k in range(n_shares - 1):
                    shares[k][i, j, c] = (alpha_weights[k] * grad + random_values[k]) % 256
                # Generate the final share
                shares[-1][i, j, c] = (pixel - sum(shares[k][i, j, c] for k in range(n_shares - 1))) % 256

    return shares

def reconstruct_image(shares):
    """
    Reconstruct the image from its shares.
    """
    n_shares = len(shares)
    height, width, channels = shares[0].shape
    reconstructed = np.zeros((height, width, channels), dtype=np.uint8)

    for c in range(channels):
        for i in range(height):
            for j in range(width):
                pixel_sum = sum(share[i, j, c] for share in shares)
                reconstructed[i, j, c] = pixel_sum % 256

    return reconstructed

def main():
    # Load the input image
    input_image_path = "Taj.jpg"
    image = cv2.imread(input_image_path)

    if image is None:
        print(f"Error: Unable to load image at {input_image_path}")
        return

    # Calculate gradient
    gradient = calculate_gradient(image)

    # Generate shares
    n_shares = 3
    shares = generate_shares(image, gradient, n_shares=n_shares)

    # Reconstruct the image
    reconstructed_image = reconstruct_image(shares)

    # Display and save results
    cv2.imshow("Original Image", image)
    cv2.imshow("Gradient", gradient)
    for idx, share in enumerate(shares):
        cv2.imshow(f"Share {idx + 1}", share)
        cv2.imwrite(f"share_{idx + 1}.png", share)
    cv2.imshow("Reconstructed Image", reconstructed_image)
    cv2.imwrite("reconstructed_image.png", reconstructed_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

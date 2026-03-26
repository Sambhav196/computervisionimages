import numpy as np
from PIL import Image


def compress_grayscale(image_path, k):
    # Load image and convert to grayscale (L mode)
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

    # Perform Singular Value Decomposition
    # U: Left singular vectors, s: Singular values, Vt: Right singular vectors
    U, s, Vt = np.linalg.svd(img_array, full_matrices=False)

    # Reconstruct the image using only the first k singular values
    # The compressed size is proportional to k
    s_k = np.diag(s[:k])
    compressed_array = np.dot(U[:, :k], np.dot(s_k, Vt[:k, :]))

    # Clip values to valid 0-255 range and convert back to image
    compressed_array = np.clip(compressed_array, 0, 255).astype(np.uint8)
    return Image.fromarray(compressed_array)


# Usage
if __name__ == "__main__":
    # k defines the quality lower k = higher compression / lower quality
    compressed_img = compress_grayscale("image.jpeg", k=50)
    compressed_img.save("compressed_grayscale.jpg")
    compressed_img.show()

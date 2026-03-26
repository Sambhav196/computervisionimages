import cv2


def apply_adaptive_threshold(image_path):
    # Load the image in Grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error: Image not found.")
        return

    # Apply Adaptive Gaussian Thresholding
    # Parameters: (Source, MaxValue, Method, Type, BlockSize, Constant)
    binary_adaptive = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Compare with standard Global Thresholding (Otsu's method)
    _, binary_global = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display results
    cv2.imshow("Original", img)
    cv2.imshow("Global (Otsu)", binary_global)
    cv2.imshow("Adaptive Gaussian", binary_adaptive)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run the function
apply_adaptive_threshold("image.jpeg")

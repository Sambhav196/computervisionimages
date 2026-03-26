import cv2
import numpy as np


def compare_he_methods(image_path):
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Image not found.")
        return

    # Standard Global Histogram Equalization (The 'Flawed' Method)
    equ = cv2.equalizeHist(img)

    # clipLimit sets the threshold for contrast limiting
    # tileGridSize defines the size of local neighborhoods
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img)

    # Stack images horizontally for comparison: Original | Global HE | CLAHE
    res = np.hstack((img, equ, cl1))

    cv2.imshow("Original vs Global HE vs CLAHE", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Run the comparison
compare_he_methods("image.jpeg")

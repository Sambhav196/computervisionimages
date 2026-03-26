import cv2


def process_images(path1, path2):
    # Input two grayscale images
    img1 = cv2.imread(path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Error: Check file paths.")
        return

    # Ensure images are the same size for the AND operation
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

    # Bitwise operation on grayscale values
    img_and = cv2.bitwise_and(img1, img2)

    # Binarization using Otsu's Thresholding
    # This automatically finds the best threshold value
    _, img_binary = cv2.threshold(img_and, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display results
    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)
    cv2.imshow("Bitwise AND Result", img_and)
    cv2.imshow("Final Binarization", img_binary)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Usage
process_images("img1.png", "img2.png")

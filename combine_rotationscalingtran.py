import cv2


def transform_image(image_path):
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print("Error: Image not found.")
        return

    rows, cols = img.shape[:2]

    # 1. Define Transformation Parameters
    center = (cols // 2, rows // 2)  # Center of rotation
    angle = 45  # Degrees (Counter-clockwise)
    scale = 0.5  # 50% of original size
    tx, ty = 50, 30  # Translation offsets (pixels)

    # Get Rotation & Scaling Matrix (2x3)#
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # Add Translation to the Matrix
    M[0, 2] += tx
    M[1, 2] += ty

    # Apply the Combined Transformation
    transformed_img = cv2.warpAffine(img, M, (cols, rows))

    # Display results
    cv2.imshow("Original", img)
    cv2.imshow("Rotated, Scaled & Translated", transformed_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    transform_image("image.jpeg")

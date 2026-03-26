import numpy as np
from PIL import Image


# RGB to Grayscale
def rgb_to_gray(img):
    rgb = np.array(img)
    gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    return gray.astype(np.uint8)


# Grayscale to Binary
def gray_to_binary(gray, T=128):
    binary = np.zeros(gray.shape, dtype=np.uint8)
    binary[gray >= T] = 255
    return binary


# Main
img = Image.open("image.jpeg").convert("RGB")

gray = rgb_to_gray(img)
binary = gray_to_binary(gray, 128)

Image.fromarray(gray).save("gray.png")
Image.fromarray(binary).save("binary.png")

print("Conversion completed")

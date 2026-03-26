import cv2
import numpy as np

# Loading Input Image
img = cv2.imread('input.jpg', cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Error: 'input.jpg' not found in the current directory.")
    exit()

# Thresholding
# Using OTSU to handle light/dark variance; INV because cells are darker than background
_, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Removing Small Connected Components
min_area = 20 
n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

# Creating a clean mask
clean_mask = np.zeros(thresh.shape, dtype=np.uint8)
final_data = []

for i in range(1, n_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area >= min_area:
        clean_mask[labels == i] = 255
        final_data.append(centroids[i])

# Cell Labeling & Reporting
output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
print(f"Total Cells Detected: {len(final_data)}")

for idx, (cx, cy) in enumerate(final_data):
    # Converting float centroids to integers for drawing
    center = (int(cx), int(cy))
    # Drawing a small red circle at the centroid and the ID number
    cv2.circle(output_img, center, 3, (0, 0, 255), -1)
    cv2.putText(output_img, str(idx+1), (center[0]+5, center[1]), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    
    print(f"Cell {idx+1}: Location (x={cx:.2f}, y={cy:.2f})")

# Results
cv2.imwrite('segmented_cells.jpg', output_img)
cv2.imshow('Step C: Filtered Mask', clean_mask)
cv2.imshow('Step D: Labeled Centroids', output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
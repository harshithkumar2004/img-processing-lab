import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image in grayscale
image = cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)

# Check if image is loaded
if image is None:
    raise ValueError("Image not found. Make sure 'image3.jpg' is in the current directory.")

# Create a 3x3 kernel
kernel = np.ones((3, 3), np.uint8)

# Apply erosion
eroded = cv2.erode(image, kernel, iterations=1)

# Subtract eroded from original
edge_erosion = cv2.subtract(image, eroded)

# Apply dilation
dilated = cv2.dilate(image, kernel, iterations=1)

# Subtract original from dilated
edge_dilation = cv2.subtract(dilated, image)

# Plot all images
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title("Edge (Erosion Subtracted)")
plt.imshow(edge_erosion, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title("Edge (Dilation Subtracted)")
plt.imshow(edge_dilation, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()


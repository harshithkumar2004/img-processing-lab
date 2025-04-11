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

# Apply erosion and dilation
eroded = cv2.erode(image, kernel, iterations=1)
dilated = cv2.dilate(image, kernel, iterations=1)

# Subtractions
sub_erosion = cv2.subtract(image, eroded)
sub_dilation = cv2.subtract(dilated, image)

# Absolute differences
abs_erosion = cv2.absdiff(image, eroded)
abs_dilation = cv2.absdiff(dilated, image)

# Plotting
plt.figure(figsize=(15, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Sub: Original - Eroded")
plt.imshow(sub_erosion, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("AbsDiff: Original & Eroded")
plt.imshow(abs_erosion, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Sub: Dilated - Original")
plt.imshow(sub_dilation, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("AbsDiff: Dilated & Original")
plt.imshow(abs_dilation, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

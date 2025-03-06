import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('girl.jpg')
resized = cv2.resize(img, (300, 300))
cv2.imshow('Resized', resized)

# Rotation
(h, w) = resized.shape[:2]
center = (w // 2, h // 2)
M_rot = cv2.getRotationMatrix2D(center, 45, 1)  # Rotate by 45 degrees
rotated = cv2.warpAffine(resized, M_rot, (w, h))
cv2.imshow('Rotated', rotated)

# Scaling
scaled = cv2.resize(resized, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)
cv2.imshow('Scaled', scaled)

# Translation
M_trans = np.float32([[1, 0, 50], [0, 1, 50]])  # Shift image 50 pixels right and 50 pixels down
translated = cv2.warpAffine(resized, M_trans, (w, h))
cv2.imshow('Translated', translated)

plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
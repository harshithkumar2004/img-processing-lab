import cv2
img=cv2.imread('img1.jpg')

#cv2.imshow('Image', img)

resized = cv2.resize(img, (300, 300))
cv2.imshow('resized',resized)

flipped = cv2.flip(resized, 1)  
cv2.imshow('Flipped', flipped)

blurred = cv2.GaussianBlur(resized, (7, 7), 0)  
cv2.imshow('Blurred', blurred) 

(h, w) = resized.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, 45, 1) 
rotated = cv2.warpAffine(resized, M, (w, h))
cv2.imshow('Rotated', rotated)  

gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray)

#edge detection 
edges = cv2.Canny(gray, 100, 200)  
cv2.imshow('Edges', edges) 

_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Thresholded', thresh) 

(h, w) = resized.shape[:2]
center_x, center_y = w // 2, h // 2
cv2.line(resized, (center_x, 0), (center_x, h), (0, 0, 0), 2)
cv2.line(resized, (0, center_y), (w, center_y), (0, 0, 0), 2)
cv2.imshow('Quadrants', resized)

cv2.waitKey(0)
cv2.destroyAllWindows()
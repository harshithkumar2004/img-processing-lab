import cv2
import numpy as np

def main():
   
    image = cv2.imread('image3.jpg', cv2.IMREAD_GRAYSCALE)
    if image is None:
        print("Error: Image not found!")
        return
    
   
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    
   
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = cv2.magnitude(sobel_x, sobel_y)
    
   
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    
    sobel_combined = cv2.convertScaleAbs(sobel_combined)
    laplacian = cv2.convertScaleAbs(laplacian)
    
    
    canny_edges = cv2.Canny(image, 100, 200)
    
    median_filtered = cv2.medianBlur(image, 5)
    
    bilateral_filtered = cv2.bilateralFilter(image, 9, 75, 75)

    cv2.imshow('Original Image', image)
    cv2.imshow('Blurred Image', blurred)
    cv2.imshow('Sobel Edge Detection', sobel_combined)
    cv2.imshow('Laplacian Edge Detection', laplacian)
    cv2.imshow('Canny Edge Detection', canny_edges)
    cv2.imshow('Median Filtered', median_filtered)
    cv2.imshow('Bilateral Filtered', bilateral_filtered)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


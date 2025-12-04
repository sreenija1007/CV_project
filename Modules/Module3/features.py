"""
-------------------------------------------------------------------------
Assignment: Module 3 - Features & Segmentation
File Name: features.py
Author: Sreenija Kanugonda
-------------------------------------------------------------------------
Description:
    This module performs fundamental feature extraction and segmentation tasks:
    1. Gradient Analysis: Calculates Magnitude and Angle using Sobel operators,
       and computes the Laplacian of Gaussian (LoG).
    2. Keypoint Detection: Detects edges using Canny and corners using Harris.
    3. Boundary Detection: basic segmentation using Otsu's thresholding 
       and Contour extraction.

Usage:
    Import this module into the main application script:
    from Module3 import features
-------------------------------------------------------------------------
"""

import cv2
import numpy as np

def get_gradient_and_log(image):
    """
    Task 1: Returns Magnitude, Angle, and Laplacian (LoG) images.
    """
    if image is None:
        return None, None, None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Gradient (Magnitude & Angle)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    magnitude, angle = cv2.cartToPolar(sobel_x, sobel_y, angleInDegrees=True)
    
    # Normalize for display (0-255)
    mag_img = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    angle_img = cv2.normalize(angle, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    # 2. Laplacian of Gaussian (LoG)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    log_image = cv2.Laplacian(blurred, cv2.CV_64F)
    log_img_vis = cv2.normalize(np.abs(log_image), None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return mag_img, angle_img, log_img_vis

def get_keypoints(image):
    """
    Task 2: Returns Canny Edges and Harris Corners images.
    """
    if image is None:
        return None, None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Canny Edges
    canny_edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    
    # 2. Harris Corners
    gray_float = np.float32(gray)
    dst = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04)
    dst_dilated = cv2.dilate(dst, None)
    
    # Draw red dots on a COPY of the image
    corner_image = image.copy()
    corner_image[dst_dilated > 0.01 * dst_dilated.max()] = [0, 0, 255] # Red dots
    
    return canny_edges, corner_image

def get_boundary(image):
    """
    Task 3: Returns Threshold Mask and Object Boundary image.
    """
    if image is None:
        return None, None
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 1. Thresholding
    (T, thresh_mask) = cv2.threshold(blurred, 0, 255, 
                                 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # 2. Contours (Boundary)
    contours, _ = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boundary_image = image.copy()
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(boundary_image, [largest_contour], -1, (0, 255, 0), 3) # Green Line
        
    return thresh_mask, boundary_image
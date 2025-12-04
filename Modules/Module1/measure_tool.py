"""
-------------------------------------------------------------------------
Assignment: Module 1 - Single View Metrology
File Name: measure_tool.py
Author: Sreenija Kanugonda
-------------------------------------------------------------------------
Description:
    This utility script handles the core mathematical operations for 
    Single View Metrology. It includes functions to:
    1. Load camera calibration data (.npz files).
    2. Undistort images to correct for lens distortion.
    3. Calculate real-world physical dimensions from 2D pixel distances 
       using the pinhole camera model formula: H = (Z * h) / f.

Usage:
    Import this module into the main application script:
    from Module1 import measure_tool
-------------------------------------------------------------------------
"""

import cv2
import numpy as np
import os

def get_calibration_data(file_path="calibration_data_mac.npz"):
    """
    Loads the calibration file.
    Returns mtx, dist, and calculated focal_length (f).
    """
    # Check if file exists to prevent crashing
    if not os.path.exists(file_path):
        return None, None, None

    data = np.load(file_path)
    mtx, dist = data["mtx"], data["dist"]
    
    # Calculate Focal Length exactly as you did in your script
    f = (mtx[0,0] + mtx[1,1]) / 2
    return mtx, dist, f

def undistort_image(img, mtx, dist):
    """
    Applies the undistortion to the image using the loaded matrices.
    """
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    undistorted = cv2.undistort(img, mtx, dist, None, newcameramtx)
    return undistorted

def calculate_real_distance(p1, p2, Z, f):
    """
    Your specific math formula.
    p1, p2: Coordinates (x,y)
    Z: Distance from camera
    f: Focal length
    """
    # Calculate pixel distance
    px_dist = np.linalg.norm(np.array(p1) - np.array(p2))
    
    # Calculate real size
    real_size = (Z * px_dist) / f
    return real_size
"""
-------------------------------------------------------------------------
Assignment: Module 3 - Features & Segmentation (Advanced Tasks)
File Name: cv_processors.py
Author: Sreenija Kanugonda
-------------------------------------------------------------------------
Description:
    This module handles advanced segmentation tasks to compare Geometric methods
    against Deep Learning methods:
    
    1. ArUco Dictionary Auto-Detection: Automatically determines which ArUco 
       dictionary (4x4, 5x5, 6x6) is used in the image to prevent detection errors.
    
    2. Task 4 (Geometric): Computes the Convex Hull (smallest polygon enclosing 
       a set of points) based on detected ArUco marker corners.
    
    3. Task 5 (AI - SAM2): Uses Meta's Segment Anything Model 2. It utilizes 
       detected ArUco markers as "point prompts" to guide the AI in segmenting 
       the specific object.

Execution:
    Imported and used by the main 'app.py' script.
    Requires 'sam2' and 'torch' installed.
-------------------------------------------------------------------------
"""

import cv2
import numpy as np
import torch

# --- Helper: Get ArUco Detector (Auto-Detect Dictionary) ---
def get_aruco_detector(img=None):
    """
    Tries different ArUco dictionaries to see which one works for the input image.
    Defaults to DICT_6X6_250 if no image is provided or no markers found.
    """
    # List of common dictionaries to check
    possible_dicts = [
        cv2.aruco.DICT_4X4_50,       
        cv2.aruco.DICT_5X5_100,
        cv2.aruco.DICT_6X6_250,       # The original default
        cv2.aruco.DICT_ARUCO_ORIGINAL
    ]
    
    parameters = cv2.aruco.DetectorParameters()

    # If we have an image, test which dictionary finds markers
    if img is not None:
        for dict_id in possible_dicts:
            try:
                aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
                detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
                corners, ids, _ = detector.detectMarkers(img)
                
                if corners:
                    # Found markers with this dictionary! Use it.
                    # print(f"DEBUG: Auto-detected dictionary ID: {dict_id}")
                    return detector
            except Exception:
                continue

    # Fallback default (6x6)
    default_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    return cv2.aruco.ArucoDetector(default_dict, parameters)

# --- Task 4: ArUco Convex Hull ---
def process_aruco_boundary(img):
    # Ensure image is valid
    if img is None: return None, "Image is empty"
    
    # Auto-detect which dictionary (4x4, 6x6, etc.) is being used
    detector = get_aruco_detector(img)
    corners, ids, rejected = detector.detectMarkers(img)
    
    if not corners:
        return None, "No ArUco markers found. Ensure they have a white border and are visible."

    # Flatten corners into a single list of points
    all_corners = [pt for marker in corners for pt in marker[0]]
    points = np.array(all_corners, dtype=np.int32)

    # Find Convex Hull (The "Rubber Band" boundary)
    hull_indices = cv2.convexHull(points, returnPoints=False)
    hull_points = points[hull_indices.squeeze()]

    # Draw Results
    result = img.copy()
    cv2.aruco.drawDetectedMarkers(result, corners, ids)
    cv2.drawContours(result, [hull_points], -1, (255, 0, 0), 3) # Blue Line

    return result, None

# --- Task 5: SAM2 Segmentation ---
def process_sam2_segmentation(img, predictor):
    if img is None: return None, "Image is empty"
    
    # 1. Find Markers
    detector = get_aruco_detector(img)
    corners, ids, _ = detector.detectMarkers(img)
    if not corners: return None, "No markers found."

    # 2. Get Centers
    prompt_points = []
    for marker in corners:
        M = cv2.moments(marker[0])
        if M["m00"] != 0:
            prompt_points.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
    
    # 3. Predict
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)
        
        masks, scores, _ = predictor.predict(
            point_coords=np.array(prompt_points),
            point_labels=np.ones(len(prompt_points)),
            multimask_output=True
        )

        best_mask = masks[2] 

        # 4. Overlay
        overlay = img.copy()
        overlay[best_mask > 0] = (0, 0, 255) # Red Mask
        vis = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
        
        # Draw Prompts
        for pt in prompt_points:
            cv2.circle(vis, tuple(pt), 6, (0, 255, 0), -1)
            
        return vis, None

    except Exception as e:
        return None, f"SAM2 Error: {str(e)}"
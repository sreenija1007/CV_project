"""
-------------------------------------------------------------------------
Assignment: Module 4 - Image Stitching (Task 1)
File Name: panorama.py
Author: Sreenija Kanugonda
-------------------------------------------------------------------------
Description:
    This module implements vertical panorama stitching.
    It takes a sequence of images (ordered from Bottom to Top), detects features
    using SIFT, aligns them using Homography, and blends them into a single
    continuous image.

    Key Algorithm Steps:
    1. Feature Detection: SIFT (Scale-Invariant Feature Transform).
    2. Matching: FLANN or BFMatcher with Lowe's Ratio Test.
    3. Alignment: Find Homography using RANSAC.
    4. Warping: Warps the accumulated panorama to the plane of the new image.
    5. Post-processing: Auto-cropping to remove black artifacts.

Usage:
    Import this module into the main application script:
    from Module4.task1 import panorama
-------------------------------------------------------------------------
"""

import cv2
import numpy as np

def stitch_pair(img_bottom, img_top):
    """
    Stitches two images together vertically.
    img_bottom is the accumulated panorama so far.
    img_top is the new image to be added above.
    Warps img_bottom to match img_top's perspective.
    """
    # 1. Feature Detection
    sift = cv2.SIFT_create()
    kp_top, des_top = sift.detectAndCompute(img_top, None)
    kp_bottom, des_bottom = sift.detectAndCompute(img_bottom, None)

    if des_top is None or des_bottom is None:
        return None

    # 2. Match Features
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_bottom, des_top, k=2)

    # 3. Lowe's Ratio Test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return None

    # 4. Find Homography
    # We map Bottom (src) -> Top (dst)
    src_pts = np.float32([kp_bottom[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_top[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        return None

    # 5. Warp and Stitch
    h_top, w_top = img_top.shape[:2]
    h_bottom, w_bottom = img_bottom.shape[:2]

    # Calculate canvas size
    # User's logic: Simple sum of heights, max of widths.
    # This works well if images are vertically aligned.
    canvas_height = h_top + h_bottom
    canvas_width = max(w_top, w_bottom)

    # Warp the bottom image (accumulated pano) to the top image's plane
    dst = cv2.warpPerspective(img_bottom, H, (canvas_width, canvas_height))
    
    # Overlay the top image (unwarped) at the origin (0,0)
    # Since H mapped bottom -> top, top is at 0,0 relative to itself.
    dst[0:h_top, 0:w_top] = img_top

    # 6. Crop black padding
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        cropped_result = dst[y:y+h, x:x+w]
        return cropped_result
    else:
        return dst

def stitch_vertical_sequence(images):
    """
    Main handler function for the Streamlit app.
    Accepts a list of images (Bottom -> Top).
    """
    if len(images) < 2:
        return images[0] if images else None, "Need at least 2 images."

    # Start with the bottom-most image
    current_panorama = images[0]

    # Iteratively stitch the next image on top
    for i in range(1, len(images)):
        img_top = images[i]
        
        # Apply the user's specific logic: stitch(accumulated, new_top)
        result = stitch_pair(current_panorama, img_top)
        
        if result is not None:
            current_panorama = result
        else:
            return current_panorama, f"Stitching failed at image {i+1} (matches insufficient or homography failed)."

    return current_panorama, "Stitching Complete!"
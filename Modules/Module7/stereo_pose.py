"""
-------------------------------------------------------------------------
Assignment: Module 7 - Stereo Vision & Pose Estimation
File Name: stereo_pose.py
Author: Sreenija Kanugonda
-------------------------------------------------------------------------
Description:
    This module combines two advanced Computer Vision tasks:
    
    1. Stereo Vision (Depth Estimation):
       - Uses the principle of Disparity to calculate depth from two 
         aligned images (Left and Right views).
       - Formula: Z = (f * Baseline) / Disparity.
       - Also calculates real-world dimensions (Height/Width) using the 
         estimated depth.

    2. Pose & Hand Estimation (MediaPipe):
       - Utilizes Google's MediaPipe framework for real-time skeletal tracking.
       - Extracts (x, y, z) coordinates for 33 body landmarks and 21 hand landmarks.
       - Visualizes connections and logging data for analysis.

Usage:
    Import this module into the main application script:
    from Module7 import stereo_pose
-------------------------------------------------------------------------
"""

import cv2
import numpy as np
import mediapipe as mp
import os

# ==========================================
# PART 1: STEREO MEASUREMENT LOGIC
# ==========================================
def load_calibration(file_path="calibration_data_mac.npz"):
    """Loads calibration data or returns default focal length."""
    if os.path.exists(file_path):
        try:
            data = np.load(file_path)
            mtx = data['mtx']
            # Focal length is usually fx (element 0,0)
            f = mtx[0, 0] 
            return f
        except:
            return 1250.0 # Fallback from your script
    return 1250.0

def calculate_stereo_metrics(points_L, points_R, baseline, f):
    """
    Calculates depth and real-world dimensions based on 3 pairs of points.
    Expected Order: [Top-Left, Top-Right, Bottom-Left]
    """
    if len(points_L) != 3 or len(points_R) != 3:
        return None, "Please click exactly 3 points on BOTH images."

    # Unpack points
    (xL1, yL1) = points_L[0]; (xR1, yR1) = points_R[0] # Top-Left
    (xL2, yL2) = points_L[1]; (xR2, yR2) = points_R[1] # Top-Right
    (xL3, yL3) = points_L[2]; (xR3, yR3) = points_R[2] # Bottom-Left

    # 1. Calculate Disparities (Difference in X)
    d1 = abs(xL1 - xR1)
    d2 = abs(xL2 - xR2)
    d3 = abs(xL3 - xR3)

    if 0 in [d1, d2, d3]:
        return None, "Disparity is 0. Points are too aligned vertically. Try again."

    # 2. Calculate Depths (Z = f * B / d)
    z1 = (f * baseline) / d1
    z2 = (f * baseline) / d2
    z3 = (f * baseline) / d3
    z_avg = (z1 + z2 + z3) / 3

    # 3. Calculate Pixel Dimensions (Using Left Image)
    # Width: Distance P1 -> P2
    pix_w = np.sqrt((xL1 - xL2)**2 + (yL1 - yL2)**2)
    # Height: Distance P1 -> P3
    pix_h = np.sqrt((xL1 - xL3)**2 + (yL1 - yL3)**2)

    # 4. Calculate Real Dimensions
    real_w = (pix_w * z_avg) / f
    real_h = (pix_h * z_avg) / f

    results = {
        "z_avg": round(z_avg, 2),
        "real_w": round(real_w, 2),
        "real_h": round(real_h, 2)
    }
    return results, "Success"

# ==========================================
# PART 2: POSE & HAND TRACKING LOGIC
# ==========================================
def process_pose(image, pose_model):
    """
    Takes an image and a MediaPipe Pose model instance.
    Returns: Annotated Image, Data List (for CSV)
    """
    # Convert to RGB
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process
    results = pose_model.process(image_rgb)
    
    # Draw
    image.flags.writeable = True
    annotated_image = image.copy() # Work on a copy
    
    data_rows = []
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    
    if results.pose_landmarks:
        # 1. Draw visual skeleton
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
        
        # 2. Extract Data
        landmark_names = [name.name for name in mp_pose.PoseLandmark]
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            data_rows.append({
                "landmark_id": idx,
                "name": landmark_names[idx],
                "x": landmark.x,
                "y": landmark.y,
                "z": landmark.z,
                "visibility": landmark.visibility
            })
            
    return annotated_image, data_rows

def process_hands(image, hands_model):
    """
    Takes an image and a MediaPipe Hands model instance.
    Returns: Annotated Image, Data List (for CSV)
    """
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = hands_model.process(image_rgb)
    
    image.flags.writeable = True
    annotated_image = image.copy()
    
    data_rows = []
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Draw
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
            )
            
            # Extract Data
            handedness = results.multi_handedness[hand_idx].classification[0].label
            landmark_names = [name.name for name in mp_hands.HandLandmark]
            
            for idx, landmark in enumerate(hand_landmarks.landmark):
                data_rows.append({
                    "hand_id": hand_idx,
                    "handedness": handedness,
                    "landmark_id": idx,
                    "name": landmark_names[idx],
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z
                })
                
    return annotated_image, data_rows
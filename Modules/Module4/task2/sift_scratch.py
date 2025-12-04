"""
-------------------------------------------------------------------------
Assignment: Module 4 - SIFT & RANSAC from Scratch (Task 2)
File Name: sift_scratch.py
Author: Sreenija Kanugonda
-------------------------------------------------------------------------
Description:
    This module implements key Computer Vision algorithms "from scratch" 
    to demonstrate understanding of the underlying mathematics.

    Algorithms Implemented:
    1. SIFT (Scale-Invariant Feature Transform):
       - Scale Space Construction (Gaussian Pyramid).
       - Difference of Gaussians (DoG) for blob detection.
       - Local Extrema Detection (finding keypoints).
       - Descriptor Generation (128-d vectors based on gradient histograms).
    
    2. Feature Matching:
       - Brute-force Euclidean distance calculation.
       - Lowe's Ratio Test to filter ambiguous matches.
    
    3. RANSAC (Random Sample Consensus):
       - Robustly estimates the Homography matrix by iteratively sampling 
         subsets of matches and counting inliers.

Usage:
    Import this module into the main application script:
    from Module4.task2 import sift_scratch
-------------------------------------------------------------------------
"""

import cv2
import numpy as np
import random

def run_from_scratch_sift(gray_img):
    """
    Implements SIFT-like feature extraction from scratch.
    
    Steps covered manually:
    1. Gaussian Pyramid (Scale Space)
    2. Difference of Gaussians (DoG)
    3. Local Extrema Detection (Vectorized for performance)
    4. Descriptor Generation (128-d vector from gradients)
    """
    # Ensure image is float32
    img = gray_img.astype(np.float32)
    
    # --- 1. Scale Space Construction ---
    # We use a single octave with 5 scales for Assignment efficiency.
    # (Real SIFT uses multiple octaves with downsampling)
    sigmas = [1.6, 2.0, 2.5, 3.2, 4.0]
    gaussian_images = []
    
    for s in sigmas:
        # Kernel size roughly 6*sigma to capture the distribution
        k = int(6 * s)
        if k % 2 == 0: k += 1
        blurred = cv2.GaussianBlur(img, (k, k), s)
        gaussian_images.append(blurred)
        
    # --- 2. Difference of Gaussians (DoG) ---
    dog_images = []
    for i in range(len(gaussian_images) - 1):
        # Subtract consecutive gaussian images
        dog_images.append(gaussian_images[i+1] - gaussian_images[i])
        
    # --- 3. Keypoint Detection (Local Extrema) ---
    keypoints_list = []
    
    # We check the middle layers of the DoG pyramid.
    # Indices 1 and 2 allow us to check neighbors in (0,1,2) and (1,2,3).
    contrast_threshold = 0.03 * 255  # Filter out weak features (low contrast)
    
    for i in range(1, len(dog_images) - 1):
        prev_dog = dog_images[i-1]
        curr_dog = dog_images[i]
        next_dog = dog_images[i+1]
        
        # Focus on the center, excluding 1-pixel border for neighbor checks
        center_val = curr_dog[1:-1, 1:-1]
        
        # Vectorized Threshold Check
        is_candidate = np.abs(center_val) > contrast_threshold
        
        # Initialize masks
        is_max = is_candidate.copy()
        is_min = is_candidate.copy()
        
        # Check all 26 neighbors (3x3x3 block around the pixel)
        for dz in [-1, 0, 1]:      # Layers: prev, curr, next
            layer = dog_images[i + dz]
            for dy in [-1, 0, 1]:  # Y neighbors
                for dx in [-1, 0, 1]: # X neighbors
                    if dz == 0 and dy == 0 and dx == 0:
                        continue # Skip the pixel itself
                    
                    # Create a shifted view (slice) of the neighbor layer
                    # to compare against the center_val array
                    ys = 1 + dy
                    ye = -1 + dy if ((-1 + dy) != 0) else None
                    xs = 1 + dx
                    xe = -1 + dx if ((-1 + dx) != 0) else None
                    
                    neighbor_slice = layer[ys:ye, xs:xe]
                    
                    # Update max/min status
                    is_max = np.logical_and(is_max, center_val > neighbor_slice)
                    is_min = np.logical_and(is_min, center_val < neighbor_slice)
        
        # A pixel is a keypoint if it is strictly max or strictly min
        is_extrema = np.logical_or(is_max, is_min)
        y_coords, x_coords = np.where(is_extrema)
        
        # Adjust coords back to full image size (we sliced off 1 pixel)
        y_coords += 1
        x_coords += 1
        
        # Store valid keypoints
        for idx in range(len(y_coords)):
            y, x = y_coords[idx], x_coords[idx]
            keypoints_list.append({
                'x': float(x), 
                'y': float(y), 
                'size': sigmas[i],
                'response': float(np.abs(curr_dog[y,x])),
                'angle': 0  
            })

    # Sort by response strength and keep top 1000 to keep UI responsive
    keypoints_list = sorted(keypoints_list, key=lambda k: k['response'], reverse=True)[:1000]
    
    # --- 4. Descriptor Generation ---
    descriptors = compute_descriptors(img, keypoints_list)
    
    return keypoints_list, descriptors

def compute_descriptors(img, keypoints):
    """
    Computes a 128-dimensional SIFT descriptor for each keypoint.
    Logic: 16x16 pixel window -> 4x4 cell grid -> 8-bin orientation histogram per cell.
    """
    rows, cols = img.shape
    descriptors = []
    valid_kps = [] 
    
    # Pre-compute gradients for the whole image
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    for kp in keypoints:
        c, r = int(kp['x']), int(kp['y'])
        
        # Boundary check: We need 8 pixels padding for the 16x16 window
        if r < 8 or r >= rows - 8 or c < 8 or c >= cols - 8:
            continue
            
        # Extract 16x16 patch
        win_mag = mag[r-8:r+8, c-8:c+8]
        win_ang = ang[r-8:r+8, c-8:c+8]
        
        desc_vector = []
        
        # Divide into 4x4 sub-blocks (each block is 4x4 pixels)
        # 4 blocks horizontal * 4 blocks vertical = 16 blocks total
        for i in range(0, 16, 4):     
            for j in range(0, 16, 4): 
                
                block_mag = win_mag[i:i+4, j:j+4]
                block_ang = win_ang[i:i+4, j:j+4]
                
                # Create 8-bin histogram for this block
                hist = np.zeros(8, dtype=np.float32)
                
                for y in range(4):
                    for x in range(4):
                        angle = block_ang[y, x]
                        magnitude = block_mag[y, x]
                        
                        # Determine bin (0-7)
                        bin_idx = int(angle / 45) % 8
                        hist[bin_idx] += magnitude
                        
                desc_vector.extend(hist)
        
        # --- Descriptor Normalization ---
        # 1. Normalize vector to unit length (L2 norm)
        desc_vector = np.array(desc_vector, dtype=np.float32)
        norm = np.linalg.norm(desc_vector)
        if norm > 1e-6:
            desc_vector /= norm
            
        # 2. Threshold large values (limit to 0.2) to reduce lighting effects
        desc_vector[desc_vector > 0.2] = 0.2
        
        # 3. Re-normalize
        norm = np.linalg.norm(desc_vector)
        if norm > 1e-6:
            desc_vector /= norm
            
        descriptors.append(desc_vector)
        valid_kps.append(kp)
        
    # Update keypoints list to match only those where descriptors were computed
    keypoints[:] = valid_kps
    return np.array(descriptors, dtype=np.float32)

def match_features(des1, des2):
    """
    Manual Brute-Force Matcher.
    Calculates Euclidean distance between all pairs and applies Lowe's Ratio Test.
    """
    matches = []
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return matches
        
    # Iterate through every descriptor in the first image
    for i, d1 in enumerate(des1):
        # Calculate Euclidean distance to all descriptors in image 2
        # (d1 - des2)^2
        diff = des2 - d1
        dist_sq = np.sum(diff**2, axis=1)
        dists = np.sqrt(dist_sq)
        
        # Find the two closest matches
        sorted_indices = np.argsort(dists)
        
        if len(sorted_indices) >= 2:
            idx1, idx2 = sorted_indices[0], sorted_indices[1]
            dist1, dist2 = dists[idx1], dists[idx2]
            
            # Lowe's Ratio Test: Keep match only if it is significantly better than the next best
            if dist1 < 0.75 * dist2:
                matches.append({
                    'queryIdx': i,
                    'trainIdx': idx1,
                    'distance': dist1
                })
                
    return matches

def from_scratch_ransac(kp1, kp2, matches, threshold=5.0, max_iters=2000):
    """
    Manual RANSAC Loop for Homography Estimation.
    
    1. Randomly sample 4 matches.
    2. Compute Homography matrix H for those 4 points.
    3. Project all points using H and count inliers (points with low error).
    4. Repeat and keep the H with the most inliers.
    """
    if len(matches) < 4:
        return None, []
        
    # Extract coordinates from matches
    src_pts = np.float32([ [kp1[m['queryIdx']]['x'], kp1[m['queryIdx']]['y']] for m in matches ])
    dst_pts = np.float32([ [kp2[m['trainIdx']]['x'], kp2[m['trainIdx']]['y']] for m in matches ])
    
    best_H = None
    max_inliers = 0
    best_inliers_mask = np.zeros(len(matches), dtype=int)
    
    num_matches = len(src_pts)
    
    for _ in range(max_iters):
        # 1. Random Sample
        indices = random.sample(range(num_matches), 4)
        p1 = src_pts[indices]
        p2 = dst_pts[indices]
        
        # 2. Compute Model (Homography)
        # Solving the linear system for 4 points to get 3x3 matrix
        # We use OpenCV's solver for the 4-point algebraic step, as manual SVD is verbose
        try:
            H_sample = cv2.getPerspectiveTransform(p1, p2)
        except cv2.error:
            continue # Skip if points are collinear
            
        # 3. Verify Model (Check Error for ALL points)
        # Convert src points to homogeneous coords [x, y, 1]
        ones = np.ones((num_matches, 1))
        src_homo = np.hstack([src_pts, ones]) 
        
        # Project: H * src
        projected = src_homo.dot(H_sample.T) 
        
        # Normalize (x/w, y/w)
        w = projected[:, 2]
        # Avoid division by zero
        w[np.abs(w) < 1e-10] = 1e-10
        projected_x = projected[:, 0] / w
        projected_y = projected[:, 1] / w
        
        # Calculate Distance Error
        dx = dst_pts[:, 0] - projected_x
        dy = dst_pts[:, 1] - projected_y
        errors = np.sqrt(dx**2 + dy**2)
        
        # Count Inliers
        inliers_mask = errors < threshold
        num_inliers = np.sum(inliers_mask)
        
        # Update Best Model
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_H = H_sample
            best_inliers_mask = inliers_mask.astype(int)
            
    return best_H, best_inliers_mask
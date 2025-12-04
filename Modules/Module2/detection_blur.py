import cv2
import numpy as np
import os

def load_templates(template_dir):
    """
    Loads all images from the specified directory.
    Returns: list of (filename, grayscale_image)
    """
    templates = []
    if not os.path.exists(template_dir):
        return []
        
    for filename in os.listdir(template_dir):
        if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            path = os.path.join(template_dir, filename)
            # Load as grayscale for matching
            img = cv2.imread(path, 0)
            if img is not None:
                templates.append((filename, img))
    return templates

def process_channel_fourier(channel):
    """
    Helper function: Blurs and then Deblurs a single 2D image channel 
    using Fourier Transform.
    """
    h, w = channel.shape
    
    # 1. Create a Gaussian Kernel (The "Blur Function")
    # We use a specific size/sigma so we know exactly what to reverse later
    ksize = 95
    sigma = 20
    k_1d = cv2.getGaussianKernel(ksize, sigma)
    kernel_spatial = k_1d @ k_1d.T  # Make it 2D
    
    # 2. Prepare Kernel for FFT (Pad to image size)
    kernel_padded = np.zeros_like(channel, dtype=np.float32)
    kh, kw = kernel_spatial.shape
    
    # Center the kernel padding
    pad_h, pad_w = (h - kh) // 2, (w - kw) // 2
    kernel_padded[pad_h:pad_h+kh, pad_w:pad_w+kw] = kernel_spatial
    
    # Shift center to (0,0) for correct phase in FFT
    kernel_shifted = np.fft.ifftshift(kernel_padded)
    
    # 3. Apply FFT
    fft_image = np.fft.fft2(channel.astype(float))
    fft_kernel = np.fft.fft2(kernel_shifted)
    
    # 4. BLUR (Convolution Theorem: Image * Kernel)
    fft_blurred = fft_image * fft_kernel
    blurred_channel = np.abs(np.fft.ifft2(fft_blurred)).astype(np.uint8)
    
    # 5. DEBLUR (Inverse Filter: Blurred / Kernel)
    epsilon = 1e-5  # Prevent division by zero
    recovered_fft = fft_blurred / (fft_kernel + epsilon)
    recovered_channel = np.abs(np.fft.ifft2(recovered_fft)).astype(np.uint8)
    
    return blurred_channel, recovered_channel

def detect_and_process_pipeline(scene_color, templates, threshold=0.5):
    """
    1. Finds the SINGLE best match across all templates.
    2. Blurs that region using FFT.
    3. Deblurs that region using Inverse FFT.
    
    Returns:
        detection_img (with box), 
        blurred_img (only region blurred), 
        restored_img (only region restored),
        match_info (string)
    """
    scene_gray = cv2.cvtColor(scene_color, cv2.COLOR_BGR2GRAY)
    s_h, s_w = scene_gray.shape[:2]
    
    # Variables to track the single best match
    best_score = -1
    best_match = None  # Will store (label, top_left, size_w, size_h)

    # --- STEP 1: FIND BEST MATCH ---
    for temp_name, temp_img in templates:
        t_h, t_w = temp_img.shape[:2]
        
        # Auto-Resize Check (Prevent crash)
        if t_h >= s_h or t_w >= s_w:
            scale = min((s_h * 0.5)/t_h, (s_w * 0.5)/t_w)
            temp_img = cv2.resize(temp_img, None, fx=scale, fy=scale)
            t_h, t_w = temp_img.shape[:2]
            
        try:
            res = cv2.matchTemplate(scene_gray, temp_img, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # Update best match if this one is better
            if max_val > best_score:
                best_score = max_val
                best_match = (temp_name, max_loc, t_w, t_h)
                
        except Exception:
            continue

    # Initialize outputs as copies of original
    img_detection = scene_color.copy()
    img_blurred = scene_color.copy()
    img_restored = scene_color.copy()
    info_text = "No detection found."

    # --- STEP 2: PROCESS THE REGION ---
    if best_score >= threshold and best_match is not None:
        name, top_left, w, h = best_match
        bottom_right = (top_left[0] + w, top_left[1] + h)
        info_text = f"Detected: {name} (Score: {best_score:.2f})"
        
        # 1. Draw Box on Detection Image
        cv2.rectangle(img_detection, top_left, bottom_right, (0, 255, 0), 3)
        cv2.putText(img_detection, name, (top_left[0], top_left[1]-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # 2. Extract ROI
        roi = scene_color[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        
        if roi.size > 0:
            # Split into B, G, R channels
            b, g, r = cv2.split(roi)
            
            # Apply Fourier Blur & Restore to each channel
            b_blur, b_rec = process_channel_fourier(b)
            g_blur, g_rec = process_channel_fourier(g)
            r_blur, r_rec = process_channel_fourier(r)
            
            # Merge back
            roi_blurred = cv2.merge([b_blur, g_blur, r_blur])
            roi_restored = cv2.merge([b_rec, g_rec, r_rec])
            
            # 3. Paste blurred ROI into Blur Image
            img_blurred[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = roi_blurred
            # Draw a box (optional, usually good to show where it happened)
            cv2.rectangle(img_blurred, top_left, bottom_right, (0, 255, 0), 2)
            
            # 4. Paste restored ROI into Restore Image
            img_restored[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] = roi_restored
            cv2.rectangle(img_restored, top_left, bottom_right, (0, 255, 0), 2)

    return img_detection, img_blurred, img_restored, info_text
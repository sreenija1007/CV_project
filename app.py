"""
-------------------------------------------------------------------------
Assignment: Computer Vision Portfolio Dashboard (CSC 8830)
File Name: app.py
Author: Sreenija Kanugonda
-------------------------------------------------------------------------
Description:
    This is the main entry point (Orchestrator) for the Computer Vision Portfolio.
    It is a Streamlit-based web application that unifies multiple CV assignments
    into a single interactive dashboard.
    
    Modules Included:
    1. Single View Metrology (Measurement from a single image)
    2. Object Detection & Fourier Restoration
    3. Features & Segmentation (including SAM2 integration)
    4. Image Stitching (Panorama) & SIFT from Scratch
    5. Motion Tracking (ArUco, Markerless, & SAM2)
    6. Stereo Vision & Pose Estimation (MediaPipe)

    Key Features:
    - Interactive Canvas: Users can draw on images for measurements/tracking.
    - Deep Learning: Integration of Meta's SAM2 model.

Execution Instructions:
    1. Dependencies: Install the required libraries.
       Command: pip install streamlit opencv-python numpy pandas streamlit-drawable-canvas Pillow torch face_recognition mediapipe
       *Note: SAM2 must be installed from the official Meta repository.*
    
    2. Prerequisites:
       - Ensure the 'modules' directory is present in the same folder.
       - Ensure 'calibration_data_mac.npz' is present for metrology tasks.

    3. Execution: Run the application using Streamlit.
       Command: streamlit run app.py
-------------------------------------------------------------------------
"""

import os
import streamlit as st
import cv2
import numpy as np
import pandas as pd 
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import sys
import torch 
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Ensure 'modules' folder is in the python path
if "Modules" not in sys.path:
    sys.path.append("modules")

# Import Custom Modules
from Module1 import measure_tool
from Module2 import detection_blur
from Module3 import features 
from Module3 import cv_processors
from Module4.task1 import panorama
from Module4.task2 import sift_scratch
from Module5_6 import tracker
from Module7 import stereo_pose
import mediapipe as mp

# PAGE CONFIGURATION
st.set_page_config(page_title="CV Dashboard", layout="wide")

@st.cache_resource
def load_sam2_model():
    # Detect Hardware (Mac MPS, NVIDIA CUDA, or CPU)
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        
    print(f"Loading SAM2 on {device}... (This may take a minute)")
    try:
        # Download/Load the model
        model = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-large", device=device)
        return model
    except Exception as e:
        print(f"Error loading SAM2: {e}")
        return None

# SIDEBAR NAVIGATION
st.sidebar.title("CV Modules Dashboard")
module_selection = st.sidebar.radio(
    "Go to:",
    [
        "Home",
        "1. Single View Metrology",
        "2. Object Detection (Templates)",
        "3. Features & Segmentation",
        "4. Stitching & SIFT",
        "5. Object Tracking",
        "6. Stereo & Pose Estimation"
    ]
)

# --- HOME PAGE ---
if module_selection == "Home":
    st.title("Computer Vision Portfolio")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Student: Sreenija Kanugonda
        **Course:** CSC 8830  
        **Degree:** MS in Computer Science
        """)
    
    st.divider()

    # --- 1. GITHUB LINK ---
    st.subheader("📂 Source Code Repository")
    st.markdown("""
    All code for these assignments is managed in the GitHub repository below. 
    It includes the `app.py` orchestrator and all `modules`.
    """)
    
    # 🔴 TODO: REPLACE WITH YOUR ACTUAL GITHUB REPO LINK
    github_url = "https://github.com/sreenija1007/CV_project"
    
    st.link_button("View Source on GitHub 🌟", github_url)

    st.divider()

    # --- 2. VIDEO RECORDINGS (Google Drive) ---
    st.subheader("🎥 Module Demonstrations")
    st.info("Click the buttons below to watch the demo videos on Google Drive.")

    videos = {
        "Module 1: Single View Metrology": "https://drive.google.com/drive/folders/1Wd1MMo2ep4hdy-2rnzIP0_H1x1Zs3sQZ?usp=share_link",
        "Module 2: Object Detection, Blurring & Restoration": "https://drive.google.com/drive/folders/1aat8NI18LhPOHXVKinGco3dywxXM-jBH?usp=share_link",
        "Module 3: Features & Segmentation": "https://drive.google.com/drive/folders/17pTh-2SPXj-zn2gd8xfHo_1JCcqOCr0a?usp=share_link",
        "Module 4: Stitching & SIFT": "https://drive.google.com/drive/folders/1oV4fqWuvTSKlLdns7C1X-dXLYkjddyXX?usp=sharing",
        "Module 5: Object Tracking": "https://drive.google.com/drive/folders/1Qk4BclWMD96tXUYTkHLu_zmWS8Gou-31?usp=share_link",
        "Module 7: Stereo & Pose Estimation": "https://drive.google.com/file/d/1hw7WnKYiz8iZSdGbpCGFbeT2utuHTNhq/view?usp=share_link",
        "Evaluation Video": "https://drive.google.com/file/d/1hw7WnKYiz8iZSdGbpCGFbeT2utuHTNhq/view?usp=share_link"
    }

    # Iterate and display links
    for title, link in videos.items():
        with st.container():
            st.write(f"**{title}**")
            if "YOUR_LINK_HERE" in link:
                st.warning("⚠️ Link not yet added.")
            else:
                # Creates a clean button that opens the Drive link in a new tab
                st.link_button(f"▶️ Watch {title}", link)
            st.write("---")

# --- MODULE 1: METROLOGY ---
elif module_selection == "1. Single View Metrology":
    st.header("1. Single View Metrology")
    
    # 1. Load Calibration
    # Ensure calibration_data_mac.npz exists in the root or adjust path
    try:
        mtx, dist, f = measure_tool.get_calibration_data("calibration_data_mac.npz")
    except:
        f = None

    if f is None:
        st.error("⚠️ 'calibration_data_mac.npz' not found! Using default f=700.0")
        f = 700.0
        is_calibrated = False
    else:
        st.success(f"Calibration Loaded. Focal Length (f) = {f:.2f}")
        is_calibrated = True

    # 2. Inputs
    col1, col2 = st.columns(2)
    with col1:
        Z = st.number_input("Distance from Camera (Z) in cm:", min_value=1.0, value=50.0)
    with col2:
        st.info("Draw a line on the image to measure real dimensions.")

    input_source = st.radio("Select Image Source:", ("Upload Image", "Take Photo"), horizontal=True)
    
    img_file = None
    if input_source == "Upload Image":
        img_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
    else:
        img_file = st.camera_input("Take a picture of the object")

    # 3. Process Image
    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        if is_calibrated:
            img = measure_tool.undistort_image(img, mtx, dist)
            st.caption("Image Undistorted using Calibration Data")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 4. Interactive Canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=Image.fromarray(img_rgb),
            update_streamlit=True,
            height=img.shape[0],
            width=img.shape[1],
            drawing_mode="line",
            key="canvas_m1",
        )

        if canvas_result.json_data is not None:
            objects = pd.json_normalize(canvas_result.json_data["objects"]) 
            
            if not objects.empty:
                obj = objects.iloc[-1]
                x1, y1 = obj["left"], obj["top"]
                x2, y2 = x1 + obj["width"] * obj["scaleX"], y1 + obj["height"] * obj["scaleY"]
                
                real_size = measure_tool.calculate_real_distance((x1,y1), (x2,y2), Z, f)
                st.metric(label="Calculated Real Size", value=f"{real_size:.2f} cm")

# ==========================================
# MODULE 2: OBJECT DETECTION & RESTORATION
# ==========================================
elif module_selection == "2. Object Detection (Templates)":
    # --- TASK 3: DETECT, BLUR & RESTORE PIPELINE ---
    st.header("Task 2: Detection & Fourier Restoration Pipeline")
    
    # 1. Load Database
    # Ensure this path matches your folder structure
    templates = detection_blur.load_templates("Modules/Module2/templates") 
    
    if not templates:
        st.warning("No templates found! Please check your 'Modules/Module2/templates' folder.")
    
    uploaded_file = st.file_uploader("Upload Scene", type=['jpg', 'png', 'jpeg'])
    
    if uploaded_file and templates:
        # Convert uploaded file to OpenCV format
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        scene_color = cv2.imdecode(file_bytes, 1)
        
        # Display original Scene
        st.subheader("Original Scene")
        st.image(scene_color, channels="BGR", use_column_width=True)
        
        if st.button("Run Detection & Restoration Pipeline"):
            # CALL THE NEW PIPELINE FUNCTION
            det_img, blur_img, rest_img, status_text = detection_blur.detect_and_process_pipeline(scene_color, templates)
            
            # Display Status
            if "Detected" in status_text:
                st.success(status_text)
            else:
                st.warning(status_text)
            
            # Display Results in 3 Columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.image(det_img, caption="1. Object Detected", channels="BGR", use_column_width=True)
            
            with col2:
                st.image(blur_img, caption="2. Blurred (Fourier)", channels="BGR", use_column_width=True)
                
            with col3:
                st.image(rest_img, caption="3. Restored (Inverse FFT)", channels="BGR", use_column_width=True)

# --- MODULE 3: FEATURES ---
elif module_selection == "3. Features & Segmentation":
    st.header("3. Features & Segmentation")
    
    # --- UPDATED SELECTION MENU ---
    task_mode = st.radio("Select Task:", [
        "Task 1: Gradient & LoG", 
        "Task 2: Keypoints (Edges/Corners)", 
        "Task 3: Object Boundary",
        "Task 4: ArUco Boundary (Convex Hull)",
        "Task 5: SAM2 Segmentation"             
    ], horizontal=True)
    
    st.divider()
    input_source = st.radio("Select Image Source:", ("Upload Image", "Take Photo"), horizontal=True)
    
    img_file = None
    if input_source == "Upload Image":
        img_file = st.file_uploader("Upload Image for Analysis", type=['jpg', 'png', 'jpeg'])
    else:
        img_file = st.camera_input("Take a picture for Analysis")
    
    if img_file is not None:
        file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        st.subheader("Original Image")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width=400)
        st.divider() 
        
        # --- ORIGINAL TASKS ---
        if task_mode == "Task 1: Gradient & LoG":
            st.subheader("Gradient Analysis")
            mag, angle, log = features.get_gradient_and_log(img)
            col1, col2, col3 = st.columns(3)
            with col1: st.image(mag, caption="Gradient Magnitude")
            with col2: st.image(angle, caption="Gradient Angle")
            with col3: st.image(log, caption="Laplacian of Gaussian")

        elif task_mode == "Task 2: Keypoints (Edges/Corners)":
            st.subheader("Keypoint Detection")
            canny, corners = features.get_keypoints(img)
            col1, col2 = st.columns(2)
            with col1: st.image(canny, caption="Canny Edges")
            with col2: st.image(cv2.cvtColor(corners, cv2.COLOR_BGR2RGB), caption="Harris Corners")

        elif task_mode == "Task 3: Object Boundary":
            st.subheader("Boundary Detection")
            mask, boundary = features.get_boundary(img)
            col1, col2 = st.columns(2)
            with col1: st.image(mask, caption="Threshold Mask")
            with col2: st.image(cv2.cvtColor(boundary, cv2.COLOR_BGR2RGB), caption="Object Boundary")

        # --- NEW TASKS 4 & 5 ---
        elif task_mode == "Task 4: ArUco Boundary (Convex Hull)":
            st.subheader("ArUco Convex Hull")
            res_img, error = cv_processors.process_aruco_boundary(img)
            
            if error:
                st.error(error)
            else:
                st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="ArUco Hull (Blue Line)")

        elif task_mode == "Task 5: SAM2 Segmentation":
            st.subheader("Comparison: Geometric Hull vs. AI Segmentation")
            st.info("Visualizing the difference between simple geometry (Task 4) and deep learning (Task 5).")
            
            # Load Model Only When Needed
            with st.spinner("Loading SAM2 Model..."):
                predictor = load_sam2_model()
            
            if predictor:
                # --- Run Both Tasks ---
                
                # 1. Run Task 4 (Geometric Hull)
                res_hull, err_hull = cv_processors.process_aruco_boundary(img)
                
                # 2. Run Task 5 (SAM2 AI)
                res_sam, err_sam = cv_processors.process_sam2_segmentation(img, predictor)
                
                # --- Display Side-by-Side ---
                col1, col2 = st.columns(2)
                
                with col1:
                    if err_hull:
                        st.error(f"Task 4 Error: {err_hull}")
                    else:
                        st.image(cv2.cvtColor(res_hull, cv2.COLOR_BGR2RGB), caption="Task 4: ArUco Convex Hull (Geometric)")
                        
                with col2:
                    if err_sam:
                        st.error(f"Task 5 Error: {err_sam}")
                    else:
                        st.image(cv2.cvtColor(res_sam, cv2.COLOR_BGR2RGB), caption="Task 5: SAM2 Segmentation (AI)")
                
                # --- Final Summary ---
                if not err_hull and not err_sam:
                    st.success("Comparison Complete! Notice how the Blue Hull bridges gaps (concave areas), while the Red SAM2 Mask follows the exact curve.")

            else:
                st.error("Failed to load SAM2 Model. Check your installation.")

# --- MODULE 4: STITCHING & SIFT ---
elif module_selection == "4. Stitching & SIFT":
    st.header("4. Image Stitching & SIFT")
    
    tab1, tab2 = st.tabs(["Task 1: Panorama Stitching", "Task 2: SIFT & RANSAC Comparison"])
    
    # --- TAB 1: STITCHING ---
    with tab1:
        st.subheader("Task 1: Panorama Stitching")
        st.write("Upload your sequence of images (Landscape 4x or Portrait 8x).")
        st.info("⚠️ Important: Please name your files `1.jpg`, `2.jpg`, etc., so they are sorted correctly.")

        uploaded_files = st.file_uploader("Upload Images", type=['jpg','png','jpeg'], accept_multiple_files=True, key="pano_task1")
        
        if uploaded_files:
            # Sort files by name to ensure correct stitching order (1, 2, 3...)
            uploaded_files.sort(key=lambda x: x.name)
            
            # Display thumbnails of input
            st.write("Input Sequence:")
            cols = st.columns(len(uploaded_files))
            for idx, file in enumerate(uploaded_files):
                cols[idx].image(file, caption=file.name, use_column_width=True)

            if st.button("Stitch Images", key="btn_stitch_t1"):
                with st.spinner("Stitching images..."):
                    
                    # Convert to OpenCV format
                    cv_images = []
                    for f in uploaded_files:
                        file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
                        img = cv2.imdecode(file_bytes, 1)
                        # Reset file pointer for re-reading if needed
                        f.seek(0)
                        cv_images.append(img)
                    
                    if len(cv_images) < 2:
                        st.error("You need at least 2 images to create a panorama.")
                    else:
                        # CALL THE FUNCTION FROM PANORAMA.PY
                        try:
                            final_pano, msg = panorama.stitch_vertical_sequence(cv_images)
                            
                            if "failed" in msg:
                                st.warning(msg)
                                st.image(final_pano, channels="BGR", caption="Partial Result")
                            else:
                                st.success(msg)
                                st.image(final_pano, channels="BGR", caption="Final Stitched Panorama")
                        except NameError:
                             st.error("Module 'panorama' not found. Please ensure 'modules/Module4/task1/panorama.py' exists.")
    # --- TAB 2: SIFT & RANSAC ---
    with tab2:
        st.subheader("Task 2: SIFT from Scratch & RANSAC Comparison")
        st.markdown("""
        **Instructions:**
        1. Upload 2 overlapping images.
        2. **Order Matters:** The "Source" image will be warped to fit the "Dest" image.
        """)
        
        st.warning("⚠️ **Upload Order:** For correct alignment, **Source** should be the Right/Bottom image, and **Dest** should be the Left/Top image.")

        col1, col2 = st.columns(2)
        with col1:
            img_file1 = st.file_uploader("1. Upload Source (Right/Bottom)", type=['jpg', 'png', 'jpeg'], key="sift_src")
        with col2:
            img_file2 = st.file_uploader("2. Upload Dest (Left/Top)", type=['jpg', 'png', 'jpeg'], key="sift_dst")

        if img_file1 and img_file2:
            # Convert uploaded files to OpenCV format
            file_bytes1 = np.asarray(bytearray(img_file1.read()), dtype=np.uint8)
            file_bytes2 = np.asarray(bytearray(img_file2.read()), dtype=np.uint8)
            img1 = cv2.imdecode(file_bytes1, 1)
            img2 = cv2.imdecode(file_bytes2, 1)
            
            # Convert to Grayscale
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

            if st.button("Run SIFT & RANSAC Pipeline"):
                with st.spinner("Running SIFT from scratch... (This may take a moment)"):
                    try:
                        # 1. Run CUSTOM SIFT (from sift_scratch.py)
                        # Expecting list of dicts for keypoints, and numpy array for descriptors
                        kp1_custom, des1_custom = sift_scratch.run_from_scratch_sift(gray1)
                        kp2_custom, des2_custom = sift_scratch.run_from_scratch_sift(gray2)
                        
                        st.write(f"Custom SIFT: Found {len(kp1_custom)} keypoints in Source, {len(kp2_custom)} in Dest.")

                        # 2. Run OPENCV SIFT (for comparison)
                        sift_cv = cv2.SIFT_create()
                        kp1_cv, _ = sift_cv.detectAndCompute(gray1, None)

                        # --- DISPLAY PART 1: SIFT COMPARISON ---
                        st.divider()
                        st.markdown("### 1. Feature Detection Comparison")
                        
                        # Draw Custom Keypoints (Green)
                        img1_custom_viz = img1.copy()
                        for kp in kp1_custom:
                            # kp is a dictionary {'x': val, 'y': val, ...}
                            cv2.circle(img1_custom_viz, (int(kp['x']), int(kp['y'])), 3, (0, 255, 0), 1)
                        
                        # Draw OpenCV Keypoints (Red)
                        img1_cv_viz = img1.copy()
                        # drawKeypoints draws colorful circles by default
                        img1_cv_viz = cv2.drawKeypoints(gray1, kp1_cv, img1_cv_viz, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

                        # Show side-by-side
                        comp_col1, comp_col2 = st.columns(2)
                        with comp_col1:
                            st.image(img1_custom_viz, channels="BGR", caption=f"My From-Scratch SIFT ({len(kp1_custom)} KPs)")
                        with comp_col2:
                            st.image(img1_cv_viz, channels="BGR", caption=f"OpenCV SIFT ({len(kp1_cv)} KPs)")

                        # --- DISPLAY PART 2: RANSAC STITCHING ---
                        st.divider()
                        st.markdown("### 2. RANSAC Optimization Result")
                        
                        # Match Features
                        matches = sift_scratch.match_features(des1_custom, des2_custom)
                        st.write(f"Found {len(matches)} putative matches using custom matcher.")
                        
                        if len(matches) < 4:
                             st.error("Not enough matches found to compute Homography.")
                        else:
                            # Run RANSAC
                            H, inliers = sift_scratch.from_scratch_ransac(kp1_custom, kp2_custom, matches)
                            
                            if H is not None:
                                st.success(f"RANSAC Converged! Found {np.sum(inliers)} inliers.")
                                
                                # Stitching Logic using the found H
                                h1, w1 = img1.shape[:2]
                                h2, w2 = img2.shape[:2]
                                
                                # Create canvas (Ample space to avoid cropping)
                                canvas_w = w1 + w2
                                canvas_h = max(h1, h2) * 2 
                                
                                # Warp Image 1 (Source) using H to align with Image 2 (Dest)
                                stitched = cv2.warpPerspective(img1, H, (canvas_w, canvas_h))
                                
                                # Overlay Image 2 (Dest) at 0,0
                                stitched[0:h2, 0:w2] = img2
                                
                                # Simple Crop
                                gray_st = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
                                _, thresh = cv2.threshold(gray_st, 1, 255, cv2.THRESH_BINARY)
                                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                if contours:
                                    c = max(contours, key=cv2.contourArea)
                                    x,y,w,h = cv2.boundingRect(c)
                                    stitched = stitched[y:y+h, x:x+w]
                                
                                st.image(stitched, channels="BGR", caption="Final Stitched Image using From-Scratch RANSAC")
                            else:
                                st.error("RANSAC failed to find a consistent homography. Try images with more overlap.")
                    except NameError:
                        st.error("Module 'sift_scratch' not found. Please ensure 'modules/Module4/task2/sift_scratch.py' exists.")
                    except Exception as e:
                         st.error(f"An error occurred: {e}")


# --- MODULE 5: MOTION TRACKING ---
elif module_selection == "5. Motion Tracking":
    st.header("5. Motion Tracking")
    
    track_mode = st.radio("Select Tracking Method:", 
                          ["1. Marker-based (ArUco)", 
                           "2. Marker-less (CSRT)", 
                           "3. SAM2 Segmentation Demo"])

    # --- MODE 1: MARKER BASED ---
    if track_mode == "1. Marker-based (ArUco)":
        st.subheader("Marker-based Tracking")
        st.info("Show an ArUco 4x4 Marker to the camera.")
        
        if st.checkbox("Start Webcam (Marker)"):
            cap = cv2.VideoCapture(0)
            st_frame = st.image([])
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                output = tracker.track_markers_aruco(frame)
                st_frame.image(output, channels="BGR")
            cap.release()

    # --- MODE 2: MARKER-LESS (CSRT) ---
    elif track_mode == "2. Marker-less (CSRT)":
        st.subheader("Marker-less Object Tracking")
        st.info("1. Take photo. 2. Draw box. 3. START. (Keep object still between photo and start!)")

        # Initialize Session State
        if 'track_init_frame' not in st.session_state: st.session_state['track_init_frame'] = None
        if 'track_bbox' not in st.session_state: st.session_state['track_bbox'] = None

        col_controls, col_display = st.columns([1, 2])
        
        with col_controls:
            # 1. Take Photo
            img_buffer = st.camera_input("Take a photo to define object")
            
            if img_buffer is not None:
                bytes_data = img_buffer.getvalue()
                cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                # Resize to standard webcam resolution (640x480)
                # This ensures coordinates from photo match the video stream later
                cv2_img = cv2.resize(cv2_img, (640, 480))
                
                # CRITICAL FIX 2: Convert to RGB for consistency
                st.session_state['track_init_frame'] = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)

        # 2. Draw Box
        if st.session_state['track_init_frame'] is not None:
            st.markdown("### Draw a box around the object:")
            
            # Display Canvas
            # We enforce fixed 640x480 dimensions to match the resize above
            canvas_result = st_canvas(
                fill_color="rgba(255, 165, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=Image.fromarray(st.session_state['track_init_frame']),
                height=480,
                width=640,
                drawing_mode="rect",
                key="tracker_canvas"
            )

            # Check for drawing
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if len(objects) > 0:
                    obj = objects[-1]
                    # No scaling needed because canvas = image size = 640x480
                    x = int(obj["left"])
                    y = int(obj["top"])
                    w = int(obj["width"])
                    h = int(obj["height"])
                    
                    if w > 10 and h > 10:
                        st.session_state['track_bbox'] = (x, y, w, h)
            
            # Show Status
            if st.session_state['track_bbox']:
                st.success(f"Target Locked: {st.session_state['track_bbox']}")
                
                # 3. Start Tracking
                if st.button("Start Tracking"):
                    st.warning("Starting... Don't move the object yet!")
                    
                    cap = cv2.VideoCapture(0)
                    
                    # Force Webcam to 640x480 to match our photo
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    # Warmup
                    for _ in range(10): cap.read()
                    
                    ret, frame = cap.read()
                    if ret:
                        # Ensure Frame is RGB to match the Photo
                        # OpenCV gives BGR, we convert to RGB before initializing tracker
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        
                        bbox = st.session_state['track_bbox']
                        tracker_obj = tracker.MarkerlessTracker(frame_rgb, bbox)
                        
                        st_track_window = st.image([])
                        stop_btn = st.button("Stop Tracking")
                        
                        while cap.isOpened() and not stop_btn:
                            ret, frame = cap.read()
                            if not ret: break
                            
                            # Convert to RGB for tracking update
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            
                            # Update Tracker
                            success, new_box = tracker_obj.update(frame_rgb)
                            
                            if success:
                                x, y, w, h = [int(v) for v in new_box]
                                cv2.rectangle(frame_rgb, (x, y), (x + w, y + h), (0, 255, 0), 3)
                                cv2.putText(frame_rgb, "Tracking", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                            else:
                                cv2.putText(frame_rgb, "Lost", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)
                            
                            # Display (It is already RGB)
                            st_track_window.image(frame_rgb)
                            
                        cap.release()
            else:
                st.warning("Draw a box first!")

    # --- MODE 3: SAM2 DEMO ---
    elif track_mode == "3. SAM2 Segmentation Demo":
        st.subheader("SAM2 Segmentation Overlay")
        v_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
        n_file = st.file_uploader("Upload NPZ Mask File", type=['npz'])
        
        if v_file and n_file and st.button("Run SAM2 Demo"):
            with open("temp_video.mp4", "wb") as f: f.write(v_file.read())
            mask_data = np.load(n_file)
            cap = cv2.VideoCapture("temp_video.mp4")
            st_sam_window = st.image([])
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                output = tracker.overlay_sam2_mask(frame, frame_count, mask_data)
                st_sam_window.image(output, channels="BGR")
                frame_count += 1
                cv2.waitKey(30) 
            cap.release()

# --- MODULE 6: STEREO & POSE ESTIMATION ---
elif module_selection == "6. Stereo & Pose Estimation":
    st.header("6. Stereo Vision & Pose Estimation")
    mode_7 = st.radio("Select Task:", ["Task 1: Stereo Size Estimation", "Task 2: Pose & Hand Tracking"], horizontal=True)
    
    if mode_7 == "Task 1: Stereo Size Estimation":
        st.subheader("Stereo Size Estimation")
        try:
            f = stereo_pose.load_calibration("calibration_data_mac.npz")
        except:
            f = 700.0 # Default if file not found
            
        baseline = st.number_input("Baseline (Distance between cameras) in cm:", value=30.0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 1. Left View")
            src_L = st.radio("Input Source:", ("Upload", "Camera"), key="src_L", horizontal=True)
            file_L = st.file_uploader("Upload Left", key="fL") if src_L == "Upload" else st.camera_input("Cam Left", key="cL")
            points_L = []
            if file_L:
                file_L.seek(0)
                img_L = cv2.imdecode(np.asarray(bytearray(file_L.read()), dtype=np.uint8), 1)
                img_L_rgb = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)
                orig_h, orig_w = img_L.shape[:2]
                canvas_h = 350
                canvas_w = int(orig_w * (canvas_h / orig_h))
                scale_factor_L = orig_h / canvas_h 
                st.write("**Click 3 Points:**")
                canvas_L = st_canvas(fill_color="rgba(0, 255, 0, 0.3)", stroke_color="#00FF00", background_image=Image.fromarray(img_L_rgb), height=canvas_h, width=canvas_w, drawing_mode="point", key="canvas_L")
                if canvas_L.json_data:
                    for obj in canvas_L.json_data["objects"]:
                        points_L.append((obj["left"] * scale_factor_L, obj["top"] * scale_factor_L))
        
        with col2:
            st.markdown("### 2. Right View")
            src_R = st.radio("Input Source:", ("Upload", "Camera"), key="src_R", horizontal=True)
            file_R = st.file_uploader("Upload Right", key="fR") if src_R == "Upload" else st.camera_input("Cam Right", key="cR")
            points_R = []
            if file_R:
                file_R.seek(0)
                img_R = cv2.imdecode(np.asarray(bytearray(file_R.read()), dtype=np.uint8), 1)
                img_R_rgb = cv2.cvtColor(img_R, cv2.COLOR_BGR2RGB)
                orig_h_R, orig_w_R = img_R.shape[:2]
                canvas_h = 350
                canvas_w_R = int(orig_w_R * (canvas_h / orig_h_R))
                scale_factor_R = orig_h_R / canvas_h
                st.write("**Click SAME 3 Points:**")
                canvas_R = st_canvas(fill_color="rgba(255, 0, 0, 0.3)", stroke_color="#FF0000", background_image=Image.fromarray(img_R_rgb), height=canvas_h, width=canvas_w_R, drawing_mode="point", key="canvas_R")
                if canvas_R.json_data:
                    for obj in canvas_R.json_data["objects"]:
                        points_R.append((obj["left"] * scale_factor_R, obj["top"] * scale_factor_R))

        if st.button("Calculate Real Dimensions", type="primary"):
            if len(points_L) == 3 and len(points_R) == 3:
                results, msg = stereo_pose.calculate_stereo_metrics(points_L, points_R, baseline, f)
                if results:
                    st.success("Calculation Successful!")
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Avg Depth (Z)", f"{results['z_avg']} cm")
                    m2.metric("Real Width", f"{results['real_w']} cm")
                    m3.metric("Real Height", f"{results['real_h']} cm")
                else:
                    st.error(f"Calculation Failed: {msg}")
            else:
                st.warning(f"Selection Incomplete. Left: {len(points_L)}/3. Right: {len(points_R)}/3.")

    elif mode_7 == "Task 2: Pose & Hand Tracking":
        st.subheader("Real-Time Tracking (Pose & Hands)")
        track_type = st.radio("Tracking Target:", ["Body Pose", "Hand Tracking"], horizontal=True)
        if 'pose_data_log' not in st.session_state: st.session_state['pose_data_log'] = []
        col_act, col_vid = st.columns([1, 3])
        with col_act:
            run_cam = st.checkbox("Start Webcam")
            if st.button("Clear Data Log"):
                st.session_state['pose_data_log'] = []
        with col_vid: FRAME_WINDOW = st.image([])
        if run_cam:
            cap = cv2.VideoCapture(0)
            if track_type == "Body Pose": mp_model = mp.solutions.pose.Pose()
            else: mp_model = mp.solutions.hands.Hands()
            while run_cam:
                ret, frame = cap.read()
                if not ret: break
                if track_type == "Body Pose": annotated_frame, data = stereo_pose.process_pose(frame, mp_model)
                else: annotated_frame, data = stereo_pose.process_hands(frame, mp_model)
                if data: st.session_state['pose_data_log'].extend(data)
                FRAME_WINDOW.image(annotated_frame, channels="BGR")
            cap.release()
            mp_model.close()
        if st.session_state['pose_data_log']:
            df = pd.DataFrame(st.session_state['pose_data_log'])
            st.download_button("Download CSV", df.to_csv(index=False).encode('utf-8'), "tracking_data.csv", "text/csv")

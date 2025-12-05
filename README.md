# Computer Vision Portfolio Dashboard 

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
![Python 3.10](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)

A unified web application showcasing various Computer Vision algorithms and assignments. Built with **Streamlit**, this dashboard acts as an orchestrator for multiple CV modules, ranging from classical image processing (SIFT, Fourier) to modern Deep Learning (SAM2, MediaPipe).

## 🚀 Features & Modules

The application is divided into 6 core modules:

### 1. Single View Metrology
* **Goal:** Measure real-world dimensions of objects from a single 2D image.
* **Tech:** Camera Calibration, Homography, Interactive Canvas.
* **Function:** Users draw a line on an object, and based on the camera distance (Z) and focal length, the app calculates the real-world size in centimeters.

### 2. Object Detection & Restoration
* **Task:** Template Matching & Frequency Domain Filtering.
* **Tech:** Fourier Transform, Normalized Cross-Correlation.
* **Function:** Detects objects in a cluttered scene using templates and performs image restoration using Fourier techniques.

### 3. Features & Segmentation
* **Task:** Feature Extraction & AI Segmentation.
* **Tech:** Canny Edges, Harris Corners, Gradient/LoG, Convex Hull, **SAM2 (Segment Anything Model 2)**.
* **Comparison:** Visualizes the difference between geometric boundary detection (ArUco/Hull) and deep learning segmentation (SAM2).

### 4. Image Stitching & SIFT
* **Task:** Panorama creation and Feature Matching.
* **Tech:** SIFT (implemented from scratch), RANSAC, Homography, Warping.
* **Function:** Stitches multiple overlapping images into a seamless panorama and visualizes feature matching between two images.

### 5. Motion Tracking
* **Task:** Real-time object tracking in video.
* **Tech:** ArUco Markers, CSRT (Markerless Tracker), Ultralytics SAM2.
* **Cloud Note:** Uses video file uploads instead of live webcam to ensure compatibility with cloud servers.

### 6. Stereo Vision & Pose Estimation
* **Task:** 3D Depth estimation and Human Pose Tracking.
* **Tech:** Stereo Disparity, **MediaPipe** (Hands & Body).
* **Function:** Estimates depth using left/right stereo images and tracks human skeletal landmarks in real-time.

---

## 🛠️ Installation (Local)

To run this app on your local machine:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Create a Virtual Environment (Optional but recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the App:**
    ```bash
    streamlit run app.py
    ```

---

## ☁️ Deployment Notes (Streamlit Cloud)

This application is optimized for **Streamlit Community Cloud**. If you are deploying your own version, please note the following configurations:

### 1. Python Version
* **Requirement:** Python **3.9** or **3.10**.
* **Reason:** Libraries like MediaPipe are not yet fully compatible with Python 3.12+.

### 2. System Dependencies (`packages.txt`)
* The repository includes a `packages.txt` file containing `libgl1-mesa-glx`.
* **Reason:** Required to install OpenGL drivers for OpenCV on the headless Linux server.

### 3. Webcam Limitations
* Direct webcam access (`cv2.VideoCapture(0)`) is disabled in the cloud version

### 4. SAM2 Implementation
* **Reason:** The official Meta SAM2 implementation requires GPU compilation (CUDA), which causes build failures on CPU-only cloud environments.

---

## 📂 Project Structure

```text
CV_Portfolio/
├── app.py                 # Main application entry point
├── requirements.txt       # Python dependencies
├── packages.txt           # System-level dependencies (Linux)
├── calibration_data_mac.npz # Camera calibration file
└── Modules/               # Source code for assignments
    ├── Module1/
    ├── Module2/
    ├── Module3/
    ├── Module4/
    ├── Module5_6/
    └── Module7/

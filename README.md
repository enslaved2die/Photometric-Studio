# 🎨 PBR Studio Pro

**PBR Studio Pro** is a high-performance, browser-based tool designed to synthesize professional-grade PBR (Physically Based Rendering) textures from a set of photographs. By utilizing advanced **Photometric Stereo** algorithms, it extracts surface geometry, color data, and micro-surface details with high precision.

## 💡 Project Origin & Authorship
This project is a collaboration between a domain expert and an AI.
* **Concept & Domain Expertise:** The fundamental knowledge of Photometric Stereo, PBR workflows, and algorithmic requirements (such as Flat-Field correction and Slope-Based AO) were provided by the user.
* **Development:** The programming, software architecture, and UI implementation were handled by **Gemini (Google's AI)** based on the user's technical direction and feedback.

> **Disclaimer:** This tool was developed with the assistance of an AI. While the logic follows established computer vision fundamentals, users should validate results for mission-critical professional bakes.

## ✨ Key Features

* **Photometric Stereo Solver:** Extract high-fidelity Normal and Albedo maps from multiple lighting angles.
* **ACES Filmic Tone Mapping:** Prevents highlight clipping and "burn-out" in the 3D viewer, mimicking real cinema film roll-off.
* **Interactive 3D Viewport:** Evaluate results in real-time with an integrated Three.js renderer. Toggle Albedo, Normals, AO, and Roughness independently.
* **Advanced Roughness Correction:** * **High-Pass:** Isolates micro-detail by stripping lighting gradients.
    * **Corrective Flat-Field:** Uses non-linear division to eliminate point-light hotspots and vignettes.
* **Slope-Base (Geometric) AO:** Ambient Occlusion derived from surface curvature (Normal Map gradients) to ensure shadows represent physical depth.

## 🚀 Quick Start

### Prerequisites
* Python 3.9+
* `pip install streamlit numpy opencv-python-headless scipy`

### Installation & Run
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/photometric_studio.git
    cd pbr-studio-pro
    ```
2.  **Run the application:**
    ```bash
    streamlit run photometric_studio.py
    ```

## 🧪 Open Source Credits & Tech Stack

This project is built using:
* **Streamlit** (UI Framework)
* **OpenCV** (Image Processing)
* **NumPy** (Linear Algebra Solver)
* **Three.js** (WebGL Rendering)
* **ACES** (Filmic Tone Mapping logic)

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.
# AMD Sling Shot: Automated Creator Feedback (ACF)

Empowering creators with private, low-latency visual intelligence.

## Project Overview
ACF is a diagnostic engine designed for YouTube creators and social media managers to validate visual hierarchy before publishing. By leveraging AI-driven saliency mapping and design heuristics, the tool provides an instant Engagement Score to predict viewer focus.

## The Problem
Creators often guess which thumbnail or asset will perform better. Existing cloud-based eye-tracking tools are:

- Slow: Requiring long upload and processing times.
- Privacy-Risk: Proprietary, unreleased designs must be uploaded to third-party servers.
- Disconnected: They provide raw data without designer-first actionable insights.

## The AMD Advantage (Key for Judges)
This project is built to run natively on AMD Ryzen AI NPUs and Radeon GPUs.

- Privacy by Design: By using onnxruntime-directml, the model runs entirely on the user's local machine. No creative assets ever touch the cloud.
- Hardware-Software Co-Design: We utilize DirectML to tap into the high-performance execution providers of AMD hardware, ensuring sub-100ms inference times for real-time design iteration.

## Core Features
- Visual Saliency Mapping: Uses Spectral Residual algorithms to predict eye-tracking focus points.
- Rule of Thirds Analysis: Automatically checks if the design's focal points align with Power Point intersections.
- Face Detection Bonus: Integrates OpenCV Haar Cascades to score the impact of human presence in thumbnails.
- Accessibility Suite: Includes a Squint Test (Grayscale) and High Contrast filters to ensure legibility across all viewer types.
- A/B Battle Mode: Side-by-side comparison of two designs with localized heatmaps.

## Tech Stack
- Language: Python 3.10+
- Inference Engine: ONNX Runtime with DirectML (AMD/NVIDIA Acceleration)
- Computer Vision: OpenCV (Saliency, Canny Edge Detection, Face Detection)
- Frontend: Streamlit
- Numerical Processing: NumPy

## Installation and Local Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/creator-feedback.git

# Create and activate environment
python -m venv venv
# Windows
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install onnxruntime-directml streamlit opencv-contrib-python numpy
```

## Run
```bash
streamlit run app.py
```

## Future Roadmap
- Video Saliency: Real-time heatmap overlays for short-form video content (Reels/Shorts).
- NPU Optimization: Further quantization of models specifically for the XDNA architecture in Ryzen AI.

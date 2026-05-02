# Crop Disease Detection Using Edge Vision Systems 🌿

A complete edge AI application built to detect crop diseases 100% on-device with zero cloud dependency.

## Features
- **On-Device Inference**: Powered by a highly optimized TensorFlow Lite (TFLite) Google CropNet model.
- **Ultra-Lightweight**: Model size is only 2.76 MB (strict < 50MB constraint met).
- **High Speed**: Designed to meet < 100ms latency targets on standard hardware.
- **Power Efficiency Profiling**: Custom metrics engine using `psutil` to track system resources during inference.
- **Premium Web UI**: Built with Flask, featuring a beautiful glassmorphism design for demonstration and testing.

## How to Run Locally

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python app.py
   ```

3. **Test**
   Open your browser and navigate to `http://localhost:5000`. Upload an image of a leaf to see the edge inference in action!

## Tech Stack
- **AI/ML**: TensorFlow, TFLite (8-bit Quantized CropNet)
- **Backend**: Python, Flask
- **Frontend**: HTML5, Vanilla CSS, JavaScript
- **Profiling**: Psutil, high-resolution timers

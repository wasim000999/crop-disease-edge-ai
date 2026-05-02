import os
import time
import psutil
import numpy as np
import tensorflow as tf
from PIL import Image

MODEL_PATH = "model/crop_disease_model.tflite"
LABELS_PATH = "model/labels.txt"

# Load interpreter once to simulate a persistent edge service
interpreter = None
input_details = None
output_details = None
labels = []

def initialize():
    global interpreter, input_details, output_details, labels
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("TFLite model not found. Run generate_model.py first.")
    
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r") as f:
            labels = [line.strip() for line in f.readlines()]
    else:
        labels = ["Class 0", "Class 1", "Class 2", "Class 3"]

def compute_power_efficiency(latency_ms, cpu_usage):
    # A dummy formula for "Power Efficiency Score" (0-100)
    # Lower latency is better, lower CPU usage is better.
    # Score = 100 - (latency_ms/2) - (cpu_usage)
    # Normalizing...
    score = 100.0 - (min(latency_ms, 100) * 0.4) - (cpu_usage * 0.5)
    return max(0.0, min(100.0, score))

def predict(image_path):
    if interpreter is None:
        initialize()

    # Preprocess Image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    
    # Normalize or cast based on model requirements
    if input_details[0]['dtype'] == np.uint8:
        # For quantized model (like CropNet) expecting uint8 [0, 255]
        input_data = np.array(img, dtype=np.uint8)
    else:
        # For standard MobileNetV2 float32 [-1, 1]
        input_data = np.array(img, dtype=np.float32)
        input_data = (input_data / 127.5) - 1.0
        
    input_data = np.expand_dims(input_data, axis=0)

    # Get system state before inference
    cpu_before = psutil.cpu_percent(interval=None)
    
    # Inference Tracking
    start_time = time.perf_counter()
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    end_time = time.perf_counter()
    
    # Get system state after inference
    cpu_after = psutil.cpu_percent(interval=None)
    cpu_usage = max(cpu_after, cpu_before)
    
    latency_ms = (end_time - start_time) * 1000.0
    
    # Calculate Metrics
    power_score = compute_power_efficiency(latency_ms, cpu_usage)
    model_size_mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    
    predicted_idx = np.argmax(output_data[0])
    if output_details[0]['dtype'] == np.uint8:
        confidence = float(output_data[0][predicted_idx]) / 255.0
    else:
        confidence = float(output_data[0][predicted_idx])
    predicted_label = labels[predicted_idx] if predicted_idx < len(labels) else f"Class {predicted_idx}"

    return {
        "prediction": predicted_label,
        "confidence": round(confidence * 100, 2),
        "latency_ms": round(latency_ms, 2),
        "power_score": round(power_score, 1),
        "model_size_mb": round(model_size_mb, 2),
        "cpu_usage": round(cpu_usage, 1)
    }

if __name__ == "__main__":
    # Test inference if run directly
    print("Testing initialization...")
    initialize()
    print("Ready.")

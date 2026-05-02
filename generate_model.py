import tensorflow as tf
import numpy as np
import os

MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

CLASSES = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold"]

def create_model():
    print("Building a lightweight MobileNetV2 model for edge inference...")
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def representative_data_gen():
    # Provide a few dummy random samples for quantization
    for _ in range(100):
        yield [np.random.uniform(-1, 1, size=(1, 224, 224, 3)).astype(np.float32)]

def export_tflite(model):
    print("Exporting model to TFLite with Post-Training Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Apply optimizations
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Use representative dataset to ensure full integer quantization
    converter.representative_dataset = representative_data_gen
    
    # Ensure fully INT8 model (useful for Edge TPUs/microcontrollers, though FP16 is also fine)
    # We will just use default optimization which makes weights 8-bit, keeping IO float32 for simplicity.
    
    tflite_model = converter.convert()
    
    model_path = os.path.join(MODEL_DIR, "crop_disease_model.tflite")
    with open(model_path, "wb") as f:
        f.write(tflite_model)
        
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model successfully saved to {model_path}")
    print(f"Model Size: {size_mb:.2f} MB")
    
    with open(os.path.join(MODEL_DIR, "labels.txt"), "w") as f:
        f.write("\n".join(CLASSES))
    print("Labels saved.")

if __name__ == "__main__":
    model = create_model()
    export_tflite(model)
    print("Edge model generation complete!")

import urllib.request
import os
import tensorflow as tf

MODEL_DIR = "model"
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

model_path = os.path.join(MODEL_DIR, "crop_disease_model.tflite")
url = "https://tfhub.dev/google/lite-model/cropnet/classifier/cassava_disease_V1/1?lite-format=tflite"

print("Downloading real pre-trained CropNet model from Google...")
urllib.request.urlretrieve(url, model_path)

print("Downloaded successfully.")

# Write the actual labels for this model
labels = [
    "Cassava Bacterial Blight (CBB)",
    "Cassava Brown Streak Disease (CBSD)",
    "Cassava Green Mite (CGM)",
    "Cassava Mosaic Disease (CMD)",
    "Healthy",
    "Unknown"
]
with open(os.path.join(MODEL_DIR, "labels.txt"), "w") as f:
    f.write("\n".join(labels))
print("Labels updated.")

# Inspect model input
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
print("Model Input Shape:", input_details[0]['shape'])
print("Model Input Type:", input_details[0]['dtype'])

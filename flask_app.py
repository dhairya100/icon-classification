import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps
import io

# --- Configuration ---
# Set the path to your model file
MODEL_PATH = 'keras_model.h5' 
# Set the path to your labels file
LABELS_PATH = 'labels.txt'
# Teachable Machine models expect a 224x224 input size
IMG_HEIGHT = 224
IMG_WIDTH = 224

# --- Load the model and labels once when the server starts ---
model = None
CLASS_NAMES = []
try:
    # Load the model
    model = load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully.")
    
    # Load the labels
    with open(LABELS_PATH, 'r') as f:
        CLASS_NAMES = [line.strip() for line in f.readlines()]
    print(f"Labels loaded successfully. Found {len(CLASS_NAMES)} classes.")
    
except Exception as e:
    print(f"Error loading model or labels: {e}")
    
app = Flask(__name__)

@app.route('/')
def home():
    return "The Teachable Machine logo classification server is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    try:
        # Read the image data from the file
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))

        # Teachable Machine preprocessing steps
        # Resize and crop the image to the model's input size
        size = (IMG_WIDTH, IMG_HEIGHT)
        img = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
        
        # Turn the image into a numpy array
        image_array = np.asarray(img)
        
        # Normalize the image data (Teachable Machine uses -1 to 1 normalization)
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
        
        # Create the data array to feed into the model
        data = np.ndarray(shape=(1, IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
        data[0] = normalized_image_array

        # Make a prediction
        predictions = model.predict(data)
        
        # Find the highest confidence score and its corresponding class index
        score = np.max(predictions)
        predicted_class_index = np.argmax(predictions)
        
        # Get the predicted class name
        # We'll use a confidence threshold, similar to the previous example
        if score > 0.7: 
            predicted_class_name = CLASS_NAMES[predicted_class_index]
        else:
            predicted_class_name = 'Unknown'

        response = {
            'prediction': predicted_class_name,
            'confidence': float(score)
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': f"Prediction failed: {e}"}), 500

if __name__ == '__main__':
    # This part is for local testing only. 
    # PythonAnywhere will not use this.
    app.run(debug=True, port=5000)
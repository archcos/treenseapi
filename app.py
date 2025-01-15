import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the TFLite model
model_path = "health_model.tflite"  # Update with your actual model path
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the image size expected by the model
IMAGE_SIZE = 180  # Update based on your model's expected input size
class_names = ["healthy", "unhealthy"]  # Adjust class names as per your model

# Preprocessing function to resize and normalize the image
def preprocess_image(image):
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    image_array = np.array(image)  # Convert image to numpy array
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    image_array = image_array.astype(np.float32)  # Ensure proper data type
    image_array = image_array / 255.0  # Normalize pixel values to [0, 1]
    return image_array

# Prediction function
def predict_with_tflite(image_array):
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], image_array)

    # Run inference
    interpreter.invoke()

    # Get the predicted class and confidence
    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = class_names[np.argmax(output[0])]
    confidence = round(100 * np.max(output[0]), 2)

    return predicted_class, confidence

# Define a route to handle image uploads and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Open image and preprocess it
        image = Image.open(file.stream)
        image_array = preprocess_image(image)

        # Make prediction
        predicted_class, confidence = predict_with_tflite(image_array)

        # Return prediction result as JSON
        return jsonify({
            'class': predicted_class,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

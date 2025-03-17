import os
import base64
import yaml
import numpy as np
import onnxruntime as ort
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS

# Load configuration file path from environment variables (for Render)
CONFIG_PATH = os.getenv("CONFIGS_PATH", "configs.yaml")
MODEL_PATH = os.getenv("MODEL_PATH", "model.onnx")

# Ensure required files exist
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"Config file not found: {CONFIG_PATH}")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"ONNX model file not found: {MODEL_PATH}")

# Load configuration
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

char_list = config.get("char_list", "23456789ABCDEFGHKLMNPRSTUVWYZabcdefghklmnprstuvwyz")
input_size = tuple(config.get("input_size", [100, 32]))  # Default (width, height)
blank_token = len(char_list)

# Initialize ONNX model session
session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for Chrome Extension

def preprocess_image(base64_string):
    """Decode Base64 image and preprocess it for ONNX model."""
    try:
        img_data = base64.b64decode(base64_string.split(",")[1])
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise ValueError("Decoded image is None.")

        image = cv2.resize(image, input_size)
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=(0, 1))  # Shape: (1, 1, H, W)
        return image
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def run_onnx_inference(image):
    """Run OCR inference using the ONNX model."""
    try:
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: image})[0]
        return result
    except Exception as e:
        raise RuntimeError(f"ONNX inference error: {str(e)}")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "OCR API is running"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "Missing 'image' field"}), 400

        image = preprocess_image(data["image"])
        result = run_onnx_inference(image)

        # Convert result to readable text
        predicted_text = "".join([char_list[i] for i in np.argmax(result, axis=2)[0] if i != blank_token])

        return jsonify({"captcha_text": predicted_text}), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except RuntimeError as re:
        return jsonify({"error": str(re)}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000, debug=False)

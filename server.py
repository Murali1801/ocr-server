from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import onnxruntime as ort
import base64
import os
import uuid
from mltu.configs import BaseModelConfigs

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}})

# Use relative paths for deployment on Render.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.onnx")
config_path = os.path.join(BASE_DIR, "configs.yaml")

# Load configuration and model
configs = BaseModelConfigs.load(config_path)
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

class OCRModel:
    def __init__(self, configs, session):
        self.session = session
        # Use the width and height from the configs for resizing.
        self.width = configs.width
        self.height = configs.height
        # Use the vocabulary list as the character list.
        self.char_list = configs.vocab

    def predict(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error: Could not load image {image_path}")
            return ""
        
        # Resize and preprocess image using the width and height from configs.
        image = cv2.resize(image, (self.width, self.height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0).astype(np.float32)
        
        # Get model predictions.
        preds = self.session.run(None, {self.session.get_inputs()[0].name: image})[0]
        
        # CTC Greedy Decoding:
        blank_id = len(self.char_list)  # The blank token is at the last index.
        decoded = []
        previous_char = None
        
        for time_step in preds[0]:
            char_id = np.argmax(time_step)
            if char_id != blank_id:
                current_char = self.char_list[char_id]
                if current_char != previous_char:
                    decoded.append(current_char)
                previous_char = current_char
            else:
                previous_char = None
        
        return ''.join(decoded)

# Instantiate the OCR model.
ocr_model = OCRModel(configs, session)

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided."}), 400

    # Expecting a Data URL: "data:image/png;base64,...."
    try:
        header, encoded = data["image"].split(",", 1)
    except Exception as e:
        return jsonify({"error": "Invalid image format."}), 400

    try:
        image_data = base64.b64decode(encoded)
    except Exception as e:
        return jsonify({"error": "Base64 decoding failed."}), 400

    # Save the decoded image to a temporary file.
    tmp_filename = f"temp_{uuid.uuid4().hex}.png"
    try:
        with open(tmp_filename, "wb") as f:
            f.write(image_data)
    except Exception as e:
        return jsonify({"error": "Failed to write temporary image file."}), 500

    # Use the OCR model's predict function.
    result_text = ocr_model.predict(tmp_filename)
    
    # Clean up the temporary file.
    try:
        os.remove(tmp_filename)
    except Exception as e:
        print(f"Warning: Failed to delete temporary file {tmp_filename}")

    return jsonify({"captcha_text": result_text})

if __name__ == "__main__":
    # Render provides the PORT environment variable.
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)

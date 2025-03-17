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

# Load Model (Ensure model and config files are in the project directory)
model_path = os.path.join(os.path.dirname(__file__), "model.onnx")
config_path = os.path.join(os.path.dirname(__file__), "configs.yaml")

configs = BaseModelConfigs.load(config_path)
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

class OCRModel:
    def __init__(self, configs, session):
        self.session = session
        self.width = configs.width
        self.height = configs.height
        self.char_list = configs.vocab

    def predict(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            return ""

        image = cv2.resize(image, (self.width, self.height))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.expand_dims(image, axis=0).astype(np.float32)

        preds = self.session.run(None, {self.session.get_inputs()[0].name: image})[0]

        # CTC Greedy Decoding
        blank_id = len(self.char_list)
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

ocr_model = OCRModel(configs, session)

@app.route("/predict", methods=["POST"])
def predict_route():
    data = request.json
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided."}), 400

    try:
        header, encoded = data["image"].split(",", 1)
        image_data = base64.b64decode(encoded)
    except Exception:
        return jsonify({"error": "Invalid image format."}), 400

    tmp_filename = f"temp_{uuid.uuid4().hex}.png"
    try:
        with open(tmp_filename, "wb") as f:
            f.write(image_data)
    except Exception:
        return jsonify({"error": "Failed to write temporary image file."}), 500

    result_text = ocr_model.predict(tmp_filename)
    
    try:
        os.remove(tmp_filename)
    except Exception:
        print(f"Warning: Failed to delete temporary file {tmp_filename}")

    return jsonify({"captcha_text": result_text})

# Entry point for Vercel
def handler(event, context):
    return app(event, context)

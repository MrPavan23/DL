import time
import logging
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
import tensorflow as tf
import gdown
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Global variable for the model
model = None

# Google Drive file ID
file_id = '1nbTF8Y58INIuy2Qwujfs1d625vNsWbpM'
model_path = 'DL.h5'

def download_model():
    if not os.path.exists(model_path):
        gdown.download(f'https://drive.google.com/uc?export=download&id={file_id}', model_path, quiet=False)

def load_model():
    global model
    download_model()
    model = tf.keras.models.load_model(model_path)

def setup():
    start_time = time.time()
    load_model()
    logging.debug(f"Model loaded in {time.time() - start_time} seconds")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image = Image.open(file)
    image = image.resize((224, 224))  # Resize image to match model input
    image_array = np.array(image) / 255.0  # Normalize image
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension

    prediction = model.predict(image_array)
    bird_name = get_bird_name(prediction)  # Implement this function based on your model's output

    return jsonify({'birdName': bird_name})

def get_bird_name(prediction):
    # Convert model output to bird name
    # This should be implemented based on your model's output
    return 'Example Bird'

if __name__ == '__main__':
    setup()  # Manually call setup to load the model before starting the server
    app.run(debug=True)

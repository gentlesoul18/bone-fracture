from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# Load the pre-trained model
def load_model_with_custom_objects(model_path):
    try:
        logger.info(f"Attempting to load model from path: {model_path}")
        def remove_batch_shape(config):
            if 'batch_shape' in config:
                del config['batch_shape']
            return config

        def custom_object_scope():
            return tf.keras.utils.custom_object_scope({
                'InputLayer': lambda **kwargs: tf.keras.layers.InputLayer(**remove_batch_shape(kwargs))
            })

        with custom_object_scope():
            model = tf.keras.models.load_model(model_path)
            logger.info("Model loaded successfully")
            return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Get the absolute path to the model file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrainedModel.h5')
logger.info(f"Model path: {model_path}")

# Load the pre-trained model
model = load_model_with_custom_objects(model_path)
if model:
    logger.info(f"Model input shape: {model.input_shape}")
    logger.info(f"Model output shape: {model.output_shape}")
else:
    logger.error("Failed to load model")

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('L')
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 128, 128, 1)
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/api/detect-fracture', methods=['POST'])
@cross_origin()
def detect_fracture():
    try:
        logger.info("Received fracture detection request")
        logger.debug(f"Request files: {request.files}")
        
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        image_bytes = image_file.read()
        logger.info(f"Image size: {len(image_bytes)} bytes")
        
        if not model:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not available'}), 500
        
        # Preprocess the image
        preprocessed_image = preprocess_image(image_bytes)
        logger.info("Image preprocessed successfully")
        
        # Make prediction
        prediction = model.predict(preprocessed_image)
        logger.info(f"Prediction made: {prediction}")
        
        # Assuming binary classification (fractured or not fractured)
        result = 'Fractured' if prediction[0][0] > 0.5 else 'Not Fractured'
        logger.info(f"Final result: {result}")
        
        return jsonify({'result': result})
    except Exception as e:
        logger.error(f"Error in detect_fracture: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
@cross_origin()
def index():
    try:
        logger.info("Serving index.html")
        return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get port from environment variable for Render deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
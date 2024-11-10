from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import sys
import os
import logging
import traceback

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='static')
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

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
try:
    model = load_model_with_custom_objects(model_path)
    if model:
        logger.info(f"Model input shape: {model.input_shape}")
        logger.info(f"Model output shape: {model.output_shape}")
    else:
        logger.error("Failed to load model")
except Exception as e:
    logger.error(f"Error during model loading: {str(e)}")
    model = None

def preprocess_image(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('L')
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = image_array.reshape(1, 128, 128, 1)
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}\n{traceback.format_exc()}")
        raise

@app.route('/api/detect-fracture', methods=['POST'])
@cross_origin()
def detect_fracture():
    try:
        logger.info("Received fracture detection request")
        
        # Check if the post request has the file part
        if 'image' not in request.files:
            logger.error("No image file in request")
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        # If user does not select file, browser also submits an empty part without filename
        if image_file.filename == '':
            logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
        
        # Read the image file
        try:
            image_bytes = image_file.read()
            logger.info(f"Successfully read image file of size: {len(image_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error reading image file: {str(e)}")
            return jsonify({'error': 'Error reading image file'}), 400
        
        # Check if model is loaded
        if model is None:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not available'}), 500
        
        # Preprocess the image
        try:
            preprocessed_image = preprocess_image(image_bytes)
            logger.info("Image preprocessed successfully")
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            return jsonify({'error': 'Error preprocessing image'}), 400
        
         # Make prediction
        try:
            logger.info("Starting prediction...")
            logger.info(f"Preprocessed image shape: {preprocessed_image.shape}")
            logger.info(f"Model input shape: {model.input_shape}")
            
            # Verify input shape matches model's expected input
            if preprocessed_image.shape[1:] != model.input_shape[1:]:
                logger.error(f"Shape mismatch: Got {preprocessed_image.shape}, expected {model.input_shape}")
                return jsonify({'error': 'Input shape mismatch'}), 400

            # Add timeouts and catch potential memory issues
            try:
                with tf.device('/CPU:0'):  # Force CPU usage to avoid GPU memory issues
                    logger.info("Starting model.predict...")
                    prediction = model.predict(preprocessed_image, verbose=1)
                    logger.info("Completed model.predict")
            except tf.errors.ResourceExhaustedError as e:
                logger.error(f"Memory error during prediction: {str(e)}")
                return jsonify({'error': 'Memory error during prediction'}), 500
            except Exception as e:
                logger.error(f"Error during model.predict: {str(e)}")
                return jsonify({'error': 'Prediction failed'}), 500

            logger.info(f"Raw prediction shape: {prediction.shape}")
            logger.info(f"Raw prediction values: {prediction}")
            
            if not isinstance(prediction, np.ndarray):
                logger.error(f"Unexpected prediction type: {type(prediction)}")
                return jsonify({'error': 'Invalid prediction type'}), 500
                
            if prediction.size == 0:
                logger.error("Empty prediction array")
                return jsonify({'error': 'Empty prediction'}), 500

            try:
                result = 'Fractured' if prediction[0][0] > 0.5 else 'Not Fractured'
                logger.info(f"Classified as: {result}")
                
                response_data = {'result': result}
                logger.info(f"Preparing response: {response_data}")
                
                json_response = jsonify(response_data)
                logger.info("JSON response created successfully")
                
                return json_response
                
            except Exception as e:
                logger.error(f"Error formatting prediction result: {str(e)}\n{traceback.format_exc()}")
                return jsonify({'error': 'Error formatting result'}), 500
            
        except Exception as e:
            logger.error(f"Error in prediction block: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': 'Error during prediction processing'}), 500
    except Exception as e:
            logger.error(f"Error in prediction block 2: {str(e)}\n{traceback.format_exc()}")
            return jsonify({'error': 'Error during prediction processing 2'}), 500
@app.route('/')
@cross_origin()
def index():
    try:
        return send_from_directory(app.static_folder, 'index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        return jsonify({'error': 'Error serving page'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port)
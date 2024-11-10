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
import signal
from contextlib import contextmanager
import gc

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Timeout handler
class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Prediction timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

app = Flask(__name__, static_folder='static')
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Load model with memory optimization
def load_model_with_custom_objects(model_path):
    try:
        logger.info(f"Attempting to load model from path: {model_path}")
        
        # Clear any existing models and free memory
        tf.keras.backend.clear_session()
        gc.collect()
        
        # Configure memory growth
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        
        # Load model with optimization flags
        model = tf.keras.models.load_model(
            model_path,
            compile=False  # Don't compile the model if you're only doing inference
        )
        
        # Optimize for inference
        model.make_predict_function()  # Create predict function beforehand
        
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

# Get model path
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pretrainedModel.h5')
logger.info(f"Model path: {model_path}")

# Load the model
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
        # Optimize image processing
        image = Image.open(io.BytesIO(image_bytes))
        image = image.convert('L')
        image = image.resize((128, 128), Image.Resampling.LANCZOS)
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_array = image_array.reshape(1, 128, 128, 1)
        return image_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        raise

@app.route('/api/detect-fracture', methods=['POST'])
@cross_origin()
def detect_fracture():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read and preprocess image
        image_bytes = image_file.read()
        preprocessed_image = preprocess_image(image_bytes)
        
        # Clear memory before prediction
        gc.collect()
        
        try:
            # Set a timeout for prediction (adjust seconds as needed)
            logger.info("Starting prediction...")
            logger.info(f"Preprocessed image shape: {preprocessed_image.shape}")
            logger.info(f"Model input shape: {model.input_shape}")
            
            with time_limit(30):
                with tf.device('/CPU:0'):
                    # Run prediction in optimized mode
                    prediction = model.predict(
                        preprocessed_image,
                        batch_size=1,
                        verbose=0,
                    )
                    
                    # Process result immediately
                    result = 'Fractured' if prediction[0][0] > 0.5 else 'Not Fractured'
                    confidence = float(prediction[0][0])
                    
                    return jsonify({
                        'result': result,
                        'confidence': confidence,
                        'status': 'success'
                    })
                    
        except TimeoutException as e:
            logger.error("Prediction timed out")
            return jsonify({'error': 'Prediction timed out'}), 503
        except tf.errors.ResourceExhaustedError as e:
            logger.error(f"Memory error during prediction: {str(e)}")
            return jsonify({'error': 'Memory error during prediction'}), 503
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return jsonify({'error': 'Prediction failed'}), 500
            
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, threaded=False)  # Disable threading for better memory management
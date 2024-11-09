from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import sys

app = Flask(__name__, static_folder='static')
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
# Load the pre-trained model
def load_model_with_custom_objects(model_path):
    def remove_batch_shape(config):
        if 'batch_shape' in config:
            del config['batch_shape']
        return config

    def custom_object_scope():
        return tf.keras.utils.custom_object_scope({
            'InputLayer': lambda **kwargs: tf.keras.layers.InputLayer(**remove_batch_shape(kwargs))
        })

    with custom_object_scope():
        return tf.keras.models.load_model(model_path)

# Load the pre-trained model
try:
    model = load_model_with_custom_objects('./pretrainedModel.h5')
    print("Model loaded successfully")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    print(f"Error loading model: {str(e)}", file=sys.stderr)
    model = None

def preprocess_image(image_bytes):
   image = Image.open(io.BytesIO(image_bytes))
   image = image.convert('L')
   image = image.resize((128, 128))
   image_array = np.array(image) / 255.0
   image_array = image_array.reshape(1, 128, 128, 1)
   return image_array

@app.route('/api/detect-fracture', methods=['POST'])
@cross_origin()
def detect_fracture():
    print('first got here')
    print(request.files)
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    image_file = request.files['image']
    print('second got here')
    image_bytes = image_file.read()
    print('third got here')
    
    # Preprocess the image
    preprocessed_image = preprocess_image(image_bytes)
    print('fourth got here')
    
    # Make prediction
    prediction = model.predict(preprocessed_image)
    print('fifth got here')
    print(prediction)
    
    # Assuming binary classification (fractured or not fractured)
    result = 'Fractured' if prediction[0][0] > 0.5 else 'Not Fractured'
    
    return jsonify({'result': result})
@app.route('/')
@cross_origin()
def index():
    print('got here')
    return send_from_directory(app.static_folder, 'index.html')
if __name__ == '__main__':
    app.run(debug=True)

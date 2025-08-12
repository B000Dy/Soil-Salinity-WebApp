
import os
import sys
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, url_for, send_from_directory

app = Flask(__name__)
app.secret_key = 'supersecretkey'  # Required for session usage

def get_model_path():
    if getattr(sys, 'frozen', False):
        base_dir = sys._MEIPASS
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_dir, 'model-250.h5')

MODEL_PATH = get_model_path()

# Verify model exists before loading
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

class_labels = ['12K', '4K', '8K', 'Control']  # Replace with actual class names

UPLOAD_FOLDER = 'uploads'
UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Load and preprocess the image
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    confidence_score = float(np.max(preds[0]))
    confidence = round(confidence_score * 100, 2)

    if confidence_score < 0.3:  # adjust threshold as needed
        class_name = "Not a root image"
    else:
        class_idx = np.argmax(preds[0])
        class_name = class_labels[class_idx]


    # Store in session for later use
    session['label'] = class_name
    session['confidence'] = confidence
    session['filename'] = filename

    return redirect(url_for('loading'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/loading')
def loading():
    return render_template('loading.html')

@app.route('/result')
def result():
    label = session.get('label', 'Unknown')
    confidence = session.get('confidence', 0.0)
    filename = session.get('filename', None)
    return render_template('result.html', label=label, confidence=confidence, filename=filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

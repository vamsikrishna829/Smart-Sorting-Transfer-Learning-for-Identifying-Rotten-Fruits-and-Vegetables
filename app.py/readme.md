import os
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder configuration
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained model
model = load_model('Healthy_vs_rotten.h5')

# Class labels (change if your model uses different order)
classes = ['Fresh', 'Rotten']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]
    confidence = round(100 * np.max(prediction), 2)

    return result, confidence


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    result, confidence = predict_image(filepath)

    return render_template('result.html',
                           prediction=result,
                           confidence=confidence,
                           image_path=filepath)


if __name__ == '__main__':
    app.run(debug=True)

import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf

# Importing libraries for Keras:
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask libraries:
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Defining a flask app:
app = Flask(__name__)

modelPath = '/home/jackson/Desktop/Coding/Data-Science & Machine Learning/Deep-Learning/Diseased-Cotton-Leaf-Prediction/Models/modelInceptionV3.h5'

# Loading our Inception v3 model:
model = load_model(modelPath)

def model_predict(img_path, model):
    print(img_path)
    img = image.load_img(img_path, target_size = (224, 224))
    
    # Preprocessing the image:
    x = image.img_to_array(img)
    # Scaling our image:
    x = x/255
    x = np.expand_dims(x, axis=0)
    
    predictions = model.predict(x)
    predictions = np.argmax(predictions, axis=1)
    
    if predictions == 0:
        predictions = "The leaf is a diseased cotton leaf."
    elif predictions == 1:
        predictions = "The leaf is a diseased cotton plant."
    elif predictions == 2:
        predictions = "The leaf is a fresh cotton leaf."
    else:
        predictions = "The leaf is a fresh cotton plant."
        
        
    return predictions


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        
        basePath = os.path.dirname(__file__)
        file_path = os.path.join(
            basePath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Making our prediction:
        prediction = model_predict(file_path, model)
        result = prediction
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

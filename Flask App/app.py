from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Tensorflow
import tensorflow as tf
import cv2

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models//model_handmade.h5'

# Load your trained model
model = load_model(MODEL_PATH, compile = False)
# model._make_predict_function()
# print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):

    # Preprocessing the image
    IMG_SIZE_LOAD = 224
    xtest = np.zeros((1, 224, 224, 3))

    image = cv2.imread(img_path)
    image = cv2.resize(image, (IMG_SIZE_LOAD, IMG_SIZE_LOAD))
    
    x_test_array = tf.keras.preprocessing.image.img_to_array(image)
    xtest[0] = x_test_array
    xtest = xtest / 255.0

    # Make Prediction
    preds = model.predict(xtest)

    l = preds[0]
    d = {0 : 'NoDR', 1 : 'Mild', 2 : 'Moderate', 3 : 'Severe', 4 : 'Proliferative'}
    result = d[np.where(l == max(l))[0][0]]

    return result


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Return Prediction
        preds = model_predict(file_path, model)

        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)

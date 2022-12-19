#!/usr/bin/env python
# coding: utf-8

# for array operations
import numpy as np

# for img download from url
from io import BytesIO
from urllib import request

# for img preprocessing
from PIL import Image

# for running tensorflow model
import tensorflow
from tensorflow import keras

model = keras.models.load_model('cnn_2.1')

def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def predict(url):
    img = download_image(url)
    img = prepare_image(img, (299, 299))

    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.applications.xception import preprocess_input
        
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
      
    classes = ['cup', 'fork', 'glass', 'knife', 'plate', 'spoon']
    return classes[preds.argmax(axis=1)[0]]

def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return(result)



#pip install -q -U "tensorflow-gpu==2.0.0b1"
#pip install -q -U tensorflow_hub
#python predict.py --img_path ./test_images/orange_dahlia.jpg --model 1631055444.h5 --top_k 4 --category_names ./label_map.json

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import logging
import argparse
import sys
import json
from PIL import Image

logger = tf.get_logger()
logger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser ()
parser.add_argument ('--img_path', default='./test_images/cautleya_spicata.jpg', help = 'Enter full image path', type = str)
parser.add_argument('--model', default='1675553120.h5', help='Enter model path and name', type=str)
parser.add_argument ('--top_k', default = 5, help = 'Top K most likely classes', type = int)
parser.add_argument ('--category_names' , default = './label_map.json', help = 'Mapping categories to real flower names', type = str)
commands = parser.parse_args()
image_path = commands.img_path
model_path_keras = commands.model
classes = commands.category_names
top_k = commands.top_k
reloaded_model = tf.keras.models.load_model(model_path_keras,custom_objects={'KerasLayer': hub.KerasLayer})

# Create the process_image function
with open(classes, 'r') as f:
    class_names = json.load(f)

def process_image(numpy_image):
    image_size = 224
    processed_image = tf.convert_to_tensor(numpy_image, dtype=tf.float32)
    processed_image = tf.image.resize(processed_image, (image_size, image_size))
    processed_image /= 255
    return processed_image.numpy()

def predict(imagepath, model, top_k):
    image = Image.open(imagepath)
    imagenp = np.asarray(image)
    transformed_image = process_image(imagenp)
    expdim_image = np.expand_dims(transformed_image, axis=0)
    prob_pred = model.predict(expdim_image)
    prob_pred = prob_pred.tolist()
    
    prob, topindices = tf.math.top_k(prob_pred, k=top_k)
    topclasses = [class_names[str(i+1)] for i in topindices.cpu().numpy()[0]]
    prob=prob.numpy().tolist()[0]
    
    return prob,topclasses
probs, classes = predict(image_path, reloaded_model, top_k)
print('\nTop classes are - ',classes)
print('\nTop probabilities are - ',probs)

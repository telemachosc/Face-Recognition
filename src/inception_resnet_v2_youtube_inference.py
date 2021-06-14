# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 10:13:54 2021

@author: telemachos
"""

from os.path import isdir
import os
import platform

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


def normalizer(img):
    face_pixels = img.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    # Store sample
    return (face_pixels - mean) / std

# extract a single face from a given photograph
def extract_face(filename, required_size=(299, 299)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    if results != []:
        # extract the bounding box from the first face
        x1, y1, width, height = results[0]['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)

        face_array = normalizer(face_array)
        return face_array
    else:
        return None

def load_faces(directory):
    faces = list()
    # enumerate files
    for subdir in os.listdir(directory):
        # path
        path = os.path.join(directory, subdir)
        for filename in os.listdir(path):
            filepath = os.path.join(path,filename)
            # print(filename)
            # get face
            face = extract_face(filepath)
            # store
            faces.append(face)

    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
	# enumerate folders, on per class
    for subdir in os.listdir(directory):
        path = os.path.join(directory, subdir)
        print(subdir)
        # skip any files that might be in the dir
        if not isdir(path):
            continue
		# load all faces in the subdirectory
        faces = load_faces(path)
            
		# create labels
        labels = [subdir for _ in range(len(faces))]
        # summarize progress
        print('>loaded %d examples for class: %s' % (len(faces), subdir))
        
        # store
        X.extend(faces)
        y.extend(labels)

    return np.asarray(X), np.asarray(y)


if __name__=='__main__':
    path = {"server": "/data/tchatz/youtube_resnet_inference",
            "local": "data\youtube_resnet_inference"
             }
    
    if platform.system()=='Windows':
        path = path['local']
    else: 
        path = path['server']
    
    x, y = load_dataset(path)
    
    # load pretrained model
    model = tf.keras.models.load_model("pretrained/InceptionResnetV2_youtube_64_20_1e-4_adam_CCE")

    
    pr = model.predict(x)
    print(np.where(pr==1))

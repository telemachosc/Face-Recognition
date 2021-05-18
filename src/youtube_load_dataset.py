# -*- coding: utf-8 -*-
"""
Created on Mon May 17 11:54:19 2021

@author: telemachos
"""

from os import listdir
from os.path import isdir
import os
from PIL import Image
from matplotlib import pyplot
import numpy as np
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

def pathmaker(path):
    path = path.split('/')
    path = os.path.join(*path)
    return path

# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
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
 	return face_array

def load_faces(directory):
    faces = list()
    # enumerate files
    for subdir in listdir(directory):
        # path
        path = os.path.join(directory, subdir)
        for filename in listdir(path):
            filepath = os.path.join(path,filename)
            print(filename)
            # get face
            face = extract_face(filepath)
            # store
            faces.append(face)
    return faces

# load a dataset that contains one subdir for each class that in turn contains images
def load_dataset(directory):
    X, y = list(), list()
	# enumerate folders, on per class
    for subdir in listdir(directory):
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


path = ("C:\/Users/telemachos/Documents/Programming/python/"+
        "projects/Celebrity detection/datasets/youtube/aligned_images_DB")

npath = pathmaker(path)

x, y = load_dataset(npath)

#%%


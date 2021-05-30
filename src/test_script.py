# -*- coding: utf-8 -*-
"""
Created on Mon May 10 16:59:05 2021

@author: telemachos
"""

from mtcnn import MTCNN
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



#%%

def draw_face_box(path):
    img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    detected_faces = detector.detect_faces(img)
    
    x, y, k, l = detected_faces[0]['box']
    
    start, end = (x, y), (x+k, y+l)
    
    rect_color = (0, 255, 0)
    
    thickness = 1

    cv2.rectangle(img, start, end, rect_color, thickness)
    fig, ax = plt.subplots(2)
    ax[0].imshow(img)
    ax[1].imshow(img[y:y+l, x:x+k])
    # plt.show()
    return detected_faces
    

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
    # extract the bounding box from the faces
    facelist = []
    for faces in results:
        x1, y1, width, height = faces['box']
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        facelist.append(face_array)
    return facelist

def crop_image(img, y, y_, x, x_):
    print(y, y_, x, x_)
    crop = img[y:y_, x:x_]
    cv2.imshow('sf',crop)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return crop

def imshow(img):
    cv2.imshow('sf',crop)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    
#%%

file = "data/youtube/aligned_images_DB/Aaron_Eckhart/0/aligned_detect_0.575.jpg"



box = draw_face_box(file)
#%%

img = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2RGB)
detector = MTCNN()
detected_faces = detector.detect_faces(img)
    
x, y, k, l = detected_faces[0]['box']

plt.imshow(img[x:x+k, y:y+l])
plt.show()


#%%


#%%
crop =crop_image(img, **{'x': 50,
                 'x_': 150,
                 'y': 50,
                 'y_': 150})
#%%


#%%
test = extract_face(crop)


for face in test:
    plt.imshow(face)
    plt.show()
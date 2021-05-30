# -*- coding: utf-8 -*-
"""
Created on Thu May 27 14:51:07 2021

@author: telemachos
"""
import os
import platform
from pathlib import Path

from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from PIL import Image
import numpy as np
import time

tf.get_logger().setLevel('ERROR')

def load_paths(path: str, savepath) -> list:    
    x = []
    paths = []
    for base, dirs, files in os.walk(path):
        if files:
            x.extend([base+os.sep+file for file in files])
            
    if platform.system()=='Windows':
        paths = [os.path.join(savepath, *path.split('\\')[-3:-1]) for path in x]
    
    else:
        paths = [os.path.join(savepath, *path.split('/')[-3:-1]) for path in x]
    # keep only unique values of the list
    paths = list(set(paths))

    return x, paths

def make_dirs(paths):
    for fp in paths:
        try:
            Path(fp).mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass



def _savepath(path, save_path):
    if platform.system()=='Windows':
        pathsplit = path.split('\\')[-3:]
    else:
        pathsplit = path.split('/')[-3:]
    sp = os.path.join(save_path,os.path.join(*pathsplit))
    return sp

def extract_face(filepath, savepath, dim=(160,160)):
    # load image from file
    image = Image.open(filepath)
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
        image = image.resize(dim)
        sp = _savepath(filepath, savepath)
        image.save(sp)


def main(paths, save_path):
    i=0
    for path in paths:
        s = time.time()
        extract_face(path, save_path)
        e = time.time()
        i+=1
        print(i, e-s)
        

if __name__=="__main__":
    path = {"server": [
                 "/data/tchatz/aligned_images_DB",
                 "/data/tchatz/you_tube_processed"
                 ],
             "local": [
                 "data/youtube/aligned_images_DB",
                 "data/tmp"
                 ]
             }
    if platform.system()=='Windows':
        path = [os.path.join(*pt.split('/')) for pt in path['local']]
    else: 
        path = path['server']
    
    # Make paths to load images and save processed images
    filepaths, paths = load_paths(path[0], path[1])
    
    # Make directories for the processed images
    make_dirs(paths)
    
    main(filepaths, path[1])

# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:34:05 2021

@author: telemachos
"""
import platform
import os
from src.face_detection_pipeline import DataGenerator, load_paths
import time


path = {'local': "data/youtube/aligned_images_DB",
        'server': '/data/tchatz/aligned_images_DB'}
# Remake path to make it windows compatible
if platform.system()=='Windows':
    path = os.path.join(*path['local'].split('/'))
else: 
    path = path['server']

partition, labels = load_paths(path, 0.8, 0.1)

num_classes = len(set(val for val in labels.values()))

params = {
'dim': (160,160),
'batch_size': 32,
'n_classes': num_classes,
'n_channels': 3,
'shuffle': True
}


training_generator = DataGenerator(partition['train'], labels, **params)


for i in range(3):
    tmp = training_generator.__getitem__(i)
    

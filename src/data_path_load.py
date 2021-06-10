# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 13:13:53 2021

@author: telemachos

Data location loading

Retrieve paths from directory structure or text files to be used for data
generation.
"""
import platform
import random
import os

import pandas as pd
from sklearn.preprocessing import LabelBinarizer



class PathLoader():
    
    def __init__(self, paths: dict, load_type: str, train_pct: float=0.8,
                 val_pct: float=0.1):
        self.paths = paths
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.load_type = getattr(self, load_type)
        
        
    def dir_load(self, paths):
        x = []
        labels_list = []
        
        for base, dirs, files in os.walk(path):
            if files:
                x.extend([base+os.sep+file for file in files])
                if platform.system()=='Windows':
                    labels_list.extend([base.split('\\')[-2] for _ in range(len(files))])
                else:
                    labels_list.extend([base.split('/')[-2] for _ in range(len(files))])
        
        # Shuffle the x
        random.shuffle(x)
        
        # One hot encode the labels
        encoder = LabelBinarizer()
        
        labels_list = encoder.fit_transform(labels_list).tolist()
        
        # Make dictionaries for x and y with correct names
        labels = {inpt: label for inpt, label in zip(x, labels_list)}
        
        # Split train, validation and test sets
        train = x[:int(len(x)*trainpct)]
        tmp = x[int(len(x)*trainpct):]
        val = tmp[:int(len(x)*valpct)]
        test = tmp[int(len(x)*valpct):]
        
        # Check if train, val and test have the same length as x
        if not len(train) + len(val) + len(test) == len(x):
            raise ValueError(("The length of the produced dictionaries is"+
                             " not the same as the original"))
        
        return {"train": train, "val": val, "test": test}, labels


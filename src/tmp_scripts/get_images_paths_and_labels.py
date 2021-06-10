# -*- coding: utf-8 -*-
"""
Created on Sat May 22 00:51:58 2021

@author: telemachos
"""
import os

def load_paths(path: str , trainpct: float, valpct: float) -> dict:
    # Remake path to make it OS compatible
    path = os.path.join(*path.split('/'))
    
    x_list = []
    labels_list = []
    
    for base, dirs, files in os.walk(path):
        if files:
            x_list.extend([base+os.sep+file for file in files])
            labels_list.extend([base.split('\\')[-2] for _ in range(len(files))])
            
    # Make dictionaries for x and y with correct names
    x = {f"id_{num}": file for num, file in enumerate(x_list)}
    labels = {f"id_{num}": label for num, label in enumerate(labels_list)}
    
    # Split train, validation and test sets
    train = dict(list(x.items())[:int(len(x)*trainpct)])
    tmp = dict(list(x.items())[int(len(x)*trainpct):])
    val = dict(list(tmp.items())[:int(len(x)*valpct)])
    test = dict(list(tmp.items())[int(len(x)*valpct):])
    
    # Check if train, val and test have the same length as x
    if not len(train) + len(val) + len(test) == len(x):
        raise ValueError(("The length of the produced dictionaries is"+
                         " not the same as the original"))
    
    return {"train": train, "val": val, "test": test}, labels




if __name__=="__main__":
    partition, labels = load_paths("data/youtube/aligned_images_DB", 0.8, 0.1)
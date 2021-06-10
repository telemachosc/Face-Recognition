# -*- coding: utf-8 -*-
""" 
Created on Wed Jun  9 12:52:38 2021

@author: telemachos

Miscellaneous functions used when training a model
"""

import json
import os

import numpy as np
from keras.callbacks import LearningRateScheduler


class StrIntEncoder:

    @staticmethod
    def encode(s: str) -> int:
        encode = s.encode('utf-8')
        return int.from_bytes(encode, byteorder='big')

    @staticmethod
    def decode(i: int) -> str:
        tobytes = i.to_bytes(((i.bit_length() + 7) // 8), byteorder='big')
        return tobytes.decode("utf-8")



def file_exists(name: str) -> str:
    if os.path.exists('results/'+name+'.txt'):
        ds, num = name.split('_')
        hfname = ds+'_'+str(int(num)+1).zfill(3)
        if os.path.exists('results/'+hfname+'.txt'):
            hfname = file_exists(hfname)
    else:
        hfname = name
    
    return hfname


def write_history(name: str, h: dict, test_set: list, time: float,
                  epoch, lr, fnl) -> None:
    
    name = file_exists(name)    
        
    with open('results/'+name+'.txt', 'w') as convert_file:
        convert_file.write(('-----------------------------------------------'+
                           '-------------------------------\n'))
        convert_file.write(f'{name} EXPERIMENT RESULTS\n')
        convert_file.write(('-----------------------------------------------'+
                           '-------------------------------\n'))
        convert_file.write(('Train configuration for that experiment was:\n'+
                            f'Epochs: {epoch},\nLearning rate: {lr},\n'+
                            f'Fine tune layers: {fnl}'))
        convert_file.write(('Loss and accuracy on train and validation set\n'))
        convert_file.write(json.dumps(h, indent=2))
        convert_file.write(('\n\n'+ 
                           '----------------------------------------------\n'))
        convert_file.write('On the test set the loss and accuracy was:\n')
        convert_file.write(f'loss: {test_set[0]}, \naccuracy: {test_set[1]}\n')
        convert_file.write(('---------------------------------------------\n'+
                            '\nTotal time for training and evaluating the '+
                            f'model was\n{time} seconds or {time/60} minutes'))


def step_decay_schedule(initial_lr=1e-3, decay_factor=0.75, step_size=10):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch/step_size))
    
    return LearningRateScheduler(schedule)
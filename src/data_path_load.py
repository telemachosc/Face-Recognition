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
import pathlib

import pandas as pd
from sklearn.preprocessing import LabelBinarizer



class PathLoader():
    
    def __init__(self, path: str, load_type: str, train_pct: float=0.8,
                 val_pct: float=0.1):
        self.path = path
        self.train_pct = train_pct
        self.val_pct = val_pct
        self.load_type = load_type
        
    
    def path_type(self): 
        if self.load_type=='dir':
            return self.dir_load(self.paths)
        elif self.load_type=='file':
            return self.file_load(self.paths)
        else:
            raise ValueError(('The specified load_type is incorrect. Select'+
                             ' between "file" or "dir"'))
    
    def dir_load(self)-> pd.DataFrame:
        path = self.path
        x = []
        labels_list = []
        
        for base, dirs, files in os.walk(path):
            if files:
                x.extend([base+os.sep+file for file in files])
                if platform.system()=='Windows':
                    labels_list.extend([base.split('\\')[-2] for _ in range(len(files))])
                else:
                    labels_list.extend([base.split('/')[-2] for _ in range(len(files))])
        
        
        # Assign the resulting dataframe to a self value
        
        # Shuffle the x
        # random.shuffle(x)
        
        # # One hot encode the labels
        # encoder = LabelBinarizer()
        
        # labels_list = encoder.fit_transform(labels_list).tolist()
        
        # # Make dictionaries for x and y with correct names
        # labels = {inpt: label for inpt, label in zip(x, labels_list)}
        
        # # Split train, validation and test sets
        # train = x[:int(len(x)*trainpct)]
        # tmp = x[int(len(x)*trainpct):]
        # val = tmp[:int(len(x)*valpct)]
        # test = tmp[int(len(x)*valpct):]
        
        # # Check if train, val and test have the same length as x
        # if not len(train) + len(val) + len(test) == len(x):
        #     raise ValueError(("The length of the produced dictionaries is"+
        #                      " not the same as the original"))
        
        # return {"train": train, "val": val, "test": test}, labels
        
    def file_load(self):
        """
        path = 'data/CelebA'
        """
        path = self.path
        imgpath = 'Img/img_align_celeba'
        identity_file = 'Anno/identity_CelebA.txt'
        bbox_file = 'Anno/list_bbox_celeba.txt'
        eval_list = 'Eval/list_eval_partition.txt'
        
        
        data_dir = pathlib.Path(os.path.join(path, imgpath))
        
        # Load txt files in dataframes
        identity_df = pd.read_csv(os.path.join(path, identity_file), 
                                  delimiter = " ",
                                  names=['image_id','identity'])
        bbox_df = pd.read_csv(os.path.join(path, bbox_file), sep= "\s+", 
                              engine='python',
                              names=['image_id', 'x_1', 'y_1', 'width', 'height'],
                              skiprows=2)
        partition_df = pd.read_csv(os.path.join(path, eval_list), delimiter = " ",
                                   names=['image_id','partition'])
        
        # Merge the dfs 
        tmp_df = pd.merge(identity_df, bbox_df, on="image_id")
        full_df = pd.merge(tmp_df, partition_df, on="image_id")
        
        
        # Make the paths for the images and add it as a column in the final df
        full_df['path'] = data_dir.__str__() + os.sep + full_df['image_id']
        
        # Shuffle the dataframe
        full_df = full_df.sample(frac=1).reset_index(drop=True)
        
         # Assign the resulting dataframe to a self value
    
    def split_sets(self):
        """Split training, validation and test set"""
        pass
    
    def get_sets(self):
        """
        Gets the dataframes and the dictionary for the data generator.
        
        Split 

        Returns
        -------
        dict

        """
        


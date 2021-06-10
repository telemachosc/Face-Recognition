# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 13:12:26 2021

@author: telemachos
"""

import tensorflow as tf
import numpy as np
import PIL

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(160,160), 
                 n_channels=3, n_classes=10, shuffle=True, df=[]):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.df = df
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)
        # X, y = np.random.uniform(0, 1, (32,160,160,3)), np.random.uniform(0, 1, 32)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def extract_face(self, filename):
        # load image from file
        image = PIL.Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # resize pixels to the model size
        image = image.resize(self.dim)
        return np.asarray(image)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            face_pixels = self.extract_face(ID)

            face_pixels = face_pixels.astype('float32')
            # standardize pixel values across channels (global)
            mean, std = face_pixels.mean(), face_pixels.std()
            # Store sample
            X[i] = (face_pixels - mean) / std
            
            # Store class
            y[i]= self.labels[ID] - 1
            # y[i] = self.labels[ID] == self.n_classes

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes,
                                                dtype='float32')


class BBox(DataGenerator):
    def extract_face(self, filename):
        # load image from file
        image = PIL.Image.open(filename)
        # convert to RGB, if needed
        image = image.convert('RGB')
        # convert to array
        pixels = np.asarray(image)
        
        # TODO: This needs to be more generic. For celeb_a is fine
        # extract the bounding box from the first face
        x1, y1, width, height = self.df.loc[self.df['path']==filename].values[0][2:6]
        # bug fix
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        # extract the face
        face = pixels[y1:y2, x1:x2]
        # resize pixels to the model size
        image = PIL.Image.fromarray(face)
        image = image.resize(self.dim)
        return np.asarray(image)


class MTCNN(DataGenerator):
    
    def extract_face(self, filename):
        # load image from file
        image = PIL.Image.open(filename)
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
            image = PIL.Image.fromarray(face)
            image = image.resize(self.dim)
            face_array = np.asarray(image)
            return face_array
        else:
            return None
        
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            face_pixels = self.extract_face(ID)
            if isinstance(face_pixels, np.ndarray):
                face_pixels = face_pixels.astype('float32')
                # standardize pixel values across channels (global)
                mean, std = face_pixels.mean(), face_pixels.std()
                # Store sample
                X[i,] = (face_pixels - mean) / std
                
                # X[i,] = np.load('data/' + ID + '.npy')
    
                # Store class
                y[i] = int(self.labels[ID] == self.n_classes)

        return X, y
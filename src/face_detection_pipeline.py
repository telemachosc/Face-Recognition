# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:21:02 2021

@author: telemachos
"""

import os

import numpy as np
import tensorflow as tf
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
# from tensorflow.keras.models import load_model


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

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(160,160), 
                 n_channels=3, n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
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

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def extract_face(self, filename):
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
        image = image.resize(self.dim)
        face_array = np.asarray(image)
        return face_array

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
            X[i,] = (face_pixels - mean) / std
            
            # X[i,] = np.load('data/' + ID + '.npy')

            # Store class
            y[i] = self.labels[ID]

        return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)





if __name__=="__main__":
    partition, labels = load_paths("data/youtube/aligned_images_DB", 0.8, 0.1)
    
    num_classes = len(set(val for val in labels.values()))
    
    params = {
        'dim': (160,160),
        'batch_size': 64,
        'n_classes': num_classes,
        'n_channels': 3,
        'shuffle': True
        }
    
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['validation'], labels, **params)
    
    # Load the facenet model
    facenet = tf.keras.models.load_model('keras-facenet/model/facenet_keras.h5')
    
    # Freeze facenet from training
    facenet.trainable = False
    
    # Classifier design
    layer2 = tf.keras.layers.Flatten()
    layer3 = tf.keras.layers.Dense(num_classes, activation='softmax',
                                   kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.1))
    classifier = tf.keras.Sequential([facenet, layer2, layer3])
    
    classifier.compile(optimizer='adam',
                       loss=tf.keras.losses.CategoricalCrossentropy(),
                       metrics=['accuracy', 
                                tf.keras.metrics.Precision(),
                                tf.keras.metrics.Recall()])
    # Define callbacks
    
    checkpoint_path = 'tmp/checkpoints'
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_freq='epoch',
        save_weights_only=True,
        verbose=1)
    
    epochs = 10
    
    # Train model on dataset
    classifier.fit_generator(generator=training_generator,
                             epochs=epochs,
                             validation_data=validation_generator,
                             callbacks = [checkpoint],
                             use_multiprocessing=True,
                             workers=6)
# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:21:02 2021

@author: telemachos
"""

import os
import platform

import numpy as np
import tensorflow as tf
from PIL import Image
from mtcnn.mtcnn import MTCNN
# import numpy as np
tf.get_logger().setLevel('ERROR')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time


def load_paths(path: str , trainpct: float, valpct: float) -> dict:    
    x = []
    labels_list = []
    
    for base, dirs, files in os.walk(path):
        if files:
            x.extend([base+os.sep+file for file in files])
            if platform.system()=='Windows':
                labels_list.extend([base.split('\\')[-2] for _ in range(len(files))])
            else:
                labels_list.extend([base.split('/')[-2] for _ in range(len(files))])
            
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
        # X, y = np.random.uniform(0, 1, (32,160,160,3)), np.random.uniform(0, 1, 32)

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
        s = time.time()
        detector = MTCNN()
        # detect faces in the image
        results = detector.detect_faces(pixels)
        print(time.time()-s)
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





if __name__=="__main__":
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
    validation_generator = DataGenerator(partition['val'], labels, **params)
    
    # Load the facenet model
    facenet = tf.keras.models.load_model('keras-facenet/model/facenet_keras.h5')
    
    # Freeze facenet from training
    facenet.trainable = False
    
    # Classifier design
    layer2 = tf.keras.layers.Flatten()
    layer3 = tf.keras.layers.Dense(1, activation='softmax',
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
    classifier.fit(x=training_generator,
                             epochs=epochs,
                             validation_data=validation_generator,
                             callbacks = [checkpoint],
                             )
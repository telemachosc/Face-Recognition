# -*- coding: utf-8 -*-
"""
Created on Mon May 24 17:21:02 2021

@author: telemachos
"""

import os
import platform
import time
import random

import tensorflow as tf
import json
import numpy as np
import PIL
from sklearn.preprocessing import LabelBinarizer, LabelEncoder

from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
os.environ["CUDA_VISIBLE_DEVICES"]="0"



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
                            f'Fine tune layers: {fnl}\n\n'))
        convert_file.write(('Loss and accuracy on train and validation set\n'))
        convert_file.write(json.dumps(h, indent=2))
        convert_file.write(('\n\n'+ 
                           '----------------------------------------------\n'))
        convert_file.write('On the test set the loss and accuracy was:\n')
        convert_file.write(f'loss: {test_set[0]}, \naccuracy: {test_set[1]}\n')
        convert_file.write(('---------------------------------------------\n'+
                            '\nTotal time for training and evaluating the '+
                            f'model was\n{time} seconds or {time/60} minutes'))

def load_paths(path: str , trainpct: float, valpct: float) -> dict:    
    x = []
    labels_list = []
    
    # Make two lists for input and labels
    for base, dirs, files in os.walk(path):
        if files:
            # Extend the paths of images in list x
            x.extend([base+os.sep+file for file in files])
            if platform.system()=='Windows':
                # Slice the name part of the path and extend it in labels list
                labels_list.extend([base.split('\\')[-2] for _ in range(len(files))])
            else:
                labels_list.extend([base.split('/')[-2] for _ in range(len(files))])
    

    le = LabelEncoder()
    
    le.fit_transform(labels_list)
    
    # One hot encode the labels
    encoder = LabelBinarizer()
    
    labels_list = encoder.fit_transform(labels_list).tolist()
    
    # Make dictionaries for x and y with correct names
    labels = {inpt: label for inpt, label in zip(x, labels_list)}
    
    # Shuffle the x
    random.shuffle(x)
    
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
        y = np.empty((self.batch_size, self.n_classes), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            face_pixels = self.extract_face(ID)

            face_pixels = face_pixels.astype('float32')
            # standardize pixel values across channels (global)
            mean, std = face_pixels.mean(), face_pixels.std()
            # Store sample
            X[i] = (face_pixels - mean) / std
            
            # Store class
            y[i]= self.labels[ID]
            # y[i] = self.labels[ID] == self.n_classes

        return X, y

def callbacks():
    # Define callbacks
    checkpoint_path = 'checkpoints'
    
    # save best model after each epoch
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_freq='epoch',
        save_weights_only=False,
        save_best_only=True,
        verbose=1)
    
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                  patience=2,
                                                  restore_best_weights=True)
    
    # lr_sched = step_decay_schedule(initial_lr=1e-4, 
    #                                decay_factor=0.75,
    #                                step_size=2)
    
    terminate = tf.keras.callbacks.TerminateOnNaN()
    
    return [checkpoint, early_stop, terminate]

# class DataGenerator(tf.keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, labels, batch_size=32, dim=(160,160), 
#                  n_channels=3, n_classes=10, shuffle=True):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.on_epoch_end()

#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return int(np.floor(len(self.list_IDs) / self.batch_size))

#     def __getitem__(self, index):
#         'Generate one batch of data'
#         # Generate indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         # Find list of IDs
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]

#         # Generate data
#         X, y = self.__data_generation(list_IDs_temp)
#         # X, y = np.random.uniform(0, 1, (32,160,160,3)), np.random.uniform(0, 1, 32)

#         return X, y

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)
            
#     def extract_face(self, filename):
#         # load image from file
#         image = Image.open(filename)
#         # convert to RGB, if needed
#         image = image.convert('RGB')
#         # convert to array
#         pixels = np.asarray(image)
#         # create the detector, using default weights
#         s = time.time()
#         detector = MTCNN()
#         # detect faces in the image
#         results = detector.detect_faces(pixels)
#         print(time.time()-s)
#         if results != []:
#             # extract the bounding box from the first face
#             x1, y1, width, height = results[0]['box']
#             # bug fix
#             x1, y1 = abs(x1), abs(y1)
#             x2, y2 = x1 + width, y1 + height
#             # extract the face
#             face = pixels[y1:y2, x1:x2]
#             # resize pixels to the model size
#             image = Image.fromarray(face)
#             image = image.resize(self.dim)
#             face_array = np.asarray(image)
#             return face_array
#         else:
#             return None

#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size), dtype=int)

#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             face_pixels = self.extract_face(ID)
#             if isinstance(face_pixels, np.ndarray):
#                 face_pixels = face_pixels.astype('float32')
#                 # standardize pixel values across channels (global)
#                 mean, std = face_pixels.mean(), face_pixels.std()
#                 # Store sample
#                 X[i,] = (face_pixels - mean) / std
                
#                 # X[i,] = np.load('data/' + ID + '.npy')
    
#                 # Store class
#                 y[i] = int(self.labels[ID] == self.n_classes)

#         return X, y





if __name__=="__main__":
    path = {'local': "data/youtube/aligned_images_DB",
            'server': '/data/tchatz/you_tube_processed'
            }
    # Remake path to make it windows compatible
    if platform.system()=='Windows':
        path = os.path.join(*path['local'].split('/'))
    else: 
        path = path['server']
        
    partition, labels = load_paths(path, 0.8, 0.1)
    
    num_classes = 1595
    
    params = {
        'dim': (299,299),
        'batch_size': 64,
        'n_classes': num_classes,
        'n_channels': 3,
        'shuffle': True
        }
    
    checkpoint_path = 'checkpoints'
    
    
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['val'], labels, **params)
    test_generator = DataGenerator(partition['test'], labels, **params)
    
    
    # Model hyperparameters
    EPOCHS = [20]
    LRs = [1e-04]
    FINE_TUNE_LAYERS = [10]
    
    # History filename
    hfname = 'youtube_011'
    
    
    def train_model(epochs, lr, fine_tune_layers, params):
        
        # Load the facenet model
        # facenet = tf.keras.models.load_model('keras-facenet/model/facenet_keras.h5')
        img_size = (*params['dim'], params['n_channels'])
        base_model = InceptionResNetV2(include_top=False, 
                                       weights='imagenet',
                                       input_shape=(299,299,3))
        
        # Freeze facenet from training
        base_model.trainable = False
        
        # Cut the last n layers of the base model
        # base_model = tf.keras.models.Model(base_model.input, base_model.layers[-3].output)
        
        # for layer in facenet.layers:
        #    layer.trainable = False
    
        if fine_tune_layers > 0:
            for layer in base_model.layers[-fine_tune_layers:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
    
        # for layer in facenet.layers[-fine_tune_layers:]:
        #     print(layer, layer.trainable)
             
        # Classifier design
        # input_layer = tf.keras.layers.Input(shape=img_size)
        
        # model_input = tf.keras.applications.inception_resnet_v2.preprocess_input(input_layer)
        # x = base_model(input_layer)
        # x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        # classifier = tf.keras.Model(input_layer, output)
        layers = {
            0: tf.keras.Input(shape=(299,299,3)),
            # 2: tf.keras.layers.Dense(units=256, activation='relu'),
            # 3: tf.keras.layers.Dropout(.5),
            # 1: tf.keras.layers.Flatten(),
            # 4: tf.keras.layers.Dense(1752, activation='relu'),
            5: tf.keras.layers.GlobalAveragePooling2D(),
            6: tf.keras.layers.Dense(num_classes, activation='softmax')
            }
        
        classifier = tf.keras.Sequential([base_model, *layers.values()])
        
        lm = { 'sparse': [tf.keras.losses.SparseCategoricalCrossentropy(),
                          'sparse_categorical_accuracy'],
              'dense': ['categorical_crossentropy',
                        'accuracy']
            }
        
        losme = lm['dense']
        classifier.compile(optimizer=tf.keras.optimizers.Adam(lr=lr),
                            loss=losme[0],
                            metrics=losme[1])
                                    # tf.keras.metrics.Precision(),
                                    # tf.keras.metrics.Recall()])
        
        
        s = time.time()
        
        # Train model on dataset
        h = classifier.fit(x=training_generator,
                           epochs=epochs,
                           validation_data=validation_generator,
                           callbacks = [*callbacks()])
        
        # Evaluate network on test set
        ev = classifier.evaluate(test_generator)
        
        dt = time.time() - s
        # write h.history and test set metrics to file
        write_history(hfname, h.history, ev, dt, epoch, lr, fnl)
        

    session_num = 0
    
    for epoch in EPOCHS:
        for lr in LRs:
            for fnl in FINE_TUNE_LAYERS:
                print(f'--- Starting trial: run-{session_num}')
                print((f'Epochs: {epoch}, Learning rate: {lr}, '+
                       f'Fine tune layers: {fnl}'))
                train_model(epoch, lr, fnl, params)
                session_num+=1
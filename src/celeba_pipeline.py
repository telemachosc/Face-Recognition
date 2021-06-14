# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:08:01 2021

@author: telemachos
"""
import os
import pathlib
import time

import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from src.generate_data import DataGenerator
from src.misc import write_history, step_decay_schedule


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
    
    lr_sched = step_decay_schedule(initial_lr=1e-4, 
                                   decay_factor=0.75,
                                   step_size=2)
    
    terminate = tf.keras.callbacks.TerminateOnNaN()
    
    return [checkpoint, early_stop, terminate]


class my_callback(tf.keras.callbacks.Callback):
   def on_train_batch_end(self, batch, logs=None):
       keys = list(logs.keys())
       print("...Training: end of batch {}; got log keys: {}".format(batch, logs))
    
# class DataGenerator(tf.keras.utils.Sequence):
#     'Generates data for Keras'
#     def __init__(self, list_IDs, labels, batch_size=32, dim=(160,160), 
#                  n_channels=3, n_classes=10, shuffle=True, df=[]):
#         'Initialization'
#         self.dim = dim
#         self.batch_size = batch_size
#         self.labels = labels
#         self.list_IDs = list_IDs
#         self.n_channels = n_channels
#         self.n_classes = n_classes
#         self.shuffle = shuffle
#         self.df = df
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
#         image = PIL.Image.open(filename)
#         # convert to RGB, if needed
#         image = image.convert('RGB')
#         # convert to array
#         # pixels = np.asarray(image)
        
#         # # extract the bounding box from the first face
#         # x1, y1, width, height = self.df.loc[self.df['path']==filename].values[0][2:6]
#         # # bug fix
#         # x1, y1 = abs(x1), abs(y1)
#         # x2, y2 = x1 + width, y1 + height
#         # # extract the face
#         # face = pixels[y1:y2, x1:x2]
#         # resize pixels to the model size
#         # image = PIL.Image.fromarray(face)
#         image = image.resize(self.dim)
#         return np.asarray(image)
         
       

#     def __data_generation(self, list_IDs_temp):
#         'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         # Initialization
#         X = np.empty((self.batch_size, *self.dim, self.n_channels))
#         y = np.empty((self.batch_size), dtype=int)

#         # Generate data
#         for i, ID in enumerate(list_IDs_temp):
#             face_pixels = self.extract_face(ID)

#             face_pixels = face_pixels.astype('float32')
#             # standardize pixel values across channels (global)
#             mean, std = face_pixels.mean(), face_pixels.std()
#             # Store sample
#             X[i] = (face_pixels - mean) / std
            
#             # Store class
#             y[i]= self.labels[ID] - 1
#             # y[i] = self.labels[ID] == self.n_classes

#         return X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes,
#                                                 dtype='float32')



if __name__=="__main__":
    
    path = 'data/CelebA'
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
    
    full_df = full_df.sample(frac=1).reset_index(drop=True)
    
    train_list = full_df.path[:int(len(full_df)*.8)].to_list()
    tmp_list = full_df.path[int(len(full_df)*.8):].to_list()
    val_list = tmp_list[:int(len(tmp_list)*.1)]
    test_list = tmp_list[int(len(tmp_list)*.1):]

    # Construct the dictionaries needed for the Data generator
    # train_list = list(full_df.groupby(['partition']).get_group(0).path)
    # val_list = list(full_df.groupby(['partition']).get_group(1).path)
    # test_list = list(full_df.groupby(['partition']).get_group(2).path)
    
    partition = {"train": train_list,
                 "val": val_list,
                 "test": test_list}
    
    # Construct the labels
    labels = {path: label for path, label in zip(full_df.path, full_df.identity)}
    
    # Calculate number of classes
    num_classes = len(set(val for val in full_df['identity']))
    
    params = {
        'dim': (299, 299),
        'batch_size': 32,
        'n_classes': num_classes,
        'n_channels': 3,
        'shuffle': True,
        'df': full_df,
        'norm_type': 'standardization'
        }
    
    training_generator = DataGenerator(partition['train'], labels, **params)
    validation_generator = DataGenerator(partition['val'], labels, **params)
    test_generator = DataGenerator(partition['test'], labels, **params)
    
    # Model hyperparameters
    EPOCHS = [20]
    LRs = [1e-04]
    FINE_TUNE_LAYERS = [10]
    # EPOCHS = [10,20, 30, 40]
    # LRs = [1e-03, 5e-4, 1e-04]
    # FINE_TUNE_LAYERS = [10, 20, 30, 40, 50]
    
    # History filename
    hfname = 'celeba_004'
    
    def train_model(epochs, lr, fine_tune_layers, params):
        
        # facenet model path
        # facenet_path = 'keras-facenet/model/facenet_keras.h5'
        img_size = (*params['dim'], params['n_channels'])
        
        
        base_model = InceptionResNetV2(include_top=False, 
                                       weights='imagenet',
                                       input_shape=img_size)
        
         # Freeze base model from training
        base_model.trainable = False
        
        # for layer in facenet.layers:
        #    layer.trainable = False
    
        if fine_tune_layers > 0:
            for layer in base_model.layers[-fine_tune_layers:]:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
        
        layer =  tf.keras.layers.GlobalAveragePooling2D()(base_model.layers[-1].output)
        
        output = tf.keras.layers.Dense(num_classes, activation='softmax')(layer)
        
        classifier = tf.keras.Model(base_model.inputs, output)
        
       
    

        # Classifier design
        
        # input_layer = tf.keras.layers.Input(shape=img_size)
        
        # model_input = preprocess_input(input_layer)
        # x = base_model(model_input)
        # x = tf.keras.layers.GlobalAveragePooling2D()(x)
        # output = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        # classifier = tf.keras.Model(input_layer, output)
        
        # layers = {
        #     0: tf.keras.Input(shape=(299,299,3)),
        #     # 2: tf.keras.layers.Dense(units=256, activation='relu'),
        #     # 3: tf.keras.layers.Dropout(.5),
        #     # 1: tf.keras.layers.Flatten(),
        #     # 4: tf.keras.layers.Dense(1752, activation='relu'),
        #     5: tf.keras.layers.GlobalAveragePooling2D(),
        #     6: tf.keras.layers.Dense(num_classes, activation='softmax')
        #     }
        
        # classifier = tf.keras.Sequential([base_model, *layers.values()])
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

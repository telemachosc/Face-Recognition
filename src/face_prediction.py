# -*- coding: utf-8 -*-
"""
Created on Thu May  6 14:29:38 2021

@author: telemachos
"""

# develop a classifier for the 5 Celebrity Faces Dataset
from random import choice
from numpy import load
from numpy import expand_dims
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt


# load faces
data = load('data/datasets/european-dataset.npz')
testX_faces = data['arr_2']
# load face embeddings
data = load('data/datasets/european-faces-embeddings.npz')
trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']

# normalize input vectors
in_encoder = Normalizer(norm='l2')
trainX = in_encoder.transform(trainX)
testX = in_encoder.transform(testX)

# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy = out_encoder.transform(trainy)
testy = out_encoder.transform(testy)

# fit model
model = SVC(kernel='linear', probability=True)
model.fit(trainX, trainy)

# test model on all examples from the test dataset
res = []
for image in range(testX.shape[0]):
   
    random_face_pixels = testX_faces[image]
    random_face_emb = testX[image]
    random_face_class = testy[image]
    random_face_name = out_encoder.inverse_transform([random_face_class])
    # prediction for the face
    samples = expand_dims(random_face_emb, axis=0)
    yhat_class = model.predict(samples)
    yhat_prob = model.predict_proba(samples)
    
    # get name
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_names = out_encoder.inverse_transform(yhat_class)
    res.append([predict_names[0], random_face_name[0], class_probability])
    print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
    print('Expected: %s' % random_face_name[0])
    # plot for fun
    plt.imshow(random_face_pixels)
    title = '%s (%.3f)' % (predict_names[0], class_probability)
    plt.title(title)
    plt.show()

import pandas as pd

df = pd.DataFrame(res, columns=("Prediction", "Expectation", "Probabilty of prediction"))
df['Probabilty of prediction'].mean()
#%%
df.to_excel('results.xlsx')

# original code
# test model on a random example from the test dataset
# selection = choice([i for i in range(testX.shape[0])])
# random_face_pixels = testX_faces[selection]
# random_face_emb = testX[selection]
# random_face_class = testy[selection]
# random_face_name = out_encoder.inverse_transform([random_face_class])
# # prediction for the face
# samples = expand_dims(random_face_emb, axis=0)
# yhat_class = model.predict(samples)
# yhat_prob = model.predict_proba(samples)

# # get name
# class_index = yhat_class[0]
# class_probability = yhat_prob[0,class_index] * 100
# predict_names = out_encoder.inverse_transform(yhat_class)
# print('Predicted: %s (%.3f)' % (predict_names[0], class_probability))
# print('Expected: %s' % random_face_name[0])
# # plot for fun
# plt.imshow(random_face_pixels)
# title = '%s (%.3f)' % (predict_names[0], class_probability)
# plt.title(title)
# plt.show()
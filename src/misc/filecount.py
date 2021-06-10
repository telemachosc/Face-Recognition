# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:59:54 2021

@author: telemachos
"""
import os

APP_FOLDER = "data/youtube/aligned_images_DB"

totalFiles = 0
totalDir = 0

for base, dirs, files in os.walk(APP_FOLDER):
    # print('Searching in : ',base, dirs, files)
    for directories in dirs:
        totalDir += 1
    for Files in files:
        totalFiles += 1


print('Total number of files',totalFiles)
print('Total Number of directories',totalDir)
print('Total:',(totalDir + totalFiles))
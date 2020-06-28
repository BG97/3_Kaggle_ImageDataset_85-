# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 17:07:06 2020

@author: benny
"""
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D
#from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.utils import np_utils, generic_utils, to_categorical
from sklearn.utils import shuffle
import keras
import sys
#from keras import regularizers
#from keras.regularizers import l2
import numpy as np
from keras import backend as K
#K.set_image_dim_ordering('th')
from keras.layers import Input
import os
from PIL import Image
from skimage.transform import resize
from keras.applications.vgg16 import VGG16
from keras.layers import GlobalAveragePooling2D
from keras import optimizers
from keras.models import load_model
nb_classes = 5
img_channels = 3
img_rows = 112
img_cols = 112

if len(sys.argv)>1:
    test = sys.argv[1]
else:
    test = 'flowers'
#img_folders = os.listdir(test)

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
        test,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        test,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',
        subset='validation')


model = load_model(sys.argv[2])



def score():
    score = model.evaluate_generator(validation_generator)
    print('Test error: ',(1-score[1])*100)
    print('Accuracy: ',score[1]*100)

score()

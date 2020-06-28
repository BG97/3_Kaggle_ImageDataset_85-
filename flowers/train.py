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
from keras.preprocessing.image import ImageDataGenerator

nb_classes = 5
img_channels = 3
img_rows = 112
img_cols = 112

if len(sys.argv)>1:
    train = sys.argv[1]
else:
    train = 'flowers'
#img_folders = os.listdir(train)
#data = []
#labels = []

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        train,
        target_size=(100, 100),
        batch_size=32,
        class_mode='categorical',
        subset='validation')


image_input = Input(shape=(100,100,3))

#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
#print('Y_train shape:', Y_train.shape)


model = VGG16(include_top=False, weights='imagenet', input_tensor=image_input, pooling=None,classes=nb_classes)

for layer in model.layers[:7]:
    layer.trainable = False


output = model.output
output = Flatten()(output)
output = Dense(512, activation="relu")(output)
output = Dropout(0.5)(output)
output = Dense(1024,activation="relu")(output)
output = Dropout(0.5)(output)

output = Dense(1024,activation="relu")(output)

predictions = Dense(nb_classes, activation="softmax")(output)

 
model_final = Model(input = model.input, output = predictions)
#keras.utils.multi_gpu_model(model, gpus=2, cpu_merge=False, cpu_relocation=False)
#opt = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.99, decay=1e-6)# best one
opt = optimizers.SGD(lr=0.001,momentum=0.9)

model_final.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])



def train():
    model_final.fit_generator(train_generator,
              steps_per_epoch=32,
              epochs=100,
              validation_data=validation_generator)
    if len(sys.argv)>1:
        model_final.save(sys.argv[2])
    else:
        model_final.save('model.h5')
train()

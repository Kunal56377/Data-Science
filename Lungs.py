# -*- coding: utf-8 -*-
"""
Created on Mon May 25 10:56:36 2020

@author: kmugalkh
"""

from keras.layers import Input,Dense,Flatten,Lambda
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt



IMAGE_SIZE = [224, 224]

train_path = 'Datasets/train'
test_path = 'Datasets/test'

#add preprocessing layer to the front of VGG
vgg = VGG16(input_shape=IMAGE_SIZE+[3],weights='imagenet',include_top=False)

#Dont train the Existing Weights
for layer in vgg.layers:
    layer.trainable = False

# useful for getting number of classes
folders = glob('chest_xray/Datasets/train/*')

x = Flatten()(vgg.output)

prediction = Dense(len(folders),activation='softmax')(x)

model = Model(input=vgg.input,outputs = prediction)

model.summary()

model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range=0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('chest_xray/Datasets/train',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('chest_xray/Datasets/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=5,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)
    
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')


model.save('model_vgg19.h5')


# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 21:25:27 2021

@author: renzo
"""

# Redes Neuronales Convolucionales

# 1) Construir el modelo de CNN

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Iniciar la CNN
classifier = Sequential()

# Paso 1 - Convolución
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), input_shape=(64,64,3), activation='relu'))

# Paso 2 - Max Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))

## Agregando una segunda capa
classifier.add(Convolution2D(filters=32, kernel_size=(3,3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Paso 3 - Flattening
classifier.add(Flatten())

# Paso 4 - Full Conection
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'sigmoid'))   # Clasificación binaria perro-gato

# Compilar la CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) 

# 2) Ajustar la CNN a las imágenes a entrenar

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_dataset = train_datagen.flow_from_directory('dataset/training_set',
                                                     target_size=(64, 64),
                                                     batch_size=32,
                                                     class_mode='binary')

testing_dataset = test_datagen.flow_from_directory('dataset/test_set',
                                                   target_size=(64, 64),
                                                   batch_size=32,
                                                   class_mode='binary')

classifier.fit_generator(training_dataset,
                         steps_per_epoch=125,   # len(train)/batch_size -> 4000/32
                         epochs=25,
                         validation_data=testing_dataset,
                         validation_steps=2000)

'''
classifier.fit_generator(training_dataset,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=testing_dataset,
                        validation_steps=2000)
'''

### https://stackoverflow.com/questions/59864408/tensorflowyour-input-ran-out-of-data
### IMPORTS ###

import matplotlib.pyplot as plt
import cv2
import os
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import mlflow
import sys


from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Activation, Dropout, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

kernel1 = int(sys.argv[1]) if len(sys.argv) > 1 else 3
kernel2 = int(sys.argv[2]) if len(sys.argv) > 2 else 3
kernel3 = int(sys.argv[3]) if len(sys.argv) > 3 else 3
epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 10


TRAIN_BASE_DIRECTORY = "../data/test/train"
TEST_BASE_DIRECTORY = "../data/test/test"


image_data_generator = ImageDataGenerator(validation_split=0.5)


TRAIN_IMAGE_SIZE = 32
TRAIN_BATCH_SIZE = 64

train_generator = image_data_generator.flow_from_directory(
    TRAIN_BASE_DIRECTORY,
    target_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE),
    batch_size=TRAIN_BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=5)

validation_generator = image_data_generator.flow_from_directory(
    TEST_BASE_DIRECTORY,
    target_size=(TRAIN_IMAGE_SIZE, TRAIN_IMAGE_SIZE),
    batch_size=TRAIN_BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    seed=5)

      
model = Sequential()

model.add(Conv2D(32, kernel_size=kernel1, activation='relu', padding='same', input_shape=(32,32,3))) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=kernel2, padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))

model.add(Conv2D(128, kernel_size=kernel3, padding='same', activation='relu'))

#Fin obligatoire
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

training = model.fit_generator(train_generator, epochs=epochs, validation_data=validation_generator, shuffle=False)

with mlflow.start_run():
    mlflow.log_param("T kernel 1conv", kernel1)
    mlflow.log_param("T kernel 2conv", kernel2)
    mlflow.log_param("T kernel 3conv", kernel3)
    mlflow.log_param("Epochs", epochs)
    mlflow.log_metric("acc", training.history["accuracy"][-1])
    mlflow.log_metric("val acc", training.history["val_accuracy"][-1])

    mlflow.keras.log_model(model, "model")

import os, glob, gc
import pandas as pd
import cv2
import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from collections import Counter
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, Dropout, Activation 
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adamax
from keras import backend as K
from keras.models import load_model

# Load video
def load_video_as_array(videopath):
    cap = cv2.VideoCapture(videopath)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    frames = np.asarray(frames)
    x = np.mean(frames, axis=0)  # compute the mean of frames over time
    return x

# Preprocessing data
def pre_data(X, y):
    # split and convert data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train /= 255.0
    X_val /= 255.0
    y_train = np_utils.to_categorical(y_train)
    y_val = np_utils.to_categorical(y_val)
    return X_train, X_val, y_train, y_val

# Activation Swish
def swish(x):
    return x * K.sigmoid(x)

# Build network
def build_model():
    # frames.shape = (frames width, hight, channels)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', data_format='channels_last', 
                     activation=swish, input_shape=(180, 320, 3)))
    model.add(Conv2D(32, (3, 3), activation=swish))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same', activation=swish))
    model.add(Conv2D(64, (3, 3), activation=swish))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation=swish))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model


if __name__ == "__main__":
    # load csv and videos on train set
    print("Loading videos ...")
    X = []
    y = []
    with open('train_list.csv') as f:
        for i, line in enumerate(f):
            #if i >= 300:
                #break
            filename, label = line.strip().split(",")
            X.append(load_video_as_array(os.path.join('train', filename)))
            y.append(int(label))
    X = np.asarray(X).astype("float16")
    y = np.asarray(y).astype("int16")
    X_train, X_val, y_train, y_val = pre_data(X, y)
    del X
    del y
    gc.collect()
    print(X_train.shape)

    # Learning
    model = build_model()
    model.compile(loss='binary_crossentropy', optimizer=Adamax(), metrics=['acc'])
    model.fit(X_train, y_train, epochs=10, batch_size=64, 
            validation_data=(X_val, y_val), verbose=1, shuffle=True)
    
    # export model
    model.save('my_model.h5')


from keras.layers import Conv2D, MaxPooling2D, Permute, Flatten, Dense, BatchNormalization, Activation, Dropout, Input
from keras.models import Sequential


import keras
import numpy as np
import sys

import utils

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

MATLAB_FILE = sys.argv[1]
MODEL_FILE = sys.argv[2]

def construct_model(windowSize, numberChannels, lr=0.001):
	model = Sequential([
		Input(shape=(windowSize,numberChannels,1)),
		Permute((3,2,1)),
		# layer1
		Conv2D(16, kernel_size=(numberChannels, 1), padding='valid', strides=(1, 1), data_format='channels_first', activation='relu'),
		BatchNormalization(axis=1, scale=False, center=False),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'),
		# layer2
		Conv2D(8,kernel_size=(1, 64),data_format='channels_first',padding='same'),
		BatchNormalization(axis=1,scale=False, center=False),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2),strides=(2, 2),padding='same'),
		Dropout(0.5),
		# layer3
		Conv2D(4,kernel_size=(5, 5),data_format='channels_first',padding='same'),
		BatchNormalization(axis=1,scale=False,center=False),
		Activation('relu'),
		MaxPooling2D(pool_size=(2, 2), data_format='channels_first',padding='same'),
		Dropout(0.5),
		# layer4
		Flatten(),
		Dense(1024, activation='relu'),
		Dropout(0.5),
		# layer5
		Dense(2, activation='softmax'),
	])
	
	model.summary()

	print("Created model, creating optimizer")

	adam = keras.optimizers.Adam(learning_rate=lr)

	print("Optimizer created, compiling model")

	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

	return model

print("Loading dataset")

(X, V, y), _, stimulation_time, fs, fr = utils.load_nagelspuler_dataset(MATLAB_FILE)

print("Splitting data to windows")

X = utils.split_data_to_windows(X)

# Data has to 3 dimensional, so we lose the information about the different trials.
X = X.reshape(X.shape[0] * X.shape[1], X.shape[2], X.shape[3])

# Pick all the of the target stimuli, lose the last 150 samples and flatten.
V = V[np.arange(V.shape[0]), y, :-150].flatten()

# Turn every 0 into [1, 0] and every 1 into [0, 1]
V = np.column_stack((V, 1 - V))

print(y.shape, X.shape)

print("Data splitting done, constructing model")

model = construct_model(X.shape[1], X.shape[2])

model.fit(X, V, validation_split=0.2, batch_size=256, epochs=25, callbacks = [keras.callbacks.ModelCheckpoint(MODEL_FILE, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')])

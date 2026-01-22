
from keras.layers import Conv2D, MaxPooling2D, Permute, Flatten, Dense, BatchNormalization, Activation, Dropout, Input
from keras.models import Sequential

import keras
import tensorflow as tf
import numpy as np
import sys

from sys import getsizeof

import matplotlib.pyplot as plt

import utils

import tensorflow as tf

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
	
	adam = keras.optimizers.Adam(learning_rate=lr)

	model.compile(loss='binary_crossentropy', optimizer=adam, metrics=["accuracy"])

	return model

print("TensorFlow version:", tf.__version__)

subjects = 9

accuracy_subjects = np.zeros((subjects))
std_subjects = np.zeros((subjects))

for subject in range(subjects):
	data_file = sys.argv[1] + "/VP" + str(subject + 1) + ".mat"
	model_file = sys.argv[2] + "/EEG2Code" + str(subject + 1) + ".keras"

	print(data_file)
	print(model_file)

	print("Loading dataset")

	(X_train, V_train, y_train), (X_test, V_test, y_test), _, _, _, _ = utils.load_nagelspuler_dataset(data_file)

	# # Pick all the of the target stimuli, lose the last 150 samples and flatten.
	# V_train = V_train[np.arange(V_train.shape[0]), y_train, :-150].flatten()

	# # Turn every 0 into [1, 0] and every 1 into [0, 1]
	# V_train = np.column_stack((V_train, 1 - V_train))

	# print("Splitting data to windows")
	# X_train = utils.split_data_to_windows(X_train)

	# # Data has to 3 dimensional, so we lose the information about the different trials.
	# X_train = np.vstack(X_train)

	# model = construct_model(X_train.shape[1], X_train.shape[2])

	# model.fit(X_train, V_train, validation_split=0.2, batch_size=256, epochs=25, callbacks = [keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto')])

	model = keras.models.load_model(model_file)

	X_test = utils.split_data_to_windows(X_test)

	V_test = V_test[np.arange(V_test.shape[0]), y_test, :-150].flatten()

	n_folds = 8

	X_folds = np.split(X_test, n_folds)
	V_folds = np.split(V_test, n_folds)
	y_folds = np.split(y_test, n_folds)

	accuracy = np.zeros(n_folds)

	for fold, (X_tst, V_tst) in enumerate(zip(X_folds, V_folds)):
		print("Making predictions for fold", fold + 1)

		X_tst = X_tst.reshape(X_tst.shape[0] * X_tst.shape[1], X_tst.shape[2], X_tst.shape[3])

		predicted_y = model.predict(X_tst, batch_size=256)

		predicted_y = 1 - np.argmax(predicted_y, axis=1)

		accuracy[fold] = np.mean(predicted_y == V_tst)
		print("Accuracy: ", accuracy[fold])
	
	accuracy_subjects[subject] = accuracy.mean()
	std_subjects[subject] = accuracy.std()

fig, ax = plt.subplots(1, 1, figsize=(15, 8), sharex=True)
ax.bar(np.arange(subjects), accuracy_subjects, yerr=std_subjects)
ax.hlines(np.mean(accuracy_subjects), -.5, subjects - 0.5, linestyle='--', color="k", alpha=0.5)
ax.set_xlabel("subject")
ax.set_ylabel("accuracy")
ax.set_title(f"EEG2Code stimulus generation: average accuracy {accuracy_subjects.mean():.2f} & standard deviation {accuracy_subjects.std():.2f}")

plt.show()
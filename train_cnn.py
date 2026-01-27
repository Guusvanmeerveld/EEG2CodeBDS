
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

from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, f1_score

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

accuracy_subjects = np.zeros((2, subjects))
precision_subjects = np.zeros((2, subjects))
specificity_subjects = np.zeros((2, subjects))
recall_subjects = np.zeros((2, subjects))
f_score_subjects = np.zeros((2, subjects))

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
	precision = np.zeros(n_folds)
	specificity = np.zeros(n_folds)
	recall = np.zeros(n_folds)
	f_score = np.zeros(n_folds)

	for fold, (X_tst, true_V) in enumerate(zip(X_folds, V_folds)):
		print("Making predictions for fold", fold + 1)

		predicted_V = model.predict(np.vstack(X_tst), batch_size=256)

		predicted_V = 1 - np.argmax(predicted_V, axis=1)

		accuracy[fold] = accuracy_score(true_V, predicted_V)
		precision[fold] = precision_score(true_V, predicted_V)

		tn, fp, fn, tp = confusion_matrix(true_V, predicted_V).ravel()
		specificity[fold] = tn / (tn+fp)
		
		recall[fold] = recall_score(true_V, predicted_V)
		f_score[fold] = f1_score(true_V, predicted_V)

		print("Accuracy:", accuracy[fold], "Precision:", precision[fold], "Recall:", recall[fold], "F-score:", f_score[fold])
	
	accuracy_subjects[0, subject] = accuracy.mean()
	accuracy_subjects[1, subject] = accuracy.std()

	precision_subjects[0, subject] = precision.mean()
	precision_subjects[1, subject] = precision.std()

	specificity_subjects[0, subject] = specificity.mean()
	specificity_subjects[1, subject] = specificity.std()

	recall_subjects[0, subject] = recall.mean()
	recall_subjects[1, subject] = recall.std()

	f_score_subjects[0, subject] = f_score.mean()
	f_score_subjects[1, subject] = f_score.std()

fig, ax = plt.subplots(5, 1, figsize=(15, 12), sharex=True, constrained_layout=True)
ax[0].bar(np.arange(subjects), accuracy_subjects[0], yerr=accuracy_subjects[1])
ax[0].hlines(np.mean(accuracy_subjects[0]), -.5, subjects - 0.5, linestyle='--', color="k", alpha=0.5)
ax[0].set_ylim(0.5, 1)
ax[0].set_xlabel("subject")
ax[0].set_ylabel("accuracy")
ax[0].set_title(f"Average accuracy {accuracy_subjects[0].mean():.2f} with standard deviation {accuracy_subjects[0].std():.2f}")

ax[1].bar(np.arange(subjects), precision_subjects[0], yerr=precision_subjects[1])
ax[1].hlines(np.mean(precision_subjects[0]), -.5, subjects - 0.5, linestyle='--', color="k", alpha=0.5)
ax[1].set_ylim(0.5, 1)
ax[1].set_xticks(np.arange(subjects), np.arange(subjects) + 1)
ax[1].set_xlabel("subject")
ax[1].set_ylabel("precision")
ax[1].set_title(f"Average precision {precision_subjects[0].mean():.2f} with standard deviation {precision_subjects[0].std():.2f}")

ax[2].bar(np.arange(subjects), recall_subjects[0], yerr=recall_subjects[1])
ax[2].hlines(np.mean(recall_subjects[0]), -.5, subjects - 0.5, linestyle='--', color="k", alpha=0.5)
ax[2].set_ylim(0.5, 1)
ax[2].set_xticks(np.arange(subjects), np.arange(subjects) + 1)
ax[2].set_xlabel("subject")
ax[2].set_ylabel("recall")
ax[2].set_title(f"Average recall {recall_subjects[0].mean():.2f} with standard deviation {recall_subjects[0].std():.2f}")

ax[3].bar(np.arange(subjects), f_score_subjects[0], yerr=f_score_subjects[1])
ax[3].hlines(np.mean(f_score_subjects[0]), -.5, subjects - 0.5, linestyle='--', color="k", alpha=0.5)
ax[3].set_xticks(np.arange(subjects), np.arange(subjects) + 1)
ax[3].set_ylim(0.5, 1)
ax[3].set_xlabel("subject")
ax[3].set_ylabel("f-score")
ax[3].set_title(f"Average f-score {f_score_subjects[0].mean():.2f} with standard deviation {f_score_subjects[0].std():.2f}")

ax[4].bar(np.arange(subjects), specificity_subjects[0], yerr=specificity_subjects[1])
ax[4].hlines(np.mean(specificity_subjects[0]), -.5, subjects - 0.5, linestyle='--', color="k", alpha=0.5)
ax[4].set_xticks(np.arange(subjects), np.arange(subjects) + 1)
ax[4].set_ylim(0.5, 1)
ax[4].set_xlabel("subject")
ax[4].set_ylabel("specificity")
ax[4].set_title(f"Average specificity {specificity_subjects[0].mean():.2f} with standard deviation {specificity_subjects[0].std():.2f}")

plt.show()
import scipy.io as sio
import numpy as np
import keras

def load_offline_eeg2code_dataset(file_name):
	data = sio.loadmat(file_name)
	# delays = sio.loadmat(file_name)

	stimulation_time = 4

	X_train = np.array(data['train_data_x_' + str(stimulation_time) + 's'])
	y_train = np.array(data['train_data_y_' + str(stimulation_time) + 's'])
	X_test = np.array(data['test_data_x'])
	y_test = np.array(data['test_data_y'])
	fs = 600
	fr = 60

	V = {}

	test_trials = 32
	test_trials_repeat = 14

	V["test"] = np.tile(np.arange(test_trials), test_trials_repeat)
	V["train"] = np.arange(len(y_train))

	print("EEG2Code dataset:")
	print("| X (train): ", X_train.shape, "(trials x channels x samples)")  # EEG
	print("| y (train): ", y_train.shape, "(trials x targets x samples)")  # labels
	print("| X (test): ", X_test.shape, "(trials x channels x samples)")  # EEG
	print("| y (test): ", y_test.shape, "(trials x targets x samples)")  # labels
	print("| fs: ", fs, "Hz")  # sampling frequency
	print("| fr: ", fr, "Hz")  # presentation rate

	return (X_train, y_train, X_test, y_test, V)

def load_rcca_dataset():
	participant = "sub-01"
	experiment = "test_sync"
	run = "1"

	raw = mne.io.read_raw_gdf(participant + "_"+ experiment + "_" + run + ".gdf")
	mat_contents = sio.loadmat(participant + "_"+ experiment + "_" + run + ".mat")

	# Downsample to 360Hz

	# print(np.array(mat_contents["codes"]).shape)
	# print(np.array(mat_contents["durations"]).shape)

	events = mne.find_events(raw, initial_event=True)
	epochs = mne.Epochs(raw, events, preload=True)

	epochs.resample(360, npad='auto')

	X = epochs.get_data()
	V = np.array(mat_contents["codes"])
	y = np.array(mat_contents["labels"]).squeeze()

	fs = raw.info["sfreq"]
	fr = 120

	print("rCCA dataset:")
	print("| X (train): ", X_train.shape, "(trials x channels x samples)")  # EEG
	print("| y (train): ", y_train.shape, "(trials)")  # labels
	print("| X (test): ", X_test.shape, "(trials x channels x samples)")  # EEG
	print("| y (test): ", y_test.shape, "(trials)")  # labels
	print("| V: ", V.shape, "(classes, samples)")  # labels
	print("| fs: ", fs, "Hz")  # sampling frequency
	print("| fr: ", fr, "Hz")  # presentation rate

	return (X, y)

def split_data_to_windows(X, windowSize = 150):
	X = X.swapaxes(1, 2)

	X_windowed = np.lib.stride_tricks.sliding_window_view(X, windowSize, axis=1)

	X_windowed = X_windowed.swapaxes(2, 3)

	X_windowed = X_windowed[:, :-1, :, :]

	return X_windowed

import scipy.io as sio
import numpy as np
import keras

# import glob
# import os
# import h5py
# import mne
# import pyntbci


def load_nagelspuler_dataset(file_name):
	data = sio.loadmat(file_name)

	stimulation_time = {}

	stimulation_time["train"] = 5
	stimulation_time["test"] = 2

	X_train = np.array(data['train_data_x_' + str(stimulation_time["train"]) + 's'])
	V_train = np.array(data['train_data_y_' + str(stimulation_time["train"]) + 's'])
	X_test = np.array(data['test_data_x'])
	V_test = np.array(data['test_data_y'])
	fs = 600
	fr = 60

	test_trials = 32
	test_trials_repeat = 14

	y_test = np.tile(np.arange(test_trials), test_trials_repeat)
	y_train = np.arange(V_train.shape[0])

	print("EEG2Code dataset:")
	print("| X (train): ", X_train.shape, "(trials x channels x samples)")  # EEG
	print("| y (train): ", y_train.shape, "(trials x targets x samples)")  # labels
	print("| X (test): ", X_test.shape, "(trials x channels x samples)")  # EEG
	print("| y (test): ", y_test.shape, "(trials x targets x samples)")  # labels
	print("| fs: ", fs, "Hz")  # sampling frequency
	print("| fr: ", fr, "Hz")  # presentation rate

	return ((X_train, V_train, y_train), (X_test, V_test, y_test), stimulation_time, fs, fr)

def load_thielen_dataset(data_path):

	# Subjects to read and preprocess
	subjects = [f"sub-{1 + i:02d}" for i in range(1)]  # all participants

	# Configuration for the preprocessing
	fs = 2048  # target sampling frequency in Hz for resampling
	bandpass = [1.0, 65.0]  # bandpass filter cutoffs in Hz for spectral filtering
	notch = 50.0  # notch filter cutoff in Hz for spectral filtering
	tmin = 0  # trial onset in seconds for slicing
	tmax = 10.5  # trial end in seconds for slicing

	# %%
	# Reading and upsampling the stimulation sequences
	# ------------------------------------------------
	# The noise-tags used in this dataset were modulated Gold codes with a linear shift register of length 6 and
	# feedback-tap positions at [6, 1] and [6, 5, 2, 1]. They were flip-balanced optimized for the iPad screen, and a subset
	# of 20 codes was selected. For more details, see the original publication. In addition to reading the codes, the codes
	# are upsampled to the EEG sampling frequency.

	# The screen refresh rate
	FR = 120

	# Load codes
	V = sio.loadmat(os.path.join(data_path, "resources", "mgold_61_6521_flip_balanced_20.mat"))["codes"].T

	# Upsample codes from screen framerate to EEG sampling rate
	V = np.repeat(V, fs // FR, axis=1).astype("uint8")

	# %%
	# Reading and preprocessing the EEG data
	# --------------------------------------
	# The cell below performs the reading and preprocessing of the EEG data, given the configuration above. The experiment
	# consisted of 5 blocks during each of which 20 trials were recorded, one for each of the codes in random order. The
	# cell results into (1) the data `X` that is a matrix of k trials c channels, and m samples, (2) the ground-truth labels
	# `y` that is a vector of k trials, (3) the codes `V` that is a matrix of n classes and m samples, and (4) the sampling
	# frequency `fs`.
	#
	# Note, the labels in `y` refer to the index in `V` at which to find the target (i.e., attended) code for a particular
	# trial.
	#
	# The cell reads the data from the sourcedata folder in the dataset, and saves the processed data for each participant
	# in a derivatives folder with a folder structure identical to the sourcedata.

	# The experimental blocks
	BLOCKS = ["test_stop_1", "test_stop_2", "test_stop_3", "test_sync_2", "block_5"]

	# Loop over subjects
	for subject in subjects:

		epochs = []
		labels = []

		# Loop over blocks
		for block in BLOCKS:
			# Find gdf file
			folder = os.path.join(data_path, "sourcedata", "offline", subject, block, f"{subject}_*_{block}_main_eeg.gdf")
			listing = glob.glob(folder)
			assert len(listing) == 1, f"Found none or multiple files for {subject}_{block}, should be a single file!"
			fname = listing[0]

			# Read raw file
			raw = mne.io.read_raw_gdf(fname, stim_channel="status", preload=True, verbose=False)

			# Read events
			events = mne.find_events(raw, verbose=False)

			# Select only the start of a trial
			# N.B. Every 2.1 seconds a trigger was generated (15 times per trial, plus one 16th "leaking trigger")
			# N.B. This "leaking trigger" is not always present, so taking epoch[::16, :] won't work, unfortunately
			cond = np.logical_or(np.diff(events[:, 0]) < 1.8 * raw.info['sfreq'],
								np.diff(events[:, 0]) > 2.4 * raw.info['sfreq'])
			idx = np.concatenate(([0], 1 + np.where(cond)[0]))
			onsets = events[idx, :]

			# Visualize events
			# import matplotlib.pyplot as plt
			# fig, ax = plt.subplots(1, 1, figsize=(17, 3))
			# ax.scatter(events[:, 0] / raw.info['sfreq'], events[:, 2], marker=".")
			# ax.scatter(onsets[:, 0] / raw.info['sfreq'], onsets[:, 2], marker="x")

			# Spectral notch filter
			raw.notch_filter(freqs=np.arange(notch, raw.info['sfreq'] / 2, notch), verbose=False)

			# Spectral band-pass filter
			raw.filter(l_freq=bandpass[0], h_freq=bandpass[1], verbose=False)

			# Slice data to trials
			# N.B. 500 ms pre-trial is added which is removed later. This is added to catch filter artefacts from the
			# subsequent resample. The resample is done after slicing to maintain accurate marker timing.
			epo = mne.Epochs(raw, events=onsets, tmin=tmin - 0.5, tmax=tmax, baseline=None, picks="eeg", preload=True,
							verbose=False)

			# Downsample
			epo.resample(sfreq=fs, verbose=False)

			# Add to dataset
			epochs.append(epo)

			# Read labels
			# N.B. Minus 1 to convert Matlab counting from 1 to Python counting from 0
			f = h5py.File(os.path.join(data_path, "sourcedata", "offline", subject, block, "trainlabels.mat"), "r")
			labels.append(np.array(f["v"]).astype("uint8").flatten() - 1)

		# Extract data and concatenate runs
		X = mne.concatenate_epochs(epochs, verbose=False).get_data(tmin=tmin, tmax=tmax).astype("float32")
		y = np.array(labels).flatten().astype("uint8")

		# Save data
		np.savez(os.path.join(out_path, f"thielen2021_{subject}.npz"), X=X, y=y, V=V, fs=fs)

def split_data_to_windows(X, window_size = 150):
	X = X.swapaxes(1, 2)

	X_windowed = np.lib.stride_tricks.sliding_window_view(X, window_size, axis=1)

	X_windowed = X_windowed.swapaxes(2, 3)

	X_windowed = X_windowed[:, :-1, :, :]

	return X_windowed

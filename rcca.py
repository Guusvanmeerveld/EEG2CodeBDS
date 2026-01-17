import os

import numpy as np
import seaborn

import pyntbci

import sys
import utils

def calculate_bds_values(X, y, V, trial_time, n_classes, fs = 600, fr = 60):
    print("X", X.shape, "(trials x channels x samples)")  # EEG
    print("y", y.shape, "(trials)")  # labels
    print("V", V.shape, "(classes, samples)")  # codes
    print("fs", fs, "Hz")  # sampling frequency
    print("fr", fr, "Hz")  # presentation rate

    # Extract data dimensions
    n_trials, n_channels, n_samples = X.shape

    # Read cap file
    # capfile = os.path.join(path, "capfiles", "thielen8.loc")
    # with open(capfile, "r") as fid:
    #     channels = []
    #     for line in fid.readlines():
    #         channels.append(line.split("\t")[-1].strip())
    # print("Channels:", ", ".join(channels))

    # ##
    # Settings
    # --------
    # Some general settings for the following sections

    # Set trial duration
    inter_trial_time = 1.0  # ITI in seconds for computing ITR
    n_samples = int(trial_time * fs)

    # Setup rCCA
    encoding_length = 0.3  # seconds
    onset_event = True  # an event modeling the onset of a trial
    event = "refe"

    # Set size of increments of trials
    segment_time = 0.1  # seconds
    n_segments = int(trial_time / segment_time)

    # Set chronological cross-validation
    n_folds = 5
    folds = np.repeat(np.arange(n_folds), int(n_trials / n_folds))

    cr = 1.0

    # Fit classifier
    rcca = pyntbci.classifiers.rCCA(stimulus=V, fs=fs, event=event, encoding_length=encoding_length,
                                    onset_event=onset_event, score_metric="inner")

    bayes = pyntbci.stopping.BayesStopping(rcca, segment_time=segment_time, fs=fs, cr=cr, max_time=trial_time)
    bayes.fit(X, y)

    target_mean = bayes.alpha_ * bayes.b1_
    non_target_mean = bayes.alpha_ * bayes.b0_
    target_std = bayes.s1_
    non_target_std = bayes.s0_
    eta = bayes.eta_

    return target_mean, non_target_mean, target_std, non_target_std, eta


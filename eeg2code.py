from keras.models import load_model
import keras
import os
import sys
import math

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np

import utils

class EEG2CodeEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, model_file):
        self.model = load_model(model_file)

    def predict(self, X):
        return self.model.predict(X, batch_size=128)

def calculate_similarity_scores(X, V, y, segment_count, eeg2code_model):
    # eeg2code_model = EEG2CodeEstimator(model_file)

    trial_count, window_count, _, _ = X.shape

    segment_size = int(window_count / segment_count)

    # print(segment_size)

    target_scores = np.full((trial_count, segment_count), np.nan)
    non_target_scores = np.full((trial_count * (V.shape[1] - 1), segment_count), np.nan)

    for i, (trial, stimuli, label) in enumerate(zip(X, V, y)):
        print("Trial ", (i + 1))

        for j in range(segment_count):
            predicted_stimulus = eeg2code_model.predict(trial[:(segment_size * (j + 1))])

            actual_stimulus = stimuli[:, :(predicted_stimulus.shape[0])]

            predicted_stimulus = 1 - np.argmax(predicted_stimulus, axis=1)

            similarity_scores = np.logical_xor(predicted_stimulus, actual_stimulus).sum(axis=1)

            if ((j + 1) == segment_count):
                print(similarity_scores)

            target_scores[i, j] = similarity_scores[label]

            for k, non_target_score in enumerate(np.delete(similarity_scores, label)):
                non_target_scores[i * k + k, j] = non_target_score

    return target_scores, non_target_scores

def calculate_decision_boundary(target_mean, non_target_mean, target_std, non_target_std, n_classes, cr = 1.0):
    a = target_std ** 2 - non_target_std ** 2
    b = -2 * (target_std ** 2 * non_target_mean - non_target_std ** 2 * target_mean)
    c = -(target_std ** 2 * non_target_mean ** 2 + non_target_std ** 2 * target_mean ** 2) + 2 * non_target_std ** 2 * target_std ** 2 * np.log(non_target_std / (target_std * (n_classes - 1) * cr))

    eta = (-b + np.sqrt(np.clip(b ** 2 - 4 * a * c, 0, None))) / (2 * a)

    return eta

def calculate_bds_values(X, V, y, segment_count, window_size, model):
    # This function loses 1 window of data at the end of the trial because of how it functions.
    X = utils.split_data_to_windows(X, window_size=window_size)

    target_scores, non_target_scores = calculate_similarity_scores(X, V, y, segment_count, model)

    n_classes = V.shape[1]

    target_mean = np.nanmean(target_scores, axis=0)
    non_target_mean = np.nanmean(non_target_scores, axis=0)
    target_std = np.nanstd(target_scores, axis=0)
    non_target_std =  np.nanstd(non_target_scores, axis=0)

    size = target_mean.shape[0]

    normalizing_factor = non_target_mean

    # Normalize both distributions to the non target distribution.
    target_mean = target_mean - normalizing_factor
    non_target_mean = non_target_mean - normalizing_factor

    decision_boundary = calculate_decision_boundary(target_mean, non_target_mean, target_std, non_target_std, n_classes)

    return target_mean, non_target_mean, target_std, non_target_std, decision_boundary, normalizing_factor
    
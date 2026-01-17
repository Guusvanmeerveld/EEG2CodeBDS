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

def calculate_similarity_scores(X, y, V, model_file):
    eeg2code_model = EEG2CodeEstimator(model_file)

    trial_count, _, _, _ = X.shape

    group_count = 15

    target_scores = np.full((trial_count, group_count), np.nan)
    non_target_scores = np.full((trial_count * (y.shape[1] - 1), group_count), np.nan)

    # print(target_scores.shape, non_target_scores.shape)

    for i, (trial, stimuli, label) in enumerate(zip(X, y, V)):
        trial_split = np.array(np.split(trial, group_count))

        print("Trial ", (i + 1))

        if (i > 128):
            break

        for j in range(len(trial_split)):
            data_until_now = trial_split[:(j + 1)]

            data_flat = data_until_now.reshape(data_until_now.shape[0] * data_until_now.shape[1], data_until_now.shape[2], data_until_now.shape[3])

            predicted_y = eeg2code_model.predict(data_flat)

            actual_y = stimuli[:, :(len(predicted_y))]

            predicted_y = 1 - np.argmax(predicted_y, axis=1)

            similarity_scores = np.logical_xor(predicted_y, actual_y).sum(axis=1)

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

def calculate_bds_values(X, y, V, model_file = "models/EEG2Code.keras", windowed_data_dir = "data/eeg2code/windowed"):
    windowSize = 150

    X = utils.split_data_to_windows(X, windowSize=windowSize)

    target_scores, non_target_scores = calculate_similarity_scores(X, y, V, model_file)

    # print("Target scores shape: ", target_scores.shape)
    # print("Non target scores shape: ", non_target_scores.shape)

    n_classes = y.shape[1]

    target_mean = np.nanmean(target_scores, axis=0)
    non_target_mean = np.nanmean(non_target_scores, axis=0)
    target_std = np.nanstd(target_scores, axis=0)
    non_target_std =  np.nanstd(non_target_scores, axis=0)

    size = target_mean.shape[0]

    # Normalize both distributions to the non target distribution.
    target_mean = target_mean - non_target_mean
    non_target_mean = non_target_mean - non_target_mean

    decision_boundary = calculate_decision_boundary(target_mean, non_target_mean, target_std, non_target_std, n_classes)

    return target_mean, non_target_mean, target_std, non_target_std, decision_boundary
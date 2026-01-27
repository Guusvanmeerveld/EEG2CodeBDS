from keras.models import load_model
import keras
import os
import sys
import math

from sklearn.base import BaseEstimator, ClassifierMixin
from scipy import stats
import numpy as np

import utils

class EEG2CodeEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, model_file):
        self.model = load_model(model_file)

    def predict(self, X):
        return self.model.predict(X, batch_size=128)

def calculate_similarity_scores(X, V, y, segment_count, eeg2code_model):
    trial_count, window_count, _, _ = X.shape

    segment_size = int(window_count / segment_count)

    target_scores = np.full((trial_count, segment_count), np.nan)
    non_target_scores = np.full((trial_count * (V.shape[1] - 1), segment_count), np.nan)

    for i, (trial, stimuli, label) in enumerate(zip(X, V, y)):
        print("Trial ", (i + 1))

        for j in range(segment_count):
            predicted_stimulus = eeg2code_model.predict(trial[:(segment_size * (j + 1))])

            actual_stimulus = stimuli[:, :(predicted_stimulus.shape[0])]

            predicted_stimulus = np.argmax(predicted_stimulus, axis=1)

            similarity_scores = np.logical_xor(predicted_stimulus, actual_stimulus).sum(axis=1)

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

def calculate_bds_params(X, V, y, segment_count, window_size, model, cost_ratio):
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

    decision_boundary = calculate_decision_boundary(target_mean, non_target_mean, target_std, non_target_std, n_classes, cost_ratio)

    return target_mean, non_target_mean, target_std, non_target_std, decision_boundary, normalizing_factor
    
def bds_performance_test(subject, cost_ratio, segment_time):
    (X_train, V_train, y_train), (X_test, V_test, y_test), trial_time, inter_trial_time, fs, fr = utils.load_nagelspuler_dataset("data/nagelspuler/offline/VP" + str(subject + 1) + ".mat")

    eeg2code_model = EEG2CodeEstimator("models/EEG2Code" + str(subject + 1) + ".keras")

    n_test_segments = int(trial_time["test"] / segment_time)
    n_train_segments = int(trial_time["train"] / segment_time)

    n_folds = 8

    X_folds = np.split(X_test, n_folds)
    V_folds = np.split(V_test, n_folds)
    y_folds = np.split(y_test, n_folds)
    
    X_trn = X_folds.pop()
    V_trn = V_folds.pop()
    y_trn = y_folds.pop()

    # We use the last fold for calibrating BDS.
    n_folds -= 1

    window_size = 150

    segment_count = n_test_segments

    assert ((window_size / fs) % segment_time == 0)

    segment_count -= int((window_size / fs) / segment_time)

    print("Calibrating BDS for subject " + str(subject + 1) + " with cost ratio " + str(cost_ratio) + "...")

    target_mean, non_target_mean, target_std, non_target_std, decision_boundary, normalizing_factor = calculate_bds_params(X_trn, V_trn, y_trn, segment_count, window_size, eeg2code_model, cost_ratio)

    # plot_dist_param(non_target_mean, target_mean, non_target_std, target_std, decision_boundary,  np.linspace(0, segment_count * segment_time, num=target_mean.shape[0]), "EEG2Code")

    # plt.xlabel('stimulation time (s)')
    # plt.ylabel('similarity score')
    # plt.grid(True, alpha=0.4)
    # plt.legend(loc='upper left')

    # plt.tight_layout()
    # plt.show()

    print("Testing BDS for subject " + str(subject + 1) + " with cost ratio " + str(cost_ratio) + "...")

    predicted_tst = np.zeros((n_folds, V_folds[0].shape[0]))
    dur_tst = np.zeros((n_folds, V_folds[0].shape[0]))

    for fold, (X_tst, V_tst, y_tst) in enumerate(zip(X_folds, V_folds, y_folds)):
        X_tst = utils.split_data_to_windows(X_tst, window_size=window_size)

        window_count = X_tst.shape[1]

        segment_size = int(window_count / segment_count)

        for i, (trial, stimuli, label) in enumerate(zip(X_tst, V_tst, y_tst)):
            # print("Trial ", (i + 1))

            for j in range(segment_count):
                predicted_stimulus = eeg2code_model.predict(trial[:(segment_size * (j + 1))])

                actual_stimulus = stimuli[:, :(predicted_stimulus.shape[0])]

                predicted_stimulus = np.argmax(predicted_stimulus, axis=1)

                similarity_scores = np.logical_xor(predicted_stimulus, actual_stimulus).sum(axis=1)

                # Normalize and also remove decision boundary, meaning anything higher than 0 passed the boundary
                passing_scores = similarity_scores - normalizing_factor[j] - decision_boundary[j]

                # If there are passing scores, we stop. If there were no passing scores, we still run this code but at the end of the trial.
                if (((passing_scores > 0).any()) or ((j + 1) == segment_count)):
                    dur_tst[fold, i] = (1 + j) * segment_time
                    # The highest value is the most likely to be the correct target, according to Amahdi et al (2023)
                    predicted_tst[fold, i] = np.argmax(passing_scores)
                    break
    
    dur_tst = dur_tst.mean(axis=1)

    n_classes = V_test.shape[1]

    return predicted_tst, dur_tst, n_classes

def calculate_bds_params(X, V, y, segment_count, window_size, model, cost_ratio):
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

    decision_boundary = calculate_decision_boundary(target_mean, non_target_mean, target_std, non_target_std, n_classes, cost_ratio)

    return target_mean, non_target_mean, target_std, non_target_std, decision_boundary, normalizing_factor
    
def return_zero_if_negative(value):
    return value if value >= 0 else 0

def async_performance_test(subject, p_value_threshold, max_sub_trial_length):
    _, (X_test, V_test, y_test), trial_time, inter_trial_time, fs, fr = utils.load_nagelspuler_dataset("data/nagelspuler/offline/VP" + str(subject + 1) + ".mat")

    eeg2code_model = EEG2CodeEstimator("models/EEG2Code" + str(subject + 1) + ".keras")

    print(fs)

    window_size = 150

    max_sub_trial_samples_count = max_sub_trial_length * fs 
    # Remove window size, since the windowing function also removes 1 window size.
    total_sample_count = trial_time["test"] * fs - window_size

    start_amount = 150
    step_size = 25

    n_folds = 8

    X_folds = np.split(X_test, n_folds)
    V_folds = np.split(V_test, n_folds)
    y_folds = np.split(y_test, n_folds)

    print("Testing Async stopping for subject:", subject + 1, " with p_value_threshold:", p_value_threshold)

    predicted_tst = np.zeros((n_folds, V_folds[0].shape[0]))
    dur_tst = np.zeros((n_folds, V_folds[0].shape[0]))

    for fold, (X_tst, V_tst, y_tst) in enumerate(zip(X_folds, V_folds, y_folds)):
        X_tst = utils.split_data_to_windows(X_tst, window_size=window_size)

        for i, (trial, stimuli, label) in enumerate(zip(X_tst, V_tst, y_tst)):
            print("Trial ", (i + 1))

            j = start_amount

            while (j < total_sample_count):
                # print(trial[(return_zero_if_negative(j - max_sub_trial_samples_count)):(j)].shape)
                # print(return_zero_if_negative(j - max_sub_trial_samples_count))
                # print(j)
                predicted_stimulus = eeg2code_model.predict(trial[(return_zero_if_negative(int(j - max_sub_trial_samples_count))):(j)])

                actual_stimulus = stimuli[:, (return_zero_if_negative(int(j - max_sub_trial_samples_count))):(j)]

                predicted_stimulus = np.argmax(predicted_stimulus, axis=1)

                p_values = np.zeros((actual_stimulus.shape[0]))

                for k, stim_true in enumerate(actual_stimulus):
                    # print(stim_true.shape, predicted_stimulus.shape)
                    correlation, _ = stats.pearsonr(predicted_stimulus, stim_true)

                    observations = predicted_stimulus.shape[0]

                    degrees_of_freedom = observations - 2

                    t = correlation * np.sqrt(degrees_of_freedom) / np.sqrt(1 - correlation**2)

                    p_values[k] = 2 * (1 - stats.t.cdf(np.abs(t), degrees_of_freedom))

                if (((p_values < p_value_threshold).any()) or ((j + step_size) >= total_sample_count)):
                    dur_tst[fold, i] = j / fs
                    predicted_tst[fold, i] = np.argmin(p_values)

                    print("Duration:", dur_tst[fold, i], "Predicted:", predicted_tst[fold, i], "Actual:", y_tst[i])
                    break

                j += step_size


    
    dur_tst = dur_tst.mean(axis=1)

    n_classes = V_test.shape[1]

    return predicted_tst, dur_tst, n_classes
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

import keras
import pyntbci

import eeg2code
import rcca

import utils

keras.config.disable_interactive_logging()

def plot_dist_param(b0_, b1_, s0_, s1_, eta_, time_intervals, label):
    mu_t = b1_
    std_t = s1_
    mu_nt = b0_
    std_nt = s0_
    decision_boundary = eta_
    time_intervals = time_intervals
    target_color = "#2f6fb3"   # blue
    nontarget_color = "#d172ee"  # magenta / pink
    # plot
    # target distribution (blue)
    plt.plot(time_intervals, mu_t, lw=2, label=(label + ' target mean'))
    plt.plot(time_intervals, mu_t + std_t, ls='--', lw=1)
    plt.plot(time_intervals, mu_t - std_t, ls='--', lw=1)
    plt.fill_between(time_intervals, mu_t - std_t, mu_t + std_t, color=target_color,
                    alpha=0.25)
    # non-target distribution (magenta)
    plt.plot(time_intervals, mu_nt, lw=2, label=(label + ' non-target mean'))
    plt.plot(time_intervals, mu_nt + std_nt, ls='--', lw=1)
    plt.plot(time_intervals, mu_nt - std_nt, ls='--', lw=1)
    plt.fill_between(time_intervals, mu_nt - std_nt, mu_nt + std_nt, color=nontarget_color,
                    alpha=0.25)
    
    # eta
    plt.plot(time_intervals, decision_boundary, label = (label + ' eta'))
   
    # y-axis for PDFs = similarity score axis
    y_min = min((mu_nt - 2*std_nt).min(), (mu_t - 2*std_t).min())
    y_max = max((mu_nt + 2*std_nt).max(), (mu_t + 2*std_t).max())
    eta = np.linspace(y_min, y_max, 400)
    # take the last time point's parameters
    mu_t_end,  std_t_end = mu_t[-1],  std_t[-1]
    mu_nt_end, std_nt_end = mu_nt[-1], std_nt[-1]
    pdf_t = norm.pdf(eta, mu_t_end,  std_t_end)
    pdf_nt = norm.pdf(eta, mu_nt_end, std_nt_end)
    # scale PDFs to a narrow band on the right
    x0 = time_intervals[-1] + 0.1          # start of PDF band
    width = 0.4                             # how "wide" the PDFs are
    pdf_t_x = x0 + width * pdf_t / pdf_t.max()
    pdf_nt_x = x0 + width * pdf_nt / pdf_nt.max()
    plt.plot(pdf_t_x,  eta, color=target_color)
    plt.plot(pdf_nt_x, eta, color=nontarget_color)
    plt.xlim(time_intervals[0], x0 + width)

(X_train, V_train, y_train), (X_test, V_test, y_test), trial_time, fs, fr = utils.load_nagelspuler_dataset("data/nagelspuler/offline/VP2.mat")

# (X_train, V_train, y_train), (X_test, V_test, y_test), trial_time, fr, fs = utils.load_thielen_dataset("data/thielen2015")

# 250 ms segments for BDS
segment_time = 0.125

n_test_segments = int(trial_time["test"] / segment_time)
n_train_segments = int(trial_time["train"] / segment_time)

# print("Calculating BDS values for EEG2Code")

# target_mean, non_target_mean, target_std, non_target_std, decision_boundary = eeg2code.calculate_bds_values(X_train, V_train, y_train, n_train_segments)

# plot_dist_param(non_target_mean, target_mean, non_target_std, target_std, decision_boundary,  np.linspace(0, trial_time["train"], num=target_mean.shape[0]), "EEG2Code")

print("Performing BDS test for EEG2Code")

def eeg2code_bds_test():
    eeg2code_model = eeg2code.EEG2CodeEstimator("models/EEG2Code.keras")

    X_tst, X_trn = np.split(X_test, 2)
    V_tst, V_trn = np.split(V_test, 2)
    y_tst, y_trn = np.split(y_test, 2)

    window_size = 150

    segment_count = n_test_segments

    assert ((window_size / fs) % segment_time == 0)

    segment_count -= int((window_size / fs) / segment_time)

    target_mean, non_target_mean, target_std, non_target_std, decision_boundary, normalizing_factor = eeg2code.calculate_bds_values(X_trn, V_trn, y_trn, segment_count, window_size, eeg2code_model)

    plot_dist_param(non_target_mean, target_mean, non_target_std, target_std, decision_boundary,  np.linspace(0, segment_count * segment_time, num=target_mean.shape[0]), "EEG2Code")

    plt.xlabel('stimulation time (s)')
    plt.ylabel('similarity score')
    plt.grid(True, alpha=0.4)
    plt.legend(loc='upper left')

    plt.tight_layout()
    plt.show()

    X_tst = utils.split_data_to_windows(X_tst, window_size=window_size)

    window_count = X_tst.shape[1]

    segment_size = int(window_count / segment_count)

    # print(segment_size)

    predicted_test = np.zeros((V_tst.shape[0]))
    dur_test = np.zeros((V_tst.shape[0]))

    for i, (trial, stimuli, label) in enumerate(zip(X_tst, V_tst, y_tst)):
        print("Trial ", (i + 1))

        for j in range(segment_count):
            predicted_stimulus = eeg2code_model.predict(trial[:(segment_size * (j + 1))])

            actual_stimulus = stimuli[:, :(predicted_stimulus.shape[0])]

            predicted_stimulus = 1 - np.argmax(predicted_stimulus, axis=1)

            similarity_scores = np.logical_xor(predicted_stimulus, actual_stimulus).sum(axis=1)

            # Normalize and also remove decision boundary, meaning anything higher than 0 passed the boundary
            passing_scores = similarity_scores - normalizing_factor[j] - decision_boundary[j]

            # If there are passing scores, we stop.
            if (((passing_scores > 0).any()) or ((j + 1) == segment_count)):
                dur_test[i] = (1 + j) * segment_time
                # The highest value is the most likely to be the correct target, according to Amahdi et al (2023)
                predicted_test[i] = np.argmax(passing_scores)
                print("Predicted stimulus: ", np.argmax(passing_scores), "Actual stimulus: ", y_tst[i], " At timepoint: ", (1 + j) * segment_time)
                break
        

    print("Accuracy: ", np.mean(predicted_test == y_tst))
    print("Duration: ", np.mean(dur_test))

    # Compute ITR
    # itr_bds0 = pyntbci.utilities.itr(n_classes, accuracy_bds0, duration_bds0 + inter_trial_time)

eeg2code_bds_test()
# print("Calculating BDS values for rCCA")

# V = y[np.arange(y.shape[0]), V]
# y = np.arange(V.shape[0])
# print(y.shape, V.shape)

# target_mean, non_target_mean, target_std, non_target_std, decision_boundary = rcca.calculate_bds_values(X, y, V, stimulation_time["train"], 32)

# plot_dist_param(non_target_mean, target_mean, non_target_std, target_std, decision_boundary,  np.linspace(0, stimulation_time["train"], num=target_mean.shape[0]), "RCCA")

plt.xlabel('stimulation time (s)')
plt.ylabel('similarity score')
plt.grid(True, alpha=0.4)
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
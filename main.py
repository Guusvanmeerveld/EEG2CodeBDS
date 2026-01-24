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

# 250 ms segments for BDS
segment_time = 0.125

print("Performing BDS test for EEG2Code")

subjects = 9

# for cost_ratio in np.logspace(-4, 4, num=9):
segment_count = 14

target_mean_sub = np.zeros((subjects, segment_count))
non_target_mean_sub = np.zeros((subjects, segment_count))
target_std_sub = np.zeros((subjects, segment_count))
non_target_std_sub = np.zeros((subjects, segment_count))
decision_boundary_sub = np.zeros((subjects, segment_count))

for subject in range(subjects):
    (X_train, V_train, y_train), (X_test, V_test, y_test), trial_time, inter_trial_time, fs, fr = utils.load_nagelspuler_dataset("data/nagelspuler/offline/VP" + str(subject + 1) + ".mat")

    eeg2code_model = eeg2code.EEG2CodeEstimator("models/EEG2Code" + str(subject + 1) + ".keras")

    window_size = 150

    target_mean, non_target_mean, target_std, non_target_std, decision_boundary, normalizing_factor = eeg2code.calculate_bds_params(X_test, V_test, y_test, segment_count, window_size, eeg2code_model, 1.0)

    target_mean_sub[subject] = target_mean
    non_target_mean_sub[subject] = non_target_mean
    target_std_sub[subject] = target_std
    non_target_std_sub[subject] = non_target_std
    decision_boundary_sub[subject] = decision_boundary

np.savez("data/bds_calibration_normalized.npz", target_mean=target_mean_sub, non_target_mean=non_target_mean_sub, target_std=target_std_sub, non_target_std=non_target_std_sub, decision_boundary=decision_boundary_sub)

# plot_dist_param(np.mean(non_target_mean_sub, axis=0), np.mean(target_mean_sub, axis=0), np.mean(non_target_std_sub, axis=0), np.mean(target_std_sub, axis=0), np.mean(decision_boundary_sub, axis=0),  np.linspace(0, segment_count, num=target_mean.shape[0]), "EEG2Code")


save_path = "data/bds_test/"

for cost_ratio in np.logspace(-4, 4, num=9):
    for subject in range(subjects):
        predicted, duration, n_classes = eeg2code.bds_performance_test(subject, cost_ratio, segment_time)

        np.save(save_path + "predicted-" + str(subject + 1) + "-" + str(cost_ratio) + ".npy", predicted)
        np.save(save_path + "duration-" + str(subject + 1) + "-" + str(cost_ratio) + ".npy", duration)
        np.save(save_path + "n_classes-" + str(subject + 1) + "-" + str(cost_ratio) + ".npy", n_classes)



# # Compute ITR
# itr_bds0 = pyntbci.utilities.itr(n_classes, accuracy_tst, dur_tst + inter_trial_time["test"])

# fig, ax = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
# ax[0].bar(np.arange(n_folds), accuracy_tst)
# ax[0].hlines(np.mean(accuracy_tst), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
# ax[1].bar(np.arange(n_folds), dur_tst)
# ax[1].hlines(np.mean(dur_tst), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
# ax[2].bar(np.arange(n_folds), itr_bds0)
# ax[2].hlines(np.mean(itr_bds0), -.5, n_folds - 0.5, linestyle='--', color="k", alpha=0.5)
# ax[2].set_xlabel("(test) fold")
# ax[0].set_ylabel("accuracy")
# ax[1].set_ylabel("duration [s]")
# ax[2].set_ylabel("itr [bits/min]")
# ax[0].set_title(f"BDS0 dynamic stopping: avg acc {accuracy_tst.mean():.2f} | " +
#                 f"avg dur {dur_tst.mean():.2f} | avg itr {itr_bds0.mean():.1f}")

# plt.show()
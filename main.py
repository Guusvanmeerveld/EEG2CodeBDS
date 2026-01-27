import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

import keras
import pyntbci
import math

import eeg2code
import utils

from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, f1_score

keras.config.disable_interactive_logging()

def plot_dist_param(b0_, b1_, s0_, s1_, eta_, time_intervals, label, ax):
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
    ax.plot(time_intervals, mu_t, lw=2, label=(label + ' target mean'))
    ax.plot(time_intervals, mu_t + std_t, ls='--', lw=1)
    ax.plot(time_intervals, mu_t - std_t, ls='--', lw=1)
    ax.fill_between(time_intervals, mu_t - std_t, mu_t + std_t, color=target_color,
                    alpha=0.25)
    # non-target distribution (magenta)
    ax.plot(time_intervals, mu_nt, lw=2, label=(label + ' non-target mean'))
    ax.plot(time_intervals, mu_nt + std_nt, ls='--', lw=1)
    ax.plot(time_intervals, mu_nt - std_nt, ls='--', lw=1)
    ax.fill_between(time_intervals, mu_nt - std_nt, mu_nt + std_nt, color=nontarget_color,
                    alpha=0.25)
    
    # eta
    ax.plot(time_intervals, decision_boundary, label = (label + ' eta'))
   
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
    ax.plot(pdf_t_x,  eta, color=target_color)
    ax.plot(pdf_nt_x, eta, color=nontarget_color)
    ax.set_xlim(time_intervals[0], x0 + width)

    ax.set_xlabel('stimulation time (s)')
    ax.set_ylabel('similarity score')
    ax.grid(True, alpha=0.4)
    ax.legend(loc='upper left')

def plot_metric(ax, metrics, metric_name, y_range=[0, 1]):
    subjects, cost_ratios = metrics.shape

    bar_width = 0.8 / cost_ratios
    in_between_bar = 0.5

    x = np.arange(subjects)

    # print(cost_ratios)
    
    # Plot sub-bars for each subject
    for subject in range(subjects):
        for i in range(cost_ratios):
            # print(subject, metrics[subject, i])
            ax.bar(x[subject] + i * bar_width, metrics[subject, i], width=bar_width, label=i)
    
    # Draw the horizontal line for average accuracy
    # metric_mean = np.mean(metrics)
    # ax.hlines(metric_mean, -0.5, subjects - 0.5, linestyle='--', color="k", alpha=0.5)
    
    # Customize the axes
    ax.set_ylim(y_range[0], y_range[1])
    # ax.set_xlabel("Subject")
    ax.set_ylabel(metric_name)
    # ax.set_title(f"Average {metric_name} {metric_mean:.2f}")
    # ax.legend()

    plt.xticks(x + bar_width * (cost_ratios / 2), [f'Subj {i + 1}' for i in range(subjects)])

# 125 ms segments for BDS
segment_time = 0.125

subjects = 9

segment_count = 14

bds_save_path = "data/bds_test/"
async_stopping_save_path = "data/async_stopping/"


cost_ratios = 9

# target_mean_sub = np.zeros((subjects, segment_count))
# non_target_mean_sub = np.zeros((subjects, segment_count))
# target_std_sub = np.zeros((subjects, segment_count))
# non_target_std_sub = np.zeros((subjects, segment_count))
# decision_boundary_sub = np.zeros((subjects, segment_count))

# for subject in range(subjects):
#     (X_train, V_train, y_train), (X_test, V_test, y_test), trial_time, inter_trial_time, fs, fr = utils.load_nagelspuler_dataset("data/nagelspuler/offline/VP" + str(subject + 1) + ".mat")

#     eeg2code_model = eeg2code.EEG2CodeEstimator("models/EEG2Code" + str(subject + 1) + ".keras")

#     window_size = 150

#     target_mean, non_target_mean, target_std, non_target_std, decision_boundary, normalizing_factor = eeg2code.calculate_bds_params(X_test, V_test, y_test, segment_count, window_size, eeg2code_model, 1.0)

#     target_mean_sub[subject] = target_mean
#     non_target_mean_sub[subject] = non_target_mean
#     target_std_sub[subject] = target_std
#     non_target_std_sub[subject] = non_target_std
#     decision_boundary_sub[subject] = decision_boundary

# np.savez("data/bds_calibration_non_normalized.npz", target_mean=target_mean_sub, non_target_mean=non_target_mean_sub, target_std=target_std_sub, non_target_std=non_target_std_sub, decision_boundary=decision_boundary_sub)

# fig, ax = plt.subplots(2, 1, figsize=(15, 12), constrained_layout=True)

# data = np.load("data/bds_calibration_non_normalized.npz")

# plot_dist_param(np.mean(data["non_target_mean"], axis=0), np.mean(data["target_mean"], axis=0), np.mean(data["non_target_std"], axis=0), np.mean(data["target_std"], axis=0), np.mean(data["decision_boundary"], axis=0),  np.linspace(0, 2, num=segment_count), "EEG2Code Not Normalized", ax[0])

# data = np.load("data/bds_calibration_normalized.npz")

# plot_dist_param(np.mean(data["non_target_mean"], axis=0), np.mean(data["target_mean"], axis=0), np.mean(data["non_target_std"], axis=0), np.mean(data["target_std"], axis=0), np.mean(data["decision_boundary"], axis=0),  np.linspace(0, 2, num=segment_count), "EEG2Code Normalized", ax[1])

# plt.savefig("./eeg2code_bds_dist.png", dpi=300)

# for subject in (7, 8):
#     for i, cost_ratio in enumerate([10]):
#         predicted, duration, n_classes = eeg2code.bds_performance_test(subject, cost_ratio, segment_time)

#         np.save(save_path + "predicted-" + str(subject + 1) + "-" + str(cost_ratio) + ".npy", predicted)
#         np.save(save_path + "duration-" + str(subject + 1) + "-" + str(cost_ratio) + ".npy", duration)
#         np.save(save_path + "n_classes-" + str(subject + 1) + "-" + str(cost_ratio) + ".npy", n_classes)

stopping_time = np.zeros((subjects, cost_ratios))
accuracy = np.zeros((subjects, cost_ratios))
precision = np.zeros((subjects, cost_ratios))
recall = np.zeros((subjects, cost_ratios))
f_score = np.zeros((subjects, cost_ratios))
specificity = np.zeros((subjects, cost_ratios))

for subject in range(subjects):
    _, (_, _, y_train), _, _, _, _ = utils.load_nagelspuler_dataset("data/nagelspuler/offline/VP" + str(subject + 1) + ".mat")

    true_y = y_train[:(7 * 56)]

    for i, cost_ratio in enumerate(np.logspace(-4, 4, num=cost_ratios)):
        predicted = np.load(bds_save_path + "predicted-" + str(subject + 1) + "-" + str(cost_ratio) + ".npy")
        duration = np.load(bds_save_path + "duration-" + str(subject + 1) + "-" + str(cost_ratio) + ".npy")
        n_classes = np.load(bds_save_path + "n_classes-" + str(subject + 1) + "-" + str(cost_ratio) + ".npy")

        predicted_y = np.hstack(predicted)

        accuracy[subject, i] = accuracy_score(true_y, predicted_y)
        precision[subject, i] = precision_score(true_y, predicted_y, average='micro')

        # tn, fp, fn, tp = confusion_matrix(true_y, predicted_y).ravel()
        # specificity[subject, i] = tn / (tn+fp)

        recall[subject, i] = recall_score(true_y, predicted_y, average='micro')
        f_score[subject, i] = f1_score(true_y, predicted_y, average='micro')

        stopping_time[subject, i] = np.mean(duration)

fig, ax = plt.subplots(6, 1, figsize=(15, 12), sharex=True, constrained_layout=True)

plot_metric(ax[0], accuracy, "Accuracy")
plot_metric(ax[1], stopping_time, "Stopping Time", [0, 1.75])
plot_metric(ax[2], precision, "Precision")
plot_metric(ax[3], recall, "Recall")
plot_metric(ax[4], f_score, "F-score")
plot_metric(ax[5], specificity, "Specificity")

plt.savefig("./bds-perf-metrics-eeg2code-classifier.png", dpi=300)
plt.show()

for subject in range(subjects):
    for p_value_threshold in np.arange(5):
        p_value_threshold = 1 * math.pow(10, -(5 * (p_value_threshold + 1)))
        # print(p_value_threshold)

        predicted, duration, n_classes = eeg2code.async_performance_test(subject, p_value_threshold, 0.75)

        np.savez(f"{async_stopping_save_path}{str(subject + 1)}-{p_value_threshold}.npz", predicted, duration, n_classes)

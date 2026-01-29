import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from scipy.stats import norm
import numpy as np

import keras
import pyntbci
import math

import eeg2code
import utils

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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

def plot_metric(ax, metrics, parameter_list, parameters_name, metric_name, y_range=[0, 1]):
    metrics = metrics.transpose()

    parameters, subjects = metrics.shape

    bar_width = 0.8 / parameters
    in_between_bar = 0.5

    x = np.arange(subjects)

    ax.grid(True, alpha=0.4)

    for parameter in range(parameters):
        rects = ax.bar(x + parameter * bar_width, metrics[parameter], width=bar_width, label=str(parameter_list[parameter]))
    
    ax.set_ylim(y_range[0], y_range[1])
    # ax.set_xlabel("Subject")
    ax.set_ylabel(metric_name)
    ax.set_xticks(x + bar_width * (parameters / 2), [f'Sub {i + 1}' for i in range(subjects)])

    ax.legend(loc='center left', title=parameter_name, ncols=2, fancybox=True, bbox_to_anchor=(1, 0.5))

EPS = math.pow(10, -100)

def stopping_confusion_counts(y_true, y_pred, early_stopped):
    """
    TP/FP/FN/TN volgens jouw stopping-definitie (zoals je eerder beschreef):
      - 'positief' = trial is gestopt vóór sig_len (early stop)
      - TP: early stop én correct
      - FP: early stop én incorrect
      - FN: niet early stop (forced) én correct
      - TN: niet early stop (forced) én incorrect
    """
    correct = (y_pred == y_true)
    
    tp = int(np.sum( early_stopped &  correct))
    fp = int(np.sum( early_stopped & ~correct))
    fn = int(np.sum(~early_stopped &  correct))
    tn = int(np.sum(~early_stopped & ~correct))
    precision   = tp / (tp + fp + EPS)
    recall      = tp / (tp + fn + EPS)
    specificity = tn / (tn + fp + EPS)
    f1          = (2 * precision * recall) / (precision + recall + EPS)

    return precision, recall, f1, specificity

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

# for subject in np.arange(subjects):
#     for cost_ratio in np.logspace(-4, 4, num=cost_ratios):
#         predicted, duration = eeg2code.bds_performance_test(subject, cost_ratio, segment_time)

#         np.savez(f"{bds_save_path}{str(subject + 1)}-{cost_ratio}.npz", predicted=predicted, duration=duration)

stopping_time = np.zeros((subjects, cost_ratios))
accuracy = np.zeros((subjects, cost_ratios))
precision = np.zeros((subjects, cost_ratios))
recall = np.zeros((subjects, cost_ratios))
f_score = np.zeros((subjects, cost_ratios))
specificity = np.zeros((subjects, cost_ratios))

for subject in range(subjects):
    _, (_, _, y_train), stimulation_time, _, _, _ = utils.load_nagelspuler_dataset("data/nagelspuler/offline/VP" + str(subject + 1) + ".mat")

    true_y = y_train[:(7 * 56)]

    for i, cost_ratio in enumerate(np.logspace(-4, 4, num=cost_ratios)):
        data = np.load(f"{bds_save_path}{subject + 1}-{cost_ratio}.npz")

        stopped_early = ((np.hstack(data["duration"]) - 1.75) >= 0)

        duration = np.mean(data["duration"], axis=1)

        predicted_y = np.hstack(data["predicted"])

        accuracy[subject, i] = accuracy_score(true_y, predicted_y)
        stopping_time[subject, i] = np.mean(duration)

        precision_matrix, recall_matrix, f1_matrix, specificity_matrix = stopping_confusion_counts(true_y, predicted_y, stopped_early)

        precision[subject, i] = precision_matrix
        recall[subject, i] = recall_matrix
        f_score[subject, i] = f1_matrix
        specificity[subject, i] = specificity_matrix

        # precision[subject, i] = precision_score(true_y, predicted_y, average='macro')
        # recall[subject, i] = recall_score(true_y, predicted_y, average='macro')
        # f_score[subject, i] = f1_score(true_y, predicted_y, average='macro')

fig, ax = plt.subplots(6, 1, figsize=(15, 12), sharex=True, constrained_layout=True)

parameter_name = "Cost ratios"
parameter_list = np.logspace(-4, 4, num=cost_ratios)

plot_metric(ax[0], accuracy, parameter_list, parameter_name, "Accuracy")
plot_metric(ax[1], stopping_time, parameter_list, parameter_name, "Stopping Time", [0, 1.75])
plot_metric(ax[2], precision, parameter_list, parameter_name, "Precision")
plot_metric(ax[3], recall, parameter_list, parameter_name, "Recall")
plot_metric(ax[4], f_score, parameter_list, parameter_name, "F-score")
plot_metric(ax[5], specificity, parameter_list, parameter_name, "Specificity")

plt.savefig("./bds-perf-metrics-eeg2code-classifier.png", dpi=300)
plt.show()

bds_accuracy = accuracy.mean(axis=0)
bds_precision = precision.mean(axis=0)
bds_accuracy_std = accuracy.std(axis=0)
bds_precision_std = precision.std(axis=0)

bds_stopping_time = stopping_time.mean(axis=0)

p_value_thresholds = 7

# for subject in range(subjects):
#     for p_value_threshold in np.arange(p_value_thresholds):
#         p_value_threshold = 1 * math.pow(10, -(5 * (p_value_threshold + 1)))

#         predicted, duration = eeg2code.async_performance_test(subject, p_value_threshold, 0.75)

#         np.savez(f"{async_stopping_save_path}{str(subject + 1)}-{p_value_threshold}.npz", predicted=predicted, duration=duration)

stopping_time = np.zeros((subjects, p_value_thresholds))
accuracy = np.zeros((subjects, p_value_thresholds))
precision = np.zeros((subjects, p_value_thresholds))
recall = np.zeros((subjects, p_value_thresholds))
f_score = np.zeros((subjects, p_value_thresholds))
specificity = np.zeros((subjects, p_value_thresholds))

for subject in range(subjects):
    _, (_, _, y_train), _, _, fs, _ = utils.load_nagelspuler_dataset("data/nagelspuler/offline/VP" + str(subject + 1) + ".mat")

    true_y = y_train

    for i, p_value_threshold in enumerate(np.arange(p_value_thresholds)):
        p_value_threshold = 1 * math.pow(10, -(5 * (p_value_threshold + 1)))

        data = np.load(f"{async_stopping_save_path}{subject + 1}-{p_value_threshold}.npz")

        stopped_early = ((np.hstack(data["duration"]) - (1.75 - (25 / fs))) >= 0)

        duration = np.mean(data["duration"], axis=1)

        predicted_y = np.hstack(data["predicted"])

        accuracy[subject, i] = accuracy_score(true_y, predicted_y)
        stopping_time[subject, i] = np.mean(duration)

        precision_matrix, recall_matrix, f1_matrix, specificity_matrix = stopping_confusion_counts(true_y, predicted_y, stopped_early)

        precision[subject, i] = precision_matrix
        recall[subject, i] = recall_matrix
        f_score[subject, i] = f1_matrix
        specificity[subject, i] = specificity_matrix

        # precision[subject, i] = precision_score(true_y, predicted_y, average='macro')
        # recall[subject, i] = recall_score(true_y, predicted_y, average='macro')
        # f_score[subject, i] = f1_score(true_y, predicted_y, average='macro')

async_accuracy = accuracy.mean(axis=0)
async_precision = precision.mean(axis=0)
async_accuracy_std = accuracy.std(axis=0)
async_precision_std = precision.std(axis=0)

async_stopping_time = stopping_time.mean(axis=0)

fig, ax = plt.subplots(2, 1, figsize=(15, 12), constrained_layout=True)

ax[0].plot(bds_stopping_time, bds_accuracy, color='blue', marker='o', label="BDS Adaptation")
ax[0].fill_between(bds_stopping_time, bds_accuracy - bds_accuracy_std, bds_accuracy + bds_accuracy_std, color='blue', alpha=0.2)

ax[0].plot(async_stopping_time, async_accuracy, color='orange', marker='o', label="Async Stopping")
ax[0].fill_between(async_stopping_time, async_accuracy - async_accuracy_std, async_accuracy + async_accuracy_std, color='orange', alpha=0.2)

ax[0].set_xticks(np.arange(0, 1.75, step=0.25))
ax[0].set_xlim((0, 1.75))
ax[0].set_xlabel("Average stopping time (s)")

ax[0].set_yticks(np.arange(0, 1, step=0.25))
ax[0].set_ylim((0, 1))
ax[0].set_ylabel("Average classification accuracy")

ax[0].legend()

ax[0].grid()

ax[1].plot(bds_stopping_time, bds_precision, color='blue', marker='o', label="BDS Adaptation")
ax[1].fill_between(bds_stopping_time, bds_precision - bds_precision_std, bds_precision + bds_precision_std, color='blue', alpha=0.2)

ax[1].plot(async_stopping_time, async_precision, color='orange', marker='o', label="Async Stopping")
ax[1].fill_between(async_stopping_time, async_precision - async_precision_std, async_precision + async_precision_std, color='orange', alpha=0.2)

ax[1].set_xticks(np.arange(0, 1.75, step=0.25))
ax[1].set_xlim((0, 1.75))
ax[1].set_xlabel("Average stopping time (s)")

ax[1].set_yticks(np.arange(0, 1, step=0.25))
ax[1].set_ylim((0, 1))
ax[1].set_ylabel("Average classification precision")

ax[1].legend()

ax[1].grid()
fig.savefig("./accuracy-precision-stopping-time-comparison.png", dpi=300)
plt.show()
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np

import keras
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

_, _, X, y, V = utils.load_offline_eeg2code_dataset("data/eeg2code/offline/VP2.mat")

V = V["test"]

print("Calculating BDS values for EEG2Code")

target_mean, non_target_mean, target_std, non_target_std, decision_boundary = eeg2code.calculate_bds_values(X, y, V)

size = target_mean.shape[0]

plot_dist_param(non_target_mean, target_mean, non_target_std, target_std, decision_boundary,  np.linspace(0, 2, num=size), "EEG2Code")

# print("Calculating BDS values for rCCA")

# V = y[np.arange(y.shape[0]), V]
# y = np.arange(V.shape[0])
# print(y.shape, V.shape)

# target_mean, non_target_mean, target_std, non_target_std, decision_boundary = rcca.calculate_bds_values(X, y, V, 2, 32)

# size = target_mean.shape[0]

# plot_dist_param(non_target_mean, target_mean, non_target_std, target_std, decision_boundary,  np.linspace(0, 2, num=size), "RCCA")

plt.xlabel('stimulation time (s)')
plt.ylabel('similarity score')
plt.grid(True, alpha=0.4)
plt.legend(loc='upper left')

plt.tight_layout()
plt.show()
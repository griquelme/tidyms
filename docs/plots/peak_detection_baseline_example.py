import tidyms as ms
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
signal_height = 100
snr = 10
n_col = 4
x = np.arange(200)
noise_level = signal_height / snr
noise = np.random.normal(size=x.size, scale=noise_level)
fig, ax = plt.subplots(
    nrows=3, ncols=n_col, figsize=(12, 12), sharex=True, sharey=True)

# first row: one peak, different baselines
row = 0
baselines = [4, ms.utils.gauss(x, 100, 40, 20), x ** 2 * 0.002,
             np.sin(x * np.pi / 400) * 50]
for col in range(n_col):
    signal = ms.utils.gauss(x, 100, 3, signal_height)
    y = signal + noise
    noise_estimation = ms.peaks.estimate_noise(y)
    ys = ms.lcms.gaussian_filter1d(y, 1)
    baseline_estimation = ms.peaks.estimate_baseline(ys, noise_estimation)
    start, apex, end = ms.peaks.detect_peaks(
        ys, noise_estimation, baseline_estimation)
    peaks = [ms.lcms.Peak(s, a, p) for (s, a, p) in zip(start, apex, end)]
    ax[row, col].plot(x, y)
    ax[row, col].plot(x, baseline_estimation)
    for p in peaks:
        ax[row, col].fill_between(x[p.start:p.end + 1],
                                  baseline_estimation[p.start:p.end + 1],
                                  y[p.start:p.end + 1], alpha=0.25)

# second row: two peaks, same baselines as first row
row = 1
for col in range(n_col):
    gaussian_params = np.array([[100, 3, signal_height],
                                [110, 3, signal_height]])
    signal = ms.utils.gaussian_mixture(x, gaussian_params).sum(axis=0)
    y = signal + baselines[col] + noise
    noise_estimation = ms.peaks.estimate_noise(y)
    ys = ms.lcms.gaussian_filter1d(y, 1)
    baseline_estimation = ms.peaks.estimate_baseline(ys, noise_estimation)
    start, apex, end = ms.peaks.detect_peaks(
        ys, noise_estimation, baseline_estimation)
    peaks = [ms.lcms.Peak(s, a, p) for (s, a, p) in zip(start, apex, end)]
    ax[row, col].plot(x, y)
    ax[row, col].plot(x, baseline_estimation)
    for p in peaks:
        ax[row, col].fill_between(x[p.start:p.end + 1],
                                  baseline_estimation[p.start:p.end + 1],
                                  y[p.start:p.end + 1], alpha=0.25)

# third row: different peak widths:
row = 2
widths = [3, 5, 7, 10]
for col in range(n_col):
    w = widths[col]
    signal = ms.utils.gauss(x, 100, w, signal_height)
    y = signal + baselines[0] + noise
    noise_estimation = ms.peaks.estimate_noise(y)
    ys = ms.lcms.gaussian_filter1d(y, 1)
    baseline_estimation = ms.peaks.estimate_baseline(ys, noise_estimation)
    start, apex, end = ms.peaks.detect_peaks(
        ys, noise_estimation, baseline_estimation)
    peaks = [ms.lcms.Peak(s, a, p) for (s, a, p) in zip(start, apex, end)]
    ax[row, col].plot(x, y)
    ax[row, col].plot(x, baseline_estimation)
    for p in peaks:
        ax[row, col].fill_between(x[p.start:p.end + 1],
                                  baseline_estimation[p.start:p.end + 1],
                                  y[p.start:p.end + 1], alpha=0.25)

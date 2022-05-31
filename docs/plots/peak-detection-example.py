import numpy as np
import matplotlib.pyplot as plt
from tidyms import peaks
from tidyms.lcms import Peak
from tidyms.utils import gaussian_mixture

# always generate the same plot
np.random.seed(1234)

# create a signal with two gaussian peaks
x = np.arange(100)
gaussian_params = np.array([[25, 3, 30], [50, 2, 60]])
y = gaussian_mixture(x, gaussian_params).sum(axis=0)
# add a noise term
y += np.random.normal(size=y.size, scale=0.5)

noise_estimation = peaks.estimate_noise(y)
baseline_estimation = peaks.estimate_baseline(y, noise_estimation)
start, apex, end = peaks.detect_peaks(y, noise_estimation, baseline_estimation)
peaks = [Peak(s, a, p) for s, a, p in zip(start, apex, end)]
fig, ax = plt.subplots()
ax.plot(x, y)
for p in peaks:
    ax.fill_between(x[p.start:p.end], y[p.start:p.end], alpha=0.25)

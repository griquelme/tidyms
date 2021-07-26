import numpy as np
from tidyms.peaks import detect_peaks
from tidyms.peaks import get_peak_descriptors
from tidyms.utils import gaussian_mixture

# always generate the same plot
np.random.seed(1234)

# create a signal with two gaussian peaks
x = np.arange(100)
gaussian_params = np.array([[25, 3, 30], [50, 2, 60]])
y = gaussian_mixture(x, gaussian_params).sum(axis=0)
# add a noise term
y += np.random.normal(size=y.size, scale=1)

# detect_peaks also returns the noise and baseline estimation used
peaks, noise, baseline = detect_peaks(y)
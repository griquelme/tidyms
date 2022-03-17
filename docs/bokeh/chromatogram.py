from bokeh import plotting
import tidyms as ms
import numpy as np
import sys
out = sys.argv[-1]
plotting.output_file(out)

# generate always the same plot
np.random.seed(1234)

rt = np.arange(200)
chrom_params = np.array([[50, 5, 10], [70, 5, 20]])

# creates a mixture of gaussian peaks and add noise
spint = ms.utils.gaussian_mixture(rt, chrom_params).sum(axis=0)

# add a noise term
spint += np.random.normal(size=spint.size, scale=0.1)

# create a chromatogram object
chromatogram = ms.Chromatogram(rt, spint)
p = chromatogram.plot(fig_params={"height": 250}, show=False)
plotting.save(p)
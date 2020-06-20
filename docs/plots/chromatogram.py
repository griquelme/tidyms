import tidyms as ms
import numpy as np
from bokeh import plotting
plotting.output_file("example.html")

rt = np.arange(200)
chrom_params = np.array([[50, 5, 10], [70, 5, 20]])

# creates a mixture of gaussian peaks and add noise
spint = ms.utils.gaussian_mixture(rt, chrom_params).sum(axis=0)

# create a chromatogram object
chromatogram = ms.Chromatogram(rt, spint)
chromatogram.plot(fig_params={"height": 250})
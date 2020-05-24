import ms_feature_validation as mfv
import numpy as np
import bokeh as bk
bk.plotting.output_file("example.html")

rt = np.arange(200)
chrom_params = np.array([[50, 5, 10], [70, 5, 20]])

# creates a mixture of gaussian peaks and add noise
spint = mfv.utils.gaussian_mixture(rt, chrom_params).sum(axis=0)

# create a chromatogram object
chromatogram = mfv.Chromatogram(spint, rt)
chromatogram.find_peaks()
chromatogram.plot(fig_params={"height": 250})
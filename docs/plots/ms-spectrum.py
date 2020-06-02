import numpy as np
import ms_feature_validation as mfv
from bokeh import plotting
plotting.output_file("example.html")

mz = np.linspace(400, 404, 1000)

# creates three gaussian peaks simulating an isotopic envelope
sp_params = np.array([[401, 0.01, 100], [402, 0.01, 15], [403, 0.01, 2]])
spint = mfv.utils.gaussian_mixture(mz, sp_params).sum(axis=0)

spectrum = mfv.MSSpectrum(mz, spint)
spectrum.find_peaks()
spectrum.plot(fig_params={"height": 250})

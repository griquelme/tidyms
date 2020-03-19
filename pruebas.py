import ms_feature_validation as mfv
import pickle
import numpy as np

with open("notebooks/sp_data.pickle", "rb") as fin:
    d = pickle.load(fin)

with open("notebooks/sp.pickle", "rb") as fin:
    sp = pickle.load(fin)

ms = d["mz"]
spint = d["spint"]



widths = mfv.lcms.make_widths_ms(0.01, 0.1)
peaks = mfv.peaks.pick_cwt(ms, spint, widths, min_width=0.01, max_width=0.1,
                           max_distance=0.005, min_length=5)

plist = sp.get_peak_params()
mz = plist["mz"].values
a = mfv.lcms.find_isotopic_distribution(mz, 596.732, 3, 10, 0.005)
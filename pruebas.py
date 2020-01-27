import matplotlib.pyplot as plt
import ms_feature_validation as mfv
import pyopenms
import time
import pandas as pd
import numpy as np

if __name__ == "__main__":

    #las muestras estan por duplicado, este codigo las agrupa y promedia
    data = mfv.process.read_progenesis("examples/MC.csv")
    data.data_path = "examples/raw/"
    #d = pd.read_pickle("examples/dif.pickle")
    #data = data.select(d, "features")
    ft = pd.read_excel("examples/MC selección 01ener2017.xlsx")
    ft = ft["Compound"].dropna()
    ft = pd.Index(ft.values)
    sample_id = pd.read_excel("examples/MC_2016_PabloMarchi.xlsx",
                              sheet_name="SamplesList")
    sample_id.dropna(inplace=True)
    sample_id.columns = pd.Index(["filename", "description"])
    sample_id.set_index("filename", inplace=True)
    sample_id["class"] = (sample_id["description"]
                          .str.split("_")
                          .apply(lambda x: x[0]))
    sample_id_get = lambda x: (x[0] + " " + x[1]) if len(x) > 1 else x[0]
    sample_id["sample"] = (sample_id["description"]
                           .str.split("_")
                           .apply(sample_id_get))
    sample_id = sample_id.drop(columns=["description"])
    sample_id = sample_id[~sample_id["sample"].isin(["zero", "SV MeOH H2O"])]
    data.sample_metadata["id"] = sample_id["sample"]

    duplicate_averager = mfv.process.DuplicateAverager(process_classes=["MC7", "MCC", "MCH"])

    bc_params = {"corrector_classes": ["SV"],
                 "process_classes": ["MCH", "MC7"]}
    pf_params = {"process_classes": ["MCH", "MC7"],
                 "lb": 0.8,
                 "ub": 1}

    pfg_params = {"process_classes": ["MCH", "MC7", "MCC"],
                  "lb": 0.7,
                  "ub": 1,
                  "intraclass": False}

    blank_correction = mfv.process.BlankCorrector(**bc_params)
    prevalence_filter = mfv.process.PrevalenceFilter(**pf_params)
    blank_correction.process(data)
    duplicate_averager.process(data)
    data.data_matrix[data.data_matrix <= 5] = 0
    prevalence_filter.process(data)
    cmaker = mfv.process.ChromatogramMaker(verbose=True)
    cmaker.process(data)

    def assign_rt(dc, sample, tolerance, **kwargs):
        res = mfv.peaks.make_empty_peaks()
        res["feature"] = np.array([])
        rt, chroms = dc.chromatograms[sample]
        for groups, features in dc.feature_metadata.groupby("mz_cluster"):
            peaks = mfv.peaks.pick(rt, chroms[groups, :], **kwargs)
            try:
                peaks = assign_exp_rt(peaks, features["rt"], tolerance)
            except ValueError:
                peaks = mfv.peaks.make_empty_peaks()
                peaks["feature"] = np.array([])
            res = {k: np.hstack((peaks[k], res[k])) for k in res}
        return res

    def process_chromatograms(dc, tolerance, **kwargs):
        res = dict()
        for sample in dc.get_available_samples():
            res[sample] = assign_rt(dc, sample, tolerance, **kwargs)
        dc.chromatograms_info = pd.DataFrame(res)



    def assign_exp_rt(peaks, ft_rt, tolerance):
        exp_index = find_closest_index(peaks["loc"], ft_rt)
        tolerance_bound = np.abs(peaks["loc"][exp_index] - ft_rt) < tolerance
        exp_index = exp_index[tolerance_bound]
        for k in peaks:
            peaks[k] = peaks[k][exp_index]
        peaks["feature"] = ft_rt[tolerance_bound].index
        return peaks


    def find_closest_index(x, y):
        X, Y = np.meshgrid(x, y)
        return np.argmin(np.abs(X - Y), axis=1)


    process_chromatograms(data, 1000, asymmetry=True, fwhm=(5, 60),
                          integrate=True)
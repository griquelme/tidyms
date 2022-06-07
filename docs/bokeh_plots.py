from bokeh import plotting
import tidyms as ms
import numpy as np
from pathlib import Path


seed = 1234


def create_chromatogram() -> ms.Chromatogram:

    filename = "NZ_20200227_039.mzML"
    dataset = "test-nist-raw-data"
    ms.fileio.download_tidyms_data(dataset, [filename])
    path = Path(ms.fileio.get_tidyms_path())
    path = path.joinpath(dataset, filename)

    ms_data = ms.MSData(
        path,
        ms_mode="centroid",
        instrument="qtof",
        separation="uplc"
    )
    mz_list = np.array([189.0734])
    return ms.make_chromatograms(ms_data, mz_list)[0]


def plot_chromatogram():
    plotting.output_file("_static/chromatogram.html")
    chromatogram = create_chromatogram()
    p = chromatogram.plot(show=False)
    plotting.save(p)


def plot_chromatogram_with_peaks():
    # generate always the same plot
    plotting.output_file("_static/chromatogram-with-peaks.html")
    chromatogram = create_chromatogram()
    chromatogram.extract_features()
    p = chromatogram.plot(show=False)
    plotting.save(p)


def feature_plot():
    plotting.output_file("_static/feature-plot.html")
    data = ms.fileio.load_dataset("reference-materials")
    ignore = ["Z", "SV", "B", "SSS", "SCQC"]
    # search [M+H]+ from trp in the features
    mz = 205.097
    rt = 124
    # get a list of features compatible with the given m/z and rt
    ft_name = data.select_features(mz, rt)

    f = data.plot.feature(ft_name[0], draw=False, ignore_classes=ignore)
    plotting.save(f)


def pca_plot():
    plotting.output_file("_static/pca-scores.html")

    data = ms.fileio.load_dataset("reference-materials")
    ignore = ["Z", "SV", "B", "SSS", "SCQC"]
    f = data.plot.pca_scores(fig_params={"height": 250},
                             ignore_classes=ignore,
                             scaling="autoscaling",
                             draw=False)
    plotting.save(f)


def create_assay(assay_path) -> ms.Assay:
    plotting.output_file("_static/pca-scores.html")
    ms.fileio.download_dataset("test-nist-raw-data")
    ms.fileio.download_dataset("reference-materials")
    tidyms_dir = Path(ms.utils.get_tidyms_path())
    data_path = tidyms_dir.joinpath("test-nist-raw-data")
    sample_metadata_path = data_path.joinpath("sample_list.csv")

    assay = ms.Assay(
        data_path=data_path,
        assay_path=assay_path,
        sample_metadata=sample_metadata_path,
        separation="uplc",
        instrument="qtof"
    )
    return assay


def plot_roi_assay(assay: ms.Assay, save_path: str):
    plotting.output_file(save_path)
    sample_name = "NZ_20200227_039"
    p = assay.plot.roi(sample_name, show=False)
    plotting.save(p)


def plot_stacked_chromatogram(assay: ms.Assay):
    plotting.output_file("_static/stacked-chromatograms.html")
    p = assay.plot.stacked_chromatogram(6, show=False)
    plotting.save(p)


def create_assay_plots():
    assay_path = "_build/test-assay"
    assay = create_assay(assay_path)
    mz_list = np.array(
        [118.0654, 144.0810, 146.0605, 181.0720, 188.0706, 189.0738,
         195.0875, 205.0969]
    )
    make_roi_params = {
        "tolerance": 0.015,
        "min_intensity": 5000,
        "targeted_mz": mz_list,
    }
    assay.detect_features(verbose=False, **make_roi_params)
    plot_roi_assay(assay, "_static/roi-no-peaks.html")
    assay.extract_features(store_smoothed=True, verbose=False)
    assay.describe_features(verbose=False)
    assay.build_feature_table()
    assay.match_features(verbose=False)
    plot_roi_assay(assay, "_static/roi-peaks.html")
    plot_stacked_chromatogram(assay)
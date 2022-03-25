from bokeh import plotting
import tidyms as ms
import numpy as np


seed = 1234


def plot_chromatogram():
    plotting.output_file("_static/chromatogram.html")

    # generate always the same plot
    np.random.seed(seed)

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


def plot_chromatogram_with_peaks():
    # generate always the same plot
    plotting.output_file("_static/chromatogram-with-peaks.html")
    np.random.seed(seed)

    rt = np.arange(200)
    chrom_params = np.array([[50, 5, 10], [70, 5, 20]])

    # creates a mixture of gaussian peaks and add noise
    spint = ms.utils.gaussian_mixture(rt, chrom_params).sum(axis=0)

    # add a noise term
    spint += np.random.normal(size=spint.size, scale=0.1)

    # create a chromatogram object
    chromatogram = ms.Chromatogram(rt, spint)
    chromatogram.find_peaks()
    p = chromatogram.plot(fig_params={"height": 250}, show=False)
    plotting.save(p)


def feature_plot():
    plotting.output_file("_static/feature-plot.html")
    data = ms.fileio.load_dataset("reference-materials")

    # search [M+H]+ from trp in the features
    mz = 205.097
    rt = 124
    # get a list of features compatible with the given m/z and rt
    ft_name = data.select_features(mz, rt)

    f = data.plot.feature(ft_name[0], draw=False)
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

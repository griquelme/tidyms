import ms_feature_validation as mfv
from bokeh import plotting
plotting.output_file("example.html")

data = mfv.fileio.load_dataset("ltr")
data.plot.pca_scores(fig_params={"height": 250})

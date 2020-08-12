import tidyms as ms
from bokeh import plotting
plotting.output_file("example.html")

data = ms.fileio.load_dataset("reference-materials")
ignore = ["Z", "SV", "B", "SSS", "SCQC"]
data.plot.pca_scores(fig_params={"height": 250},
                     ignore_classes=ignore,
                     scaling="autoscaling")

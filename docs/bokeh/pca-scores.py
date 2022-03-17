import tidyms as ms
from bokeh import plotting
import sys
out = sys.argv[-1]
plotting.output_file(out)

data = ms.fileio.load_dataset("reference-materials")
ignore = ["Z", "SV", "B", "SSS", "SCQC"]
f = data.plot.pca_scores(fig_params={"height": 250},
                         ignore_classes=ignore,
                         scaling="autoscaling",
                         draw=False)
plotting.save(f)

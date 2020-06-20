import tidyms as ms
from bokeh import plotting
plotting.output_file("example.html")

data = ms.fileio.load_dataset("ltr")
data.plot.pca_scores(fig_params={"height": 250})

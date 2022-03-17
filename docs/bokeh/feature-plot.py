import tidyms as ms
from bokeh import plotting
import numpy as np
import sys
out = sys.argv[-1]
plotting.output_file(out)
data = ms.fileio.load_dataset("reference-materials")

# search [M+H]+ from trp in the features
mz = 205.097
rt = 124
# get a list of features compatible with the given m/z and rt
ft_name = data.select_features(mz, rt)

f = data.plot.feature(ft_name[0], draw=False)
plotting.save(f)

import ms_feature_validation as mfv
from bokeh import plotting
import numpy as np
plotting.output_file("example.html")

data = mfv.fileio.load_dataset("ltr")

# search [M+H]+ from trp in the features
mz = 205.097
rt = 124
# get a list of features compatible with the given m/z and rt
ft_name = data.select_features(mz, rt)

# add order info
data.order = np.arange(data.data_matrix.shape[0])
data.plot.feature(ft_name[0])

import ms_feature_validation as mfv
import pandas as pd
# import numpy as np


fname = "notebooks/progenesis_data_matrix_20190918.csv"
data = mfv.filter.read_progenesis(fname)
# adding order and batch information
temp = pd.Series(data=data.sample_metadata.index.str.split("_"),
                 index=data.sample_metadata.index)
order = temp.apply(lambda x: x[1]).astype(int)
dates = temp.apply(lambda x: x[0])
dates_to_batch = dict(zip(dates.unique(), range(1, dates.size + 1)))
batch = (temp.apply(lambda x: dates_to_batch[x[0]])).astype(int)

def convert_to_global_run_order(order, batch):
    max_order = order.groupby(batch).max()
    max_order[0] = 0
    global_run_order = order + batch.apply(lambda x: max_order[x - 1])
    return global_run_order

data.order = convert_to_global_run_order(order, batch)
data.batch = batch
data.id = data.sample_metadata.index

# setup sample types
sample_mapping = {"qc": ["QC d2 v1", "QC d2 v2", "QC d1 v1", "QC d1 v2"],
                 "suitability": ["standards mixture"],
                 "blank": ["solvent blank", "Solvent"],
                 "zero": ["Zero"]}
data.mapping = sample_mapping
trp = data.select_features(203.082, 128)
data.plot.feature(trp)


# if __name__ == "__main__":
#     main()

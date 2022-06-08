import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import seaborn as sns
from itertools import product

sns.set_context("paper", font_scale=1.25)


sample_size = [10, 20, 50, 100, 200, 500]
fractions = [0.1, 0.25, 0.5, 0.75, 1.0]
eps = [0.5, 1.0, 2.0, 3.0, 4.0]
n_reps = 5
results = list()

for k_rep, size, f, e in product(range(n_reps), sample_size, fractions, eps):
    X = np.random.normal(size=(size, 2))
    min_samples = round(size * f)
    dbscan = DBSCAN(eps=e, min_samples=min_samples, metric="chebyshev")
    dbscan.fit(X)
    cluster = dbscan.labels_
    noise_fraction = (cluster == -1).sum() / size
    results.append([k_rep, size, f, e, noise_fraction])
df_normal = pd.DataFrame(
    data=results,
    columns=["rep", "sample size", "sample fraction", "eps", "noise fraction"]
)

sns.catplot(
    data=df_normal,
    x="eps",
    y="noise fraction",
    palette="Set1",
    col="sample size",
    hue="sample fraction",
    legend="full",
    col_wrap=2,
    s=8
)
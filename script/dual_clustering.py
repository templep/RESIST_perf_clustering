# %%
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
import pandas as pd
import os.path as osp
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from common import split_data, load_data

performances = [
    "elapsedtime"  # we only consider this performance metric for the moment
]

# The plan for this file is to have two clusterings:
# 1. In the config feature space
# 2. In the input feature space
# The connection is via the distance function + ranking

# Notes
# a) Consistency: How different is feature-based clustering vs. rank-based clustering vs. perf-based clustering?
# b) Configuration recommendation
# - (x) Baseline: kNN: Query k nearest neighbours and let them vote on recommended configuration
# - Rank-based clustering: Cluster inputs via rank-based distance, look up closest cluster via input features and pick
# - Hierarchical clustering:
# - Metric: MAPE of configuration performance, Recall@k (open), Avg. rank of selected configuration
# c) Performance Prediction
# -


## Configuration
# Enter names of performance columns to consider
performances = ["elapsedtime"]


## Load and prepare data
perf_matrix, input_features, config_features = load_data(data_dir="../data/")
data_split = split_data(perf_matrix)
train_inp = data_split["train_inp"]
train_cfg = data_split["train_cfg"]

# This is a look up for performance measurements from inputname + configurationID
input_config_map = (
    perf_matrix[["inputname", "configurationID"] + performances]
    .sort_values(["inputname", "configurationID"])
    .set_index(["inputname", "configurationID"])
)
all_input_names = pd.Series(
    input_config_map.index.get_level_values("inputname").unique()
)
all_config_ids = pd.Series(
    input_config_map.index.get_level_values("configurationID").unique()
)

error_mape = input_config_map.groupby("inputname").transform(
    lambda x: (x - x.min()).abs() / abs(x.min())
)
average_mape = error_mape.mean(axis=1)

rank_map = input_config_map.groupby("inputname").transform(
    lambda x: stats.rankdata(x, method="min")
)
average_ranks = rank_map.mean(axis=1)

## Create functions
# Some require global information from above


# This is a look up for performance measurements from inputname + configurationID
input_config_map = (
    perf_matrix[["inputname", "configurationID"] + performances]
    .sort_values(["inputname", "configurationID"])
    .set_index(["inputname", "configurationID"])
)
all_input_names = pd.Series(
    input_config_map.index.get_level_values("inputname").unique()
)
all_config_ids = pd.Series(
    input_config_map.index.get_level_values("configurationID").unique()
)
measurements = input_config_map.values.reshape(
    (len(all_input_names), len(all_config_ids), len(performances))
)
len(all_input_names), len(all_config_ids), measurements.shape

# %%

### Rank-based distance matrix

# distance_matrix = spearman_rank_distance(measurements)
# Distance matrix may not include negative values, but spearman can be negative
# model_dist = NearestNeighbors(metric="precomputed")
# model_dist.fit(distance_matrix[:, :, 0])

# %%
## Evaluation


# %%
def distance_matrix_by_first_axis(array):
    reshaped_array = array.reshape(array.shape[0], -1)
    pairdist = cdist(reshaped_array, reshaped_array)
    return pairdist


def get_configuration_distances(measurements):
    return distance_matrix_by_first_axis(measurements)


def get_input_distances(measurements):
    return distance_matrix_by_first_axis(np.moveaxis(measurements, 0, 1))

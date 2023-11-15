# %%
from main import load_all_csv, compute_index, sort_data
import cluster
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import datetime
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram

def format_time(s):
    dtf = datetime.time.fromisoformat("00:0" + s.strip())
    return dtf.second + dtf.microsecond/1_000_000

# %%
nb_meas = 8  # this is how many measures were taken (i.e. how many measurement columns are in the dataframe)
perf_matrix, nb_data = load_all_csv("../data/res_ugc/", ext="csv", with_names=True)
idx = compute_index(perf_matrix, nb_data)
data_per_cfg = sort_data(perf_matrix, idx, nb_data)
measures = cluster.extract_feature(data=data_per_cfg, nb_meas=nb_meas+1).iloc[:, 1:nb_meas]  # don't use size column
measure_names = list(measures.columns.values)

# %%
measure_names

# %%
input_config_map = data_per_cfg[["inputname", "configurationID"] + measure_names].sort_values(["inputname", "configurationID"]).set_index(["inputname", "configurationID"])
input_config_map["elapsedtime"] = input_config_map["elapsedtime"].apply(format_time)

all_input_names = {s: i for i, s in enumerate(input_config_map.index.get_level_values("inputname").unique())}
all_config_ids = {s: i for i, s in enumerate(input_config_map.index.get_level_values("configurationID").unique())}
measurements = input_config_map.values.reshape((len(all_input_names), len(all_config_ids), len(measure_names)))
len(all_input_names), len(all_config_ids), measurements.shape

# %%
def distance_matrix_by_first_axis(array):
    reshaped_array = array.reshape(array.shape[0], -1)
    pairwise_distances = cdist(reshaped_array, reshaped_array)
    return pairwise_distances

def get_configuration_distances(measurements):
    return distance_matrix_by_first_axis(measurements)

def get_input_distances(measurements):
    return distance_matrix_by_first_axis(np.moveaxis(measurements, 0, 1))

# %%
get_configuration_distances(measurements).mean(), get_input_distances(measurements).mean()

# %%
reshaped_array = measurements.reshape(measurements.shape[0], -1)
model = AgglomerativeClustering(n_clusters=10, compute_distances=False, linkage='average')
model.fit(reshaped_array)

# %%
model.n_clusters_

# %%
model.distances_.shape

# %%
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)

plot_dendrogram(model, truncate_mode="level", p=20, color_threshold=0.01)



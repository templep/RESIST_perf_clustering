# %%
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from scipy import stats
import torch

from common import load_x264, split_data, evaluate_ii, evaluate_cc

# import argparse

# %%

# parser = argparse.ArgumentParser()
# parser.add_argument("")

## Configuration
random_seed = 33154

# Enter names of performance columns to consider
performances = ["rel_kbs"]

# Number of nearest neighbours to consider
topk_values = (1, 3, 5, 10, 20)
topr_values = (1, 3, 5, 10, 20)

## Load and prepare data
## Load and prepare data
data_dir = "../data"
perf_matrix, input_features, config_features, all_performances = load_x264(
    data_dir=data_dir
)

print(f"Loaded data x264")
print(f"perf_matrix:{perf_matrix.shape}")
print(f"input_features:{input_features.shape}")
print(f"config_features:{config_features.shape}")

data_split = split_data(perf_matrix, random_state=random_seed)
train_inp = data_split["train_inp"]
train_cfg = data_split["train_cfg"]
test_inp = data_split["test_inp"]
test_cfg = data_split["test_cfg"]

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

regret_map = input_config_map.groupby("inputname").transform(
    lambda x: (x - x.min()).abs() / abs(x.min())
)
average_mape = regret_map.mean(axis=1)

rank_map = input_config_map.groupby("inputname").transform(
    lambda x: stats.rankdata(x, method="min")
)
average_ranks = rank_map.mean(axis=1)


# %%

# 
rank_arr = torch.from_numpy(
    rank_map  #.loc[(train_inp, train_cfg), :]
    .reset_index()
    .pivot_table(index="inputname", columns="configurationID", values=performances[0])
    .values
)
regret_arr = torch.from_numpy(
    regret_map #.loc[(train_inp, train_cfg), :]
    .reset_index()
    .pivot_table(index="inputname", columns="configurationID", values=performances[0])
    .values
)

input_arr = torch.from_numpy(input_features.values).float()
config_arr = torch.from_numpy(config_features.values).float()

train_input_mask = input_features.index.isin(train_inp)
test_input_mask = input_features.index.isin(test_inp)

train_config_mask = config_features.index.isin(train_cfg)
test_config_mask = config_features.index.isin(test_cfg)

train_input_arr = input_arr[train_input_mask]
train_config_arr = config_arr[train_config_mask]


# %%

from common import top_k_closest_euclidean




train_cc = []
test_cc = []
test_ii_ranks = []
test_ii_regret = []


for topk in topk_values:
    train_cc.append(evaluate_cc(
        config_arr[train_config_mask],
        rank_arr=rank_arr[:, train_config_mask],
        n_neighbors=topk,
        n_recs=topr_values,
        # config_mask=train_config_mask
    ).numpy())

    test_cc.append(evaluate_cc(
        config_arr,
        rank_arr=rank_arr,
        n_neighbors=topk,
        n_recs=topr_values,
        config_mask=test_config_mask
    ).numpy())


    test_ii = evaluate_ii(
        input_arr,
        rank_arr=rank_arr,
        regret_arr=regret_arr,
        n_neighbors=topk,
        n_recs=topr_values,
        input_mask=train_input_mask
    )
    test_ii_ranks.append(test_ii[0].numpy())
    test_ii_regret.append(test_ii[1].numpy())
# %%
def prepare_df(results, topr_values, topk_values):
    df = pd.DataFrame(results, columns=topr_values)
    df["k"] = topk_values
    df.set_index("k", inplace=True)
    df.columns = pd.MultiIndex.from_product([["r"], df.columns])
    return df

# TODO Verify ii_regret results
# TODO Scale ii_ranks results
# TODO Share results in README

print(prepare_df(train_cc, topr_values, topk_values))
print(prepare_df(test_cc, topr_values, topk_values))
print(prepare_df(test_ii_ranks, topr_values, topk_values))
print(prepare_df(test_ii_regret, topr_values, topk_values))
# %%

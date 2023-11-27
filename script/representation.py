# %%
import torch
from torch import nn
import numpy as np
import pandas as pd
import os.path as osp
import itertools
from scipy import stats
from common import (
    split_data,
    load_data,
)
import pickle
from joblib import Parallel, delayed

# Purpose of this file
# We learn a simple embedding of the input and configuration vectors
# Input_embed() Config_embed() → joint embedding space
# Metric learning on triplets with relative relevance from collected data
# (A, P, N) triplet → Anchor, Positive Example, Negative Example
# I-I-I triplet: Inputs are closer if they benefit from the same configurations
# C-C-C triplet: Configurations are closer if they show higher effects on the same inputs
# C-I-I triplet: A configuration is closer to the input it has a higher effect on
# I-C-C triplet: An input is closer to the configuration it has a higher effect on

# Dataset construction
# Lookup table: (C,I), (I,I), (C,C)

# At each batch we need n triplets
# We can precalculate all triplets and pick from them or sample on the fly
# Let's first iterate on the pre-generation

# %%
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

# %%
ttinp = train_inp  # [:100]
ttcfg = train_cfg  # [:100]

# We define four functions to decide which item is the positive/negative for an anchor

# inpcorrs = {}
# for inp1, inp2 in itertools.combinations(sorted(ttinp), 2):
#     inpcorrs[tuple(sorted((inp1, inp2)))] = rank_map.loc[inp1].corrwith(
#         rank_map.loc[inp2]
#     )

# cfgcorrs = {}
# for cfg1, cfg2 in itertools.combinations(sorted(ttcfg), 2):
#     cfgcorrs[tuple(sorted((cfg1, cfg2)))] = rank_map.xs(cfg1, level=1).corrwith(
#         rank_map.xs(cfg2, level=1)
#     )

# def iii_cmp_fn_cache(inp1, inp2, inp3):
#     # Returns which of inp2 and inp3 is closer to inp1 in terms of rank correlation (pearson)
#     i1i2 = inpcorrs[tuple(sorted((inp1, inp2)))]
#     i1i3 = inpcorrs[tuple(sorted((inp1, inp3)))]
#     if np.argmin((i1i2, i1i3)) == 0:
#         pairs.append(("iii", inp1, inp2, inp3))
#     else:
#         pairs.append(("iii", inp1, inp3, inp2))


# def ccc_cmp_fn_cache(cfg1, cfg2, cfg3):
#     c1c2 = cfgcorrs[tuple(sorted((cfg1, cfg2)))]
#     c1c3 = cfgcorrs[tuple(sorted((cfg1, cfg3)))]

#     if np.argmin((c1c2, c1c3)) == 0:
#         pairs.append(("ccc", cfg1, cfg2, cfg3))
#     else:
#         pairs.append(("ccc", cfg1, cfg3, cfg2))


def iii_cmp_fn(inp1, inp2, inp3):
    # Returns which of inp2 and inp3 is closer to inp1 in terms of rank correlation (pearson)
    i1i2 = rank_map.loc[inp1].corrwith(rank_map.loc[inp2])
    i1i3 = rank_map.loc[inp1].corrwith(rank_map.loc[inp3])
    if (i1i2 < i1i3).item():
        return ("iii", inp1, inp2, inp3)
    else:
        return ("iii", inp1, inp3, inp2)


def ccc_cmp_fn(cfg1, cfg2, cfg3):
    c1c2 = rank_map.xs(cfg1, level=1).corrwith(rank_map.xs(cfg2, level=1))
    c1c3 = rank_map.xs(cfg1, level=1).corrwith(rank_map.xs(cfg3, level=1))
    if (c1c2 < c1c3).item():
        return ("ccc", cfg1, cfg2, cfg3)
    else:
        return ("ccc", cfg1, cfg3, cfg2)


def cii_cmp_fn(cfg, inp1, inp2):
    ci1 = rank_map.loc[(inp1, cfg)]
    ci2 = rank_map.loc[(inp2, cfg)]
    if (ci1 < ci2).item():
        return ("cii", cfg, inp1, inp2)
    else:
        return ("cii", cfg, inp2, inp1)


def icc_cmp_fn(inp, cfg1, cfg2):
    ic1 = rank_map.loc[(inp, cfg1)]
    ic2 = rank_map.loc[(inp, cfg2)]
    if (ic1 < ic2).item():
        return ("icc", inp, cfg1, cfg2)
    else:
        return ("icc", inp, cfg2, cfg1)


# %%

## This does not scale for the full dataset

# def task_generator():
#     for inps in itertools.combinations(ttinp, 3):
#         yield delayed(iii_cmp_fn)(*inps)

#     for cfgs in itertools.combinations(ttcfg, 3):
#         yield delayed(ccc_cmp_fn)(*cfgs)

#     for cfg in ttcfg:
#         for inp1, inp2 in itertools.combinations(ttinp, 2):
#             yield delayed(cii_cmp_fn)(cfg, inp1, inp2)

#     for inp in ttinp:
#         for cfg1, cfg2 in itertools.combinations(ttcfg, 2):
#             yield delayed(icc_cmp_fn)(inp, cfg1, cfg2)

# pairs = Parallel(n_jobs=-1, verbose=10)(task_generator())

# pickle.dump((ttinp, ttcfg, pairs), open("data.p", "wb"))

# Dataset creation ends


# %%
class TripletDataset(torch.utils.data.IterableDataset):
    def __init__(self, input_features, config_features, n):
        super(TripletDataset, self).__init__()
        self.input_features = input_features
        self.config_features = config_features
        self.n = n
        self.input_indices = np.arange(len(input_features))
        self.config_indices = np.arange(len(config_features))

    def make_pairs(self, input_indices, config_indices):
        # This method should be implemented to generate pairs
        # based on the sampled input and config indices.
        pass

    def __iter__(self):
        while True:
            sampled_input_indices = np.random.choice(
                self.input_indices, self.n, replace=False
            )
            sampled_config_indices = np.random.choice(
                self.config_indices, self.n, replace=False
            )
            pairs = self.make_pairs(sampled_input_indices, sampled_config_indices)
            for t, a, p, n in pairs:
                anchor = (
                    self.input_features[a] if t[0] == "i" else self.config_features[a]
                )
                positive = (
                    self.input_features[p] if t[1] == "i" else self.config_features[p]
                )
                negative = (
                    self.input_features[n] if t[2] == "i" else self.config_features[n]
                )
                yield anchor, positive, negative


# %%

# Here the actual training setup starts

train_input_arr = torch.from_numpy(input_features.loc[train_inp].values).float()
train_config_arr = torch.from_numpy(config_features.loc[train_cfg].values).float()

input_map = {s: i for i, s in enumerate(train_inp)}
config_map = {s: i for i, s in enumerate(train_cfg)}

# pairs_idx = []
# for t, a, p, n in pairs:
#     pairs_idx.append(
#         (
#             t[0] == "i",
#             input_map[a] if t[0] == "i" else config_map[a],
#             t[1] == "i",
#             input_map[p] if t[1] == "i" else config_map[p],
#             t[2] == "i",
#             input_map[n] if t[2] == "i" else config_map[n],
#         )
#     )

# pairs_idx = np.array(pairs_idx)

# %%
# def main():
num_input_features = train_input_arr.shape[1]
num_config_features = train_config_arr.shape[1]
emb_size = 24
batch_size = 256

input_emb = nn.Sequential(
    nn.Linear(num_input_features, 64), nn.ReLU(), nn.Linear(64, emb_size)
)
config_emb = nn.Sequential(
    nn.Linear(num_config_features, 64), nn.ReLU(), nn.Linear(64, emb_size)
)

optimizer = torch.optim.AdamW(
    list(input_emb.parameters()) + list(config_emb.parameters()), lr=0.0001
)


def make_batch(size):
    batch_idx = []
    for i in range(size):
        task = np.random.choice(4)
        if task == 0:  # iii
            params = np.random.choice(ttinp, size=3, replace=False)
            triplet = iii_cmp_fn(*params)
        elif task == 1:  # ccc
            params = np.random.choice(ttcfg, size=3, replace=False)
            triplet = ccc_cmp_fn(*params)
        elif task == 2:  # icc
            inp = np.random.choice(ttinp)
            cfgs = np.random.choice(ttcfg, size=2, replace=False)
            triplet = icc_cmp_fn(inp, *cfgs)
        else:  # cii
            cfg = np.random.choice(ttcfg)
            inps = np.random.choice(ttinp, size=2, replace=False)
            triplet = cii_cmp_fn(cfg, *inps)

        t, a, p, n = triplet
        batch_idx.append(
            (
                t[0] == "i",
                input_map[a] if t[0] == "i" else config_map[a],
                t[1] == "i",
                input_map[p] if t[1] == "i" else config_map[p],
                t[2] == "i",
                input_map[n] if t[2] == "i" else config_map[n],
            )
        )

    return torch.tensor(batch_idx)


def make_batch_v2(size):
    """This samples a set of `size` inputs + configs and constructs all possible triplets from them."""
    half_size = size // 2
    sampled_ttinp = np.random.choice(ttinp, size=half_size, replace=False)
    sampled_ttcfg = np.random.choice(ttcfg, size=half_size, replace=False)
    batch_idx = []

    # iii task
    for inp1, inp2, inp3 in itertools.combinations(sampled_ttinp, 3):
        batch_idx.append(iii_cmp_fn(inp1, inp2, inp3))

    # ccc task
    for cfg1, cfg2, cfg3 in itertools.combinations(sampled_ttcfg, 3):
        batch_idx.append(ccc_cmp_fn(cfg1, cfg2, cfg3))

    # icc task
    for inp in sampled_ttinp:
        for cfg1, cfg2 in itertools.combinations(sampled_ttcfg, 2):
            batch_idx.append(icc_cmp_fn(inp, cfg1, cfg2))

    # cii task
    for cfg in sampled_ttcfg:
        for inp1, inp2 in itertools.combinations(sampled_ttinp, 2):
            batch_idx.append(cii_cmp_fn(cfg, inp1, inp2))

    # Convert to indices and tensor
    batch_idx = [
        (
            t[0] == "i",
            input_map[a] if t[0] == "i" else config_map[a],
            t[1] == "i",
            input_map[p] if t[1] == "i" else config_map[p],
            t[2] == "i",
            input_map[n] if t[2] == "i" else config_map[n],
        )
        for t, a, p, n in batch_idx
    ]
    return torch.tensor(batch_idx)

# TODO For our dataset size it is relatively cheap to calculate the embeddings for all inputs and configs.
# We can every few iterations update a full collection and collect the hardest triplets from it.
# 
with torch.no_grad():
    emb_lookup = torch.empty((train_input_arr.shape[0]+train_config_arr.shape[0], emb_size))
    emb_lookup[:train_input_arr.shape[0]] = input_emb(train_input_arr)
    emb_lookup[train_input_arr.shape[0]:] = config_emb(train_config_arr)

for iteration in range(1_000):
    batch = make_batch(batch_size).reshape((-1, 2))
    input_row = batch[:, 0] == 1
    assert batch.shape[1] == 2, "Make sure to reshape batch to two columns (type, index)"

    optimizer.zero_grad()
    embeddings = torch.empty((batch.shape[0], emb_size))
    embeddings[input_row] = input_emb(train_input_arr[batch[input_row, 1]])
    embeddings[~input_row] = config_emb(train_config_arr[batch[~input_row, 1]])
    loss = nn.functional.triplet_margin_loss(
        anchor=embeddings[0::3],
        positive=embeddings[1::3],
        negative=embeddings[2::3],
    )
    loss.backward()
    optimizer.step()
    print(loss)

# %%

# TODO Validation function

# %%

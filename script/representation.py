# %%
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
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

# Here the actual training setup starts

train_input_arr = torch.from_numpy(input_features.loc[train_inp].values).float()
train_config_arr = torch.from_numpy(config_features.loc[train_cfg].values).float()

input_map = {s: i for i, s in enumerate(train_inp)}
config_map = {s: i for i, s in enumerate(train_cfg)}

# %%


def make_batch(size):
    batch_idx = []
    for _ in range(size):
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


# %%

## Validation functions


# rank_map -> IxCxP matrix
def top_k_closest_euclidean(emb1, emb2, k):
    distance = torch.cdist(emb1, emb2, p=2)
    return torch.topk(distance, k, largest=False, dim=1)[1]


def top_k_closest_cosine(emb1, emb2, k):
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)

    # Calculate cosine similarity (dot product of unit vectors)
    similarity = torch.mm(emb1_norm, emb2_norm.t())
    return torch.topk(similarity, k, largest=True, dim=1)[1]


# TODO Mapping from embeddings to correct inputs/configs
def evaluate_icc(input_embeddings, config_embeddings, rank_arr, mape_arr, k):
    # For each input, we query the k closest configurations
    # We determine their rank against the measured data and the MAPE
    # We return the best and the mean value
    top_cfg = top_k_closest_euclidean(input_embeddings, config_embeddings, k)

    # Ranks
    cfg_ranks = torch.gather(rank_arr, 1, top_cfg).float()
    best_rank = cfg_ranks.min(axis=1)[0].mean()
    avg_rank = cfg_ranks.mean(axis=1).mean()

    #  MAPE
    cfg_mape = torch.gather(mape_arr, 1, top_cfg).float()
    best_mape = cfg_mape.min(axis=1)[0].mean()
    avg_mape = cfg_mape.mean(axis=1).mean()

    return best_rank, avg_rank, best_mape, avg_mape

# TODO Maybe this is wrong, results do not change during training
def evaluate_iii(input_embeddings, rank_arr, mape_arr, k):
    # The closest input will always be the input itself,
    # therefore we ask for k+1 and drop the first column
    top_inp = top_k_closest_euclidean(input_embeddings, input_embeddings, k + 1)[:, 1:]

    # Ranks
    avg_cfg_ranks = rank_arr[top_inp].float().mean(axis=1)
    best_rank = avg_cfg_ranks.min(axis=1)[0].mean()
    avg_rank = avg_cfg_ranks.mean(axis=1).mean()

    # MAPE
    avg_cfg_mape = mape_arr[top_inp].mean(axis=1)
    best_mape = avg_cfg_mape.min(axis=1)[0].mean()
    avg_mape = avg_cfg_mape.mean(axis=1).mean()

    return best_rank, avg_rank, best_mape, avg_mape


# %%

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
    list(input_emb.parameters()) + list(config_emb.parameters()), lr=0.0003
)

# TODO For our dataset size it is relatively cheap to calculate the embeddings for all inputs and configs.
# We can every few iterations update a full collection and collect the hardest triplets from it.
#
with torch.no_grad():
    emb_lookup = torch.empty(
        (train_input_arr.shape[0] + train_config_arr.shape[0], emb_size)
    )
    emb_lookup[: train_input_arr.shape[0]] = input_emb(train_input_arr)
    emb_lookup[train_input_arr.shape[0] :] = config_emb(train_config_arr)

# For evaluation
rank_arr = torch.from_numpy(
    rank_map.loc[(ttinp, ttcfg), :]
    .reset_index()
    .pivot_table(index="inputname", columns="configurationID", values="elapsedtime")
    .values
)
mape_arr = torch.from_numpy(
    error_mape.loc[(ttinp, ttcfg), :]
    .reset_index()
    .pivot_table(index="inputname", columns="configurationID", values="elapsedtime")
    .values
)

total_loss = 0

for iteration in range(1_000):
    batch = make_batch(batch_size).reshape((-1, 2))
    input_row = batch[:, 0] == 1
    assert (
        batch.shape[1] == 2
    ), "Make sure to reshape batch to two columns (type, index)"

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
    total_loss += loss.item()

    if iteration % 10 == 0:
        with torch.no_grad():
            inputembs = input_emb(train_input_arr)
            icc_best_rank, icc_avg_rank, icc_best_mape, icc_avg_mape = evaluate_icc(
                inputembs,
                config_emb(train_config_arr),
                rank_arr,
                mape_arr,
                k=5,
            )
            iii_best_rank, iii_avg_rank, iii_best_mape, iii_avg_mape = evaluate_iii(
                inputembs,
                rank_arr,
                mape_arr,
                k=5,
            )
            print(
                f"l:{total_loss/10:.3f} | "
                + f"icc(rank(best:{icc_best_rank:.2f} "
                + f"avg:{icc_avg_rank:.2f}) "
                + f"mape(best:{icc_best_mape:.2f} "
                + f"avg:{icc_avg_mape:.2f})) | "
                + f"iii(rank(best:{iii_best_rank:.2f} "
                + f"avg:{iii_avg_rank:.2f}) "
                + f"mape(best:{iii_best_mape:.2f} "
                + f"avg:{iii_avg_mape:.2f}))"
            )
            total_loss = 0

# %%

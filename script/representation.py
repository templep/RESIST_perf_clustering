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
    load_x264,
)
import pickle
import os

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
## Load and prepare data
data_dir = "../data"
perf_matrix, input_features, config_features, performances = load_x264(
    data_dir=data_dir
)

print(f"Loaded data x264")
print(f"perf_matrix:{perf_matrix.shape}")
print(f"input_features:{input_features.shape}")
print(f"config_features:{config_features.shape}")

data_split = split_data(perf_matrix)
train_inp = data_split["train_inp"]
train_cfg = data_split["train_cfg"]

# This is a look up for performance measurements from inputname + configurationID
input_config_map = (
    perf_matrix[["inputname", "configurationID"] + performances[:1]]
    .sort_values(["inputname", "configurationID"])
    .set_index(["inputname", "configurationID"])
)
all_input_names = pd.Series(
    input_config_map.index.get_level_values("inputname").unique()
)
all_config_ids = pd.Series(
    input_config_map.index.get_level_values("configurationID").unique()
)

error_regret = input_config_map.groupby("inputname").transform(
    lambda x: (x - x.min()).abs() / abs(x.min())
)
average_regret = error_regret.mean(axis=1)

rank_map = input_config_map.groupby("inputname").transform(
    lambda x: stats.rankdata(x, method="min")
)
average_ranks = rank_map.mean(axis=1)

# Correlation cache
correlation_file = os.path.join(data_dir, "x264_correlations.p")
if os.path.exists(correlation_file):
    corr_dict = pickle.load(open(correlation_file, "rb"))
    input_correlations = corr_dict["input_correlations"]
    config_correlations = corr_dict["config_correlations"]
else:
    input_correlations = None
    config_correlations = None

# %%

# We define four functions to rank two items compared to an anchor item

# TODO cmp_fn can only handle a single performance measure in `rank_map`


def iii_cmp_fn(inp1, inp2, inp3, rank_map=None, lookup=None):
    """Returns which of inp2 and inp3 is closer to inp1 in terms of rank correlation (pearson)."""
    if lookup is not None:
        i1i2 = lookup[tuple(sorted((inp1, inp2)))]
        i1i3 = lookup[tuple(sorted((inp1, inp3)))]
    elif rank_map is not None:
        i1i2 = rank_map.loc[inp1].corrwith(rank_map.loc[inp2])
        i1i3 = rank_map.loc[inp1].corrwith(rank_map.loc[inp3])
    else:
        raise Exception("Either `rank_map` or `lookup` must be provided.")

    if (i1i2 < i1i3).item():
        return ("iii", inp1, inp2, inp3)
    else:
        return ("iii", inp1, inp3, inp2)


def ccc_cmp_fn(cfg1, cfg2, cfg3, rank_map=None, lookup=None):
    """Returns which of cfg2 and cfg3 is closer to cfg1 in terms of rank correlation (pearson)."""
    if lookup is not None:
        c1c2 = lookup[tuple(sorted((cfg1, cfg2)))]
        c1c3 = lookup[tuple(sorted((cfg1, cfg3)))]
    elif rank_map is not None:
        c1c2 = rank_map.xs(cfg1, level=1).corrwith(rank_map.xs(cfg2, level=1))
        c1c3 = rank_map.xs(cfg1, level=1).corrwith(rank_map.xs(cfg3, level=1))
    else:
        raise Exception("Either `rank_map` or `lookup` must be provided.")

    if (c1c2 < c1c3).item():
        return ("ccc", cfg1, cfg2, cfg3)
    else:
        return ("ccc", cfg1, cfg3, cfg2)


def cii_cmp_fn(cfg, inp1, inp2, rank_map):
    ci1 = rank_map.loc[(inp1, cfg)]
    ci2 = rank_map.loc[(inp2, cfg)]

    if (ci1 < ci2).item():
        return ("cii", cfg, inp1, inp2)
    else:
        return ("cii", cfg, inp2, inp1)


def icc_cmp_fn(inp, cfg1, cfg2, rank_map):
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


# TODO Make vectorized version that splits evenly between the tasks
def make_batch(inputs, configs, size, rank_map=None, lookup=None):
    batch_idx = []
    for _ in range(size):
        task = np.random.choice(4)
        if task == 0:  # iii
            params = np.random.choice(inputs, size=3, replace=False)
            triplet = iii_cmp_fn(*params, rank_map=rank_map, lookup=lookup)
        elif task == 1:  # ccc
            params = np.random.choice(configs, size=3, replace=False)
            triplet = ccc_cmp_fn(*params, rank_map=rank_map, lookup=lookup)
        elif task == 2:  # icc
            inp = np.random.choice(inputs)
            cfgs = np.random.choice(configs, size=2, replace=False)
            triplet = icc_cmp_fn(inp, *cfgs, rank_map=rank_map)
        else:  # cii
            cfg = np.random.choice(configs)
            inps = np.random.choice(inputs, size=2, replace=False)
            triplet = cii_cmp_fn(cfg, *inps, rank_map=rank_map)

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


def make_batch_v2(inputs, configs, size, rank_map=None, lookup=None):
    """This samples a set of `size` inputs + configs and constructs all possible triplets from them."""
    half_size = size // 2
    sampled_ttinp = np.random.choice(inputs, size=half_size, replace=False)
    sampled_ttcfg = np.random.choice(configs, size=half_size, replace=False)
    batch_idx = []

    # iii task
    for inp1, inp2, inp3 in itertools.combinations(sampled_ttinp, 3):
        batch_idx.append(iii_cmp_fn(inp1, inp2, inp3, rank_map=rank_map, lookup=lookup))

    # ccc task
    for cfg1, cfg2, cfg3 in itertools.combinations(sampled_ttcfg, 3):
        batch_idx.append(ccc_cmp_fn(cfg1, cfg2, cfg3, rank_map=rank_map, lookup=lookup))

    # icc task
    for inp in sampled_ttinp:
        for cfg1, cfg2 in itertools.combinations(sampled_ttcfg, 2):
            batch_idx.append(icc_cmp_fn(inp, cfg1, cfg2, rank_map=rank_map))

    # cii task
    for cfg in sampled_ttcfg:
        for inp1, inp2 in itertools.combinations(sampled_ttinp, 2):
            batch_idx.append(cii_cmp_fn(cfg, inp1, inp2, rank_map=rank_map))

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


def make_batch_v3(inputs, configs, size, rank_map=None, lookup=None):
    mask = torch.tensor([[1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 1, 1]], dtype=bool)
    batch_mask = mask[np.random.choice(mask.shape[0], size=10, replace=True)]
    n_inputs = batch_mask.sum()
    n_configs = (~batch_mask).sum()

    # selected_inputs = np.random.choice(len(inputs), size=(n_inputs,), replace=False)
    # seleced_configs = np.random.choice(len(configs), size=(n_configs,), replace=False)

    batch_idx = torch.empty((size, 6), dtype=int)
    batch_idx[:, 0::2] = batch_mask
    batch_idx[:, 1::2][batch_mask] = torch.from_numpy()
    batch_idx[:, 1::2][~batch_mask] = torch.from_numpy()

    for i in range(size):
        a, p, n = batch_idx[i, 1::2]

        if (batch_idx[i, 0::2] == mask[0]).all():
            row = iii_cmp_fn(
                inputs[a], inputs[p], inputs[n], rank_map=rank_map, lookup=lookup
            )[1:]
        elif (batch_idx[i, 0::2] == mask[1]).all():
            row = ccc_cmp_fn(
                configs[a], configs[p], configs[n], rank_map=rank_map, lookup=lookup
            )[1:]
        elif (batch_idx[i, 0::2] == mask[2]).all():
            row = icc_cmp_fn(inputs[a], configs[p], configs[n], rank_map=rank_map)[1:]
        elif (batch_idx[i, 0::2] == mask[3]).all():
            row = cii_cmp_fn(configs[a], inputs[p], inputs[n], rank_map=rank_map)[1:]
        else:
            raise Exception("Something went wrong")

        batch_idx[:, 1::2] = row

    return batch_idx


# %%

## Validation functions

# These functions work on both the embedding space and the original feature space.
# Requirements:
# - The input/config representation must be floating point torch tensor.
# - `rank_arr` and `regret_arr` must be (I, C) tensors mapping the input idx x config idx to the performance measure

# TODO Make functions for rank_arr and regret_arr or let them work on the dataframes directly
# TODO Allow multiple performance measures
# TODO Move evaluation functions to separate file once they are stable


# rank_map -> IxCxP matrix
def top_k_closest_euclidean(emb1, emb2=None, k=5):
    if emb2 is None:
        distance = torch.cdist(emb1, emb1, p=2)
        distance.fill_diagonal_(distance.max() + 1)
    else:
        distance = torch.cdist(emb1, emb2, p=2)

    return torch.topk(distance, k, largest=False, dim=1).indices


def top_k_closest_cosine(emb1, emb2, k):
    emb1_norm = F.normalize(emb1, p=2, dim=1)
    emb2_norm = F.normalize(emb2, p=2, dim=1)

    # Calculate cosine similarity (dot product of unit vectors)
    similarity = torch.mm(emb1_norm, emb2_norm.t())
    return torch.topk(similarity, k, largest=True, dim=1).indices


# TODO Mapping from embeddings to correct inputs/configs
def evaluate_ic(input_representation, config_representation, rank_arr, regret_arr, k):
    """Evaluation of the input-configuration mapping.

    For each input, we look-up the `k` closest configurations in the representation space.
    Among them we evaluate the best and average performance in terms and rank and regret.
    """
    # For each input, we query the k closest configurations
    # We determine their rank against the measured data and the regret
    # We return the best and the mean value
    top_cfg = top_k_closest_euclidean(input_representation, config_representation, k=k)

    # Ranks
    cfg_ranks = torch.gather(rank_arr, 1, top_cfg).float()
    best_rank = cfg_ranks.min(axis=1)[0].mean()
    avg_rank = cfg_ranks.mean(axis=1).mean()

    # Regret
    cfg_regret = torch.gather(regret_arr, 1, top_cfg).float()
    best_regret = cfg_regret.min(axis=1)[0].mean()
    avg_regret = cfg_regret.mean(axis=1).mean()

    return best_rank, avg_rank, best_regret, avg_regret


# TODO Check this works
# Disabled, we do not have a good use case for this yet - it's the inverse of ic
# def evaluate_ci(input_representation, config_representation, rank_arr, regret_arr, k):
#     # For each configuration, we query the k closest inputs
#     # We determine their rank against the measured data and the regret
#     # We return the best and the mean value
#     top_inp = top_k_closest_euclidean(config_representation, input_representation, k=k)

#     # Ranks
#     inp_ranks = torch.gather(rank_arr.t, 1, top_inp).float()
#     best_rank = inp_ranks.min(axis=1)[0].mean()
#     avg_rank = inp_ranks.mean(axis=1).mean()

#     # regret
#     inp_regret = torch.gather(regret_arr.t, 1, top_inp).float()
#     best_regret = inp_regret.min(axis=1)[0].mean()
#     avg_regret = inp_regret.mean(axis=1).mean()

#     return best_rank, avg_rank, best_regret, avg_regret


# TODO Maybe this is wrong, results do not change during training
def evaluate_ii(
    input_representation, rank_arr, regret_arr, n_neighbors, n_recs=[1, 3, 5]
):
    """
    Evaluation of the input representations.

    For each input, we look-up the `n_neighbors` closest inputs in the representation space.
    We evaluate their rank by:
    - The average rank/regret they have for their top `n_recs` configurations
    """
    top_inp = top_k_closest_euclidean(input_representation, k=n_neighbors)

    ranks = []
    regrets = []

    rank_aggregation = rank_arr[top_inp].float().mean(axis=1)
    regret_aggregation = regret_arr[top_inp].float().mean(axis=1)

    for r in n_recs:
        # Ranks
        avg_cfg_ranks = torch.topk(
            rank_aggregation, k=r, dim=1, largest=False
        ).indices.float()
        best_rank = avg_cfg_ranks.min(axis=1)[0].mean()
        # avg_rank = avg_cfg_ranks.mean(axis=1).mean()

        # regret
        avg_cfg_regret = torch.topk(
            regret_aggregation, k=r, dim=1, largest=False
        ).values.float()
        best_regret = avg_cfg_regret.min(axis=1)[0].mean()
        # avg_regret = avg_cfg_regret.mean(axis=1).mean()

        ranks.append(best_rank)
        regrets.append(best_regret)

    return torch.tensor(ranks), torch.tensor(regrets)


def evaluate_cc(
    config_representation, rank_arr, regret_arr, n_neighbors, n_recs=[1, 3, 5]
):
    """
    Evaluation of the configuration representations.

    For each configuration, we look-up the `n_neighbors` closest configurations in the representation space.
    We evaluate their rank/regret by:
    - Their top `n_recs` inputs; calculated from the average over all inputs.

    n_recs is a parameter for the stability of the configuration..
    """
    # TODO Aggregate, then rank vs Rank, then aggregate/vote?
    # Now, we first aggregate the config ranks over all inputs, then we take the top r inputs
    # Alternatively, we can take the top r inputs and calculate the size of the union (|S|=r => 1, |S|=r*k => 0)
    # The alternative is more a consistency metric than a performance metric
    # Probably we need both. The first for the regret, the rank is somehow interesting too, and the consistency for stability

    # (C, n_neighbors)
    top_cfg = top_k_closest_euclidean(config_representation, k=n_neighbors)

    ranks = []
    regrets = []

    # TODO I'm not sure this indexing is actually correct
    #
    rank_aggregation = rank_arr[:, top_cfg].float().mean(axis=0)
    regret_aggregation = regret_arr[:, top_cfg].float().mean(axis=0)

    for r in n_recs:
        # Ranks
        avg_cfg_ranks = torch.topk(
            rank_aggregation, k=r, dim=0, largest=False
        ).indices.float()
        best_rank = avg_cfg_ranks.min(axis=1)[0].mean()
        # avg_rank = avg_cfg_ranks.mean(axis=1).mean()

        # regret
        avg_cfg_regret = torch.topk(
            regret_aggregation, k=r, dim=1, largest=False
        ).values.float()
        best_regret = avg_cfg_regret.min(axis=1)[0].mean()
        # avg_regret = avg_cfg_regret.mean(axis=1).mean()

        ranks.append(best_rank)
        regrets.append(best_regret)

    return torch.tensor(ranks), torch.tensor(regrets)


# %%

num_input_features = train_input_arr.shape[1]
num_config_features = train_config_arr.shape[1]
emb_size = 24
batch_size = 512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_emb = nn.Sequential(
    nn.Linear(num_input_features, 64), nn.ReLU(), nn.Linear(64, emb_size)
).to(device)
config_emb = nn.Sequential(
    nn.Linear(num_config_features, 64), nn.ReLU(), nn.Linear(64, emb_size)
).to(device)
perf_predict = nn.Sequential(
    nn.Linear(2 * emb_size, 64),
    nn.ReLU(),
    nn.Linear(64, 1),  # TODO Check with |P| outputs
)

optimizer = torch.optim.AdamW(
    list(input_emb.parameters()) + list(config_emb.parameters()), lr=0.03
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# TODO For our dataset size it is relatively cheap to calculate the embeddings for all inputs and configs.
# We can every few iterations update a full collection and collect the hardest triplets from it.
#
# with torch.no_grad():
#     emb_lookup = torch.empty(
#         (train_input_arr.shape[0] + train_config_arr.shape[0], emb_size)
#     )
#     emb_lookup[: train_input_arr.shape[0]] = input_emb(train_input_arr)
#     emb_lookup[train_input_arr.shape[0] :] = config_emb(train_config_arr)

# For evaluation
rank_arr = torch.from_numpy(
    rank_map.loc[(train_inp, train_cfg), :]
    .reset_index()
    .pivot_table(index="inputname", columns="configurationID", values=performances[0])
    .values
).to(device)
regret_arr = torch.from_numpy(
    error_regret.loc[(train_inp, train_cfg), :]
    .reset_index()
    .pivot_table(index="inputname", columns="configurationID", values=performances[0])
    .values
).to(device)

train_input_arr = train_input_arr.to(device)
train_config_arr = train_config_arr.to(device)

total_loss = 0

# Early stopping
best_loss = 999
best_loss_iter = 90
patience = 400

# LR schedule


for iteration in range(5_000):
    batch = (
        make_batch(train_inp, train_cfg, batch_size, rank_map=rank_map)
        .reshape((-1, 2))
        .to(device)
    )
    input_row = batch[:, 0] == 1
    assert (
        batch.shape[1] == 2
    ), "Make sure to reshape batch to two columns (type, index)"

    optimizer.zero_grad()
    embeddings = torch.empty((batch.shape[0], emb_size), device=device)
    embeddings[input_row] = input_emb(train_input_arr[batch[input_row, 1]])
    embeddings[~input_row] = config_emb(train_config_arr[batch[~input_row, 1]])
    loss = nn.functional.triplet_margin_loss(
        anchor=embeddings[0::3],
        positive=embeddings[1::3],
        negative=embeddings[2::3],
    )
    loss.backward()
    optimizer.step()
    total_loss += loss.cpu().item()

    if iteration > 0 and iteration % 10 == 0:
        total_loss /= 10
        scheduler.step(total_loss)

        with torch.no_grad():
            inputembs = input_emb(train_input_arr)
            icc_best_rank, icc_avg_rank, icc_best_regret, icc_avg_regret = evaluate_ic(
                inputembs,
                config_emb(train_config_arr),
                rank_arr,
                regret_arr,
                k=5,
            )
            iii_ranks, iii_regret = evaluate_ii(
                inputembs,
                rank_arr,
                regret_arr,
                n_neighbors=5,
                n_recs=[1, 3, 5],
            )
            print(
                f"{iteration} "
                + f"l:{total_loss:.3f} | "
                + f"icc(rank(best:{icc_best_rank:.2f} "
                + f"avg:{icc_avg_rank:.2f}) "
                + f"regret(best:{icc_best_regret:.2f} "
                + f"avg:{icc_avg_regret:.2f})) | "
                + f"iii(rank({iii_ranks.numpy().round(2)} "
                + f"regret({iii_regret.numpy().round(2)}) "
            )

            if total_loss < best_loss:
                best_loss_iter = iteration
                best_loss = total_loss
            elif (iteration - best_loss_iter) >= patience:
                print(
                    f"No loss improvement since {patience} iterations. Stop training."
                )
                break

            total_loss = 0


# %%

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from main import load_all_csv
import numpy as np
import os
import datetime
from scipy import stats


def format_time(s):
    dtf = datetime.time.fromisoformat("00:0" + s.strip())
    return dtf.minute * 60 + dtf.second + dtf.microsecond / 1_000_000


def load_x264(data_dir="../data/"):
    input_columns = [
        "category",
        "resolution",
        "WIDTH",
        "HEIGHT",
        "SPATIAL_COMPLEXITY",
        "TEMPORAL_COMPLEXITY",
        "CHUNK_COMPLEXITY_VARIATION",
        "COLOR_COMPLEXITY",
    ]
    input_columns_cat = ["category"]
    input_columns_cont = [s for s in input_columns if s not in input_columns_cat]

    config_columns = [
        "cabac",
        "ref",
        "deblock",
        "analyse",
        "me",
        "subme",
        "mixed_ref",
        "me_range",
        "trellis",
        "8x8dct",
        "fast_pskip",
        "chroma_qp_offset",
        "bframes",
        "b_pyramid",
        "b_adapt",
        "direct",
        "weightb",
        "open_gop",
        "weightp",
        "scenecut",
        "rc_lookahead",
        "mbtree",
        "qpmax",
        "aq-mode",
    ]
    config_columns_cat = [
        "analyse",
        "me",
        "direct",
        "deblock",
        "b_adapt",
        "b_pyramid",
        "open_gop",
        "rc_lookahead",
        "scenecut",
        "weightb",
    ]
    config_columns_cont = [s for s in config_columns if s not in config_columns_cat]

    all_performances = [
        # "rel_size",  # after preprocessing, before just `size` - min
        "usertime",
        "systemtime",
        "elapsedtime",
        "cpu",
        # "frames",  # irrelevant?
        "fps",  # max
        "kbs",
        "rel_kbs",
    ]
    # True - higher is better
    # increasing_performances = {
    #     "rel_size": False,
    #     "usertime": False,
    #     "systemtime": False,
    #     "elapsedtime": False,
    #     "cpu": False,
    #     "fps": True,
    #     "kbs": False,
    #     "rel_kbs": False,
    # }

    # metadata = {}

    meas_matrix, _ = load_all_csv(
        os.path.join(data_dir, "res_ugc"), ext="csv", with_names=True
    )
    meas_matrix["elapsedtime"] = meas_matrix["elapsedtime"].apply(format_time)
    input_properties = pd.read_csv(
        os.path.join(data_dir, "res_ugc_properties.csv")
    )  # Does not match all inputs from perf_matrix?
    del input_properties["id"]

    perf_matrix = pd.merge(
        meas_matrix, input_properties, left_on="inputname", right_on="name"
    ).sort_values(by=["inputname", "configurationID"])
    del perf_matrix["name"]
    # perf_matrix["rel_size"] = perf_matrix["size"] / perf_matrix["ORIG_SIZE"]  # We have `kbs` which is a better alternative
    # perf_matrix["rel_size"] = np.log(perf_matrix["rel_size"])  # To scale value distribution more evenly
    perf_matrix["rel_kbs"] = perf_matrix["kbs"] / perf_matrix["ORIG_BITRATE"]
    perf_matrix["fps"] = -perf_matrix[
        "fps"
    ]  # fps is the only increasing performance measure

    input_features = (
        perf_matrix[["inputname"] + input_columns_cont]
        .join(pd.get_dummies(perf_matrix[input_columns_cat]))
        .set_index("inputname")
        .drop_duplicates()
    )

    config_features = (
        perf_matrix[["configurationID"] + config_columns_cont]
        .join(pd.get_dummies(perf_matrix[config_columns_cat]))
        .set_index("configurationID")
        .drop_duplicates()
    )

    return perf_matrix, input_features, config_features, all_performances


def split_data(perf_matrix, test_size=0.15, verbose=True, random_state=None):
    # We set aside 15% of configurations and 15% of inputs as test data
    # This gives us 4 sets of data, of which we set 3 aside for testing
    train_cfg, test_cfg = train_test_split(
        perf_matrix["configurationID"].unique(),
        test_size=test_size,
        random_state=random_state,
    )
    train_inp, test_inp = train_test_split(
        perf_matrix["inputname"].unique(),
        test_size=test_size,
        random_state=random_state,
    )
    train_cfg.sort()
    test_cfg.sort()
    train_inp.sort()
    test_inp.sort()
    train_data = perf_matrix[
        perf_matrix["configurationID"].isin(train_cfg)
        & perf_matrix["inputname"].isin(train_inp)
    ]
    test_cfg_new = perf_matrix[
        perf_matrix["configurationID"].isin(test_cfg)
        & perf_matrix["inputname"].isin(train_inp)
    ]
    test_inp_new = perf_matrix[
        perf_matrix["configurationID"].isin(train_cfg)
        & perf_matrix["inputname"].isin(test_inp)
    ]
    test_both_new = perf_matrix[
        perf_matrix["configurationID"].isin(test_cfg)
        & perf_matrix["inputname"].isin(test_inp)
    ]
    assert (
        test_cfg_new.shape[0]
        + test_inp_new.shape[0]
        + test_both_new.shape[0]
        + train_data.shape[0]
        == perf_matrix.shape[0]
    )

    if verbose:
        print(f"Training data: {100*train_data.shape[0]/perf_matrix.shape[0]:.2f}%")
        print(f"Both new: {100*test_both_new.shape[0]/perf_matrix.shape[0]:.2f}%")
        print(f"Config new: {100*test_cfg_new.shape[0]/perf_matrix.shape[0]:.2f}%")
        print(f"Input new: {100*test_inp_new.shape[0]/perf_matrix.shape[0]:.2f}%")

    return {
        "train_cfg": train_cfg,
        "test_cfg": test_cfg,
        "train_inp": train_inp,
        "test_inp": test_inp,
        "train_data": train_data,
        "test_data_cfg_new": test_cfg_new,
        "test_data_inp_new": test_inp_new,
        "test_data_both_new": test_both_new,
    }


## Rank-based distance calculation via correlation


# Function to calculate Pearson correlation coefficient in a vectorized manner
def pearson_correlation(X, Y):
    mean_X = np.mean(X, axis=-1, keepdims=True)
    mean_Y = np.mean(Y, axis=-1, keepdims=True)
    numerator = np.sum((X - mean_X) * (Y - mean_Y), axis=-1)
    denominator = np.sqrt(
        np.sum((X - mean_X) ** 2, axis=-1) * np.sum((Y - mean_Y) ** 2, axis=-1)
    )
    return numerator / denominator


def spearman_rank_distance(measurements):
    # Vectorized spearmanr with multiple measurements

    # Breaks ties correctly by assigning same ranks, but potentially instable
    # TODO Should work correctly if we drop `frames` which has constant value
    # ranks = stats.rankdata(measurements, axis=1, method="min")

    # Makes individual ranks
    ranks = np.argsort(measurements, axis=1)

    # The ranks array is 3D (A, B, C), and we need to expand it to 4D for pairwise comparison in A,
    # while keeping C
    expanded_rank_X_3d = ranks[:, np.newaxis, :, :]  # Expanding for A dimension
    expanded_rank_Y_3d = ranks[np.newaxis, :, :, :]  # Expanding for A dimension

    A = ranks.shape[0]
    C = ranks.shape[2]

    # Initialize the Spearman correlation matrix for each C
    spearman_correlation_matrix_3d = np.empty((A, A, C))

    # Calculate Spearman correlation matrix for each C
    for c in range(C):
        spearman_correlation_matrix_3d[:, :, c] = pearson_correlation(
            expanded_rank_X_3d[:, :, :, c], expanded_rank_Y_3d[:, :, :, c]
        )

    return spearman_correlation_matrix_3d


def rank_difference_distance(measurements):
    ranks = np.argsort(measurements, axis=1)
    expanded_ranks = ranks[:, np.newaxis, :, :] - ranks[np.newaxis, :, :, :]

    # Calculate the absolute differences and sum along the B dimension
    vectorized_distance_matrix = np.sum(np.abs(expanded_ranks), axis=2)
    return vectorized_distance_matrix


def stat_distance(measurements, stats_fn):
    ranks = np.argsort(measurements, axis=1)
    A = ranks.shape[0]
    C = ranks.shape[2]

    distance_matrix = np.zeros((A, A, C))

    # There is no good vectorized version to apply,
    # therefore we loop over all dimensions...
    for c in range(C):
        for i in range(A):
            for j in range(i + 1, A):
                try:
                    res = stats_fn(ranks[i, :, c], ranks[j, :, c])
                    stat, p_value = res.statistic, res.pvalue

                    distance_matrix[i, j, c] = stat
                    distance_matrix[j, i, c] = stat
                except ValueError:
                    # Mark as NaN in case of any other ValueError
                    distance_matrix[i, j, c] = np.nan
                    distance_matrix[j, i, c] = np.nan

    return distance_matrix


def kendalltau_distance(measurements):
    # We clip negative values to 0, because we see them as different
    # We invert such that lower values indicate higher correspondence
    return 1 - (np.maximum(stat_distance(measurements, stats_fn=stats.kendalltau), 0))


def wilcoxon_distance(measurements):
    return stat_distance(measurements, stats_fn=stats.wilcoxon)


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


## Evaluation functions

# These functions work on both the embedding space and the original feature space.
# Requirements:
# - The input/config representation must be floating point torch tensor.
# - `rank_arr` and `regret_arr` must be (I, C) tensors mapping the input idx x config idx to the performance measure

# TODO Make functions for rank_arr and regret_arr or let them work on the dataframes directly
# TODO Allow multiple performance measures
# TODO Move evaluation functions to separate file once they are stable

# rank_map -> IxCxP matrix


def evaluate_ic(
    input_representation,
    config_representation,
    rank_arr,
    regret_arr,
    k,
    distance="euclidean",
):
    """Evaluation of the input-configuration mapping.

    For each input, we look-up the `k` closest configurations in the representation space.
    Among them we evaluate the best and average performance in terms and rank and regret.
    """
    # TODO Mapping from embeddings to correct inputs/configs
    # For each input, we query the k closest configurations
    # We determine their rank against the measured data and the regret
    # We return the best and the mean value
    if distance == "euclidean":
        top_cfg = top_k_closest_euclidean(
            input_representation, config_representation, k=k
        )
    elif distance == "cosine":
        top_cfg = top_k_closest_cosine(input_representation, config_representation, k=k)

    # Ranks
    cfg_ranks = torch.gather(rank_arr, 1, top_cfg).float()
    best_rank = cfg_ranks.min(axis=1)[0].mean()
    avg_rank = cfg_ranks.mean(axis=1).mean()

    # Regret
    cfg_regret = torch.gather(regret_arr, 1, top_cfg).float()
    best_regret = cfg_regret.min(axis=1)[0].mean()
    avg_regret = cfg_regret.mean(axis=1).mean()

    return best_rank, avg_rank, best_regret, avg_regret


def evaluate_ii(
    input_representation, rank_arr, regret_arr, n_neighbors, n_recs=[1, 3, 5], input_mask=None
):
    """
    Evaluation of the input representations.

    For each input, we look-up the `n_neighbors` closest inputs in the representation space.
    We evaluate their rank by:
    - The average rank/regret they have for their top `n_recs` configurations
    """
    # TODO Maybe this is wrong, results do not change during training
    top_inp = top_k_closest_euclidean(input_representation, k=n_neighbors)

    if input_mask is not None:
        top_inp = top_inp[input_mask]

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


def evaluate_cc(config_representation, rank_arr, n_neighbors, n_recs=[1, 3, 5], config_mask=None):
    """
    Evaluation of the configuration representations.

    For each configuration, we look-up the `n_neighbors` closest configurations in the representation space.
    We evaluate the stability of configurations by:
    - The shared number of r affected inputs from 0 (no shared inputs) to 1 (all r inputs shared)

    n_recs is a parameter for the stability of the configuration.
    """
    if n_neighbors == 1:
        return torch.ones((len(n_recs)))

    # (C, n_neighbors)
    top_cfg = top_k_closest_euclidean(config_representation, k=n_neighbors)

    if config_mask is not None:
        top_cfg = top_cfg[config_mask]

    rank_aggregation = rank_arr[:, top_cfg].permute([1, 0, 2])
    assert rank_aggregation.shape == (
        top_cfg.shape[0],
        rank_arr.shape[0],
        n_neighbors,
    )

    share_ratios = torch.empty((len(n_recs)))
    n_cfg = top_cfg.shape[0]

    for i, r in enumerate(n_recs):
        # We must have at least r x num configs unique elements
        count_offset = n_cfg * r

        topinds = torch.topk(rank_aggregation, k=r, dim=1, largest=False).indices
        uniq_vals = torch.tensor([torch.unique(row).numel() for row in topinds])
        share_ratios[i] = 1 - (torch.sum(uniq_vals)-count_offset) / (topinds.numel()-count_offset)

    return share_ratios

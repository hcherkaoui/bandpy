""" Define all check utility functions in Bandpy. """
# Authors: Hamza Cherkaoui <hamza.cherkaoui@huawei.com>

import os
import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.cluster import KMeans

from ..utils import fill_mising_values, check_random_state


MAX_K = 30
N_FEATURES_YAHOO = 700


def _movie_lens_loader(dirname, N, K, d, seed=None):
    """Load and preprocess the internal MovieLens dataset."""
    assert K <= MAX_K, f"Maximum number of arms is {MAX_K}, got {K}"

    rng = check_random_state(seed)

    movies_filename = os.path.join(dirname, "movies.csv")
    ratings_filename = os.path.join(dirname, "ratings.csv")
    movies = pd.read_csv(movies_filename, sep=",")
    ratings = pd.read_csv(ratings_filename, sep=",")

    ratings.drop(columns="timestamp", inplace=True)

    # reindex randomly users
    old_user_id = np.unique(ratings.userId.astype(int))
    new_user_id = np.arange(len(old_user_id))
    new_user_id = rng.permutation(new_user_id)
    ratings.userId = ratings.userId.map(dict(zip(old_user_id, new_user_id)))

    col_name = {"userId": "user_id", "movieId": "item_id", "rating": "rating"}
    ratings = fill_mising_values(ratings, K=K, col_name=col_name)
    ratings = ratings[ratings.user_id < N]
    ratings = ratings[ratings.item_id < K]
    ratings.reset_index(inplace=True, drop=True)

    for i, labels in enumerate(movies["genres"].str.split("|")):
        for label in labels:
            if label not in movies.columns:
                movies[label] = np.zeros(movies.shape[0], dtype=float)
            movies.loc[i, label] = 1.0
    movies = movies.drop(["genres", "title"], axis=1)
    movies = movies[movies.movieId <= K]  # start on 1

    col_name = {"user_id": "agent_id", "item_id": "arm_id", "rating": "reward"}
    data = ratings.rename(columns=col_name)

    arms = [movies.iloc[i, 1:].to_numpy()[:d] for i in range(K)]

    agent_i_to_env_agent_i = dict()
    for i, id in enumerate(np.unique(data.agent_id)):
        agent_i_to_env_agent_i[f"agent_{i}"] = id

    return data, arms, agent_i_to_env_agent_i


def _yahoo_loader(dirname, N, K, d, n_clusters_k_means=100, seed=None):
    """Load the internal Yahoo dataset."""
    assert K <= MAX_K, f"Maximum number of arms is {MAX_K}, got {K}"

    rng = check_random_state(seed)

    def _format_line(entry):
        """Format entry to numeric."""
        if ":" in entry:
            x_id, x, *_ = entry.split(":")

            if x_id == "qid":
                return 1, int(x)

            else:
                return int(x_id), float(x)

        else:
            return 0, int(entry)

    # get filename (set2.train.txt is the lightest one)
    filename = os.path.join(dirname, "set2.train.txt")

    # manually load data
    all_raw_lines = []
    with open(filename, "r") as f:
        for raw_line in f:
            # When a feature is undefined for a set, its value is 0
            # (from Yahoo! Learning to Rank Challenge Overview)
            raw_lines = np.zeros(N_FEATURES_YAHOO + 2)
            for entry in raw_line.split(" "):
                i, x = _format_line(entry)
                raw_lines[i] = x
            all_raw_lines.append(raw_lines)
    all_raw_lines = np.c_[all_raw_lines]

    # define internal dataset
    data = pd.DataFrame(all_raw_lines)

    # rename columns
    columns = ["rating", "user_id"]
    for i in np.arange(data.shape[1])[1:]:
        columns += [f"{i}"]
    data.rename(columns=dict(zip(data.columns, columns)), inplace=True)

    # reduce the number of consulted documents (arms) with k-means
    filter_arms = (data.columns != "rating") & (data.columns != "user_id")
    X_train = np.unique(data.loc[:, filter_arms], axis=0)
    kmeans = KMeans(n_clusters=n_clusters_k_means, n_init='auto').fit(X_train)
    X_test = data.loc[:, filter_arms].to_numpy()
    data.loc[:, "item_id"] = kmeans.predict(X_test)

    # retain only the user-arm-reward informations
    data = data[["rating", "user_id", "item_id"]]

    # force user_id as int
    data.loc[:, "user_id"] = data.loc[:, "user_id"].astype(int)

    # drop duplicate
    data = data.drop_duplicates(["user_id", "item_id"])

    # fill missing user-arm-reward information
    col_name = {"user_id": "user_id", "item_id": "item_id", "rating": "rating"}
    data = fill_mising_values(data, K=K, col_name=col_name)

    # reduce dimensionality
    user_id_retained = rng.choice(np.unique(data.user_id), size=N, replace=False)
    data = data.loc[data.user_id.isin(user_id_retained)]
    data = data.loc[data.item_id < K]

    # reset index
    data.reset_index(inplace=True, drop=True)

    # standardize column names
    col_name = {"user_id": "agent_id", "item_id": "arm_id", "rating": "reward"}
    data = data.rename(columns=col_name)

    arms = [kmeans.cluster_centers_[label_].flatten()[:d] for label_ in range(K)]

    # register the agent_id dataset-env mapping
    agent_i_to_env_agent_i = dict()
    for i, id in enumerate(np.unique(data.agent_id)):
        agent_i_to_env_agent_i[f"agent_{i}"] = id

    return data, arms, agent_i_to_env_agent_i


movie_lens_loader = Memory("__cache__", verbose=0).cache(_movie_lens_loader)
yahoo_loader = Memory("__cache__", verbose=0).cache(_yahoo_loader)

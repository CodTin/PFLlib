import json
import logging
import os
import random

import numpy as np

from pfllib.data.language_utils import letter_to_index, word_to_indices

logger = logging.getLogger(__name__)

random.seed(1)
np.random.seed(1)

data_path_train = "utils/LEAF/data/shakespeare/data/train/all_data_niid_2_keep_0_train_9.json"
data_path_test = "utils/LEAF/data/shakespeare/data/test/all_data_niid_2_keep_0_test_9.json"


# https://github.com/TalwalkarLab/leaf/blob/master/models/shakespeare/stacked_lstm.py#L40
def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = np.array(x_batch)
    return x_batch


def process_y(raw_y_batch):
    y_batch = [letter_to_index(c) for c in raw_y_batch]
    y_batch = np.array(y_batch)
    return y_batch


# Allocate data to users
def generate_dataset(dir_path):
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    with open(data_path_train) as f:
        raw_train_data = json.load(f)
    with open(data_path_test) as f:
        raw_test_data = json.load(f)

    train_data_ = []
    train_data_len = []
    test_data_ = []

    for k, v in raw_train_data["user_data"].items():
        train_data_.append({"x": process_x(v["x"]), "y": process_y(v["y"])})
        train_data_len.append(len(train_data_[-1]["x"]))
    for k, v in raw_test_data["user_data"].items():
        test_data_.append({"x": process_x(v["x"]), "y": process_y(v["y"])})

    train_data = []
    test_data = []

    inds = sorted(range(len(train_data_len)), key=lambda k: train_data_len[k])
    for ind in inds:
        train_data.append(train_data_[ind])
        test_data.append(test_data_[ind])

    logger.info(f"Number of clients: {len(train_data)}")
    logger.info("Saving to disk.")

    for idx, train_dict in enumerate(train_data):
        with open(train_path + str(idx) + ".npz", "wb") as f:
            np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + ".npz", "wb") as f:
            np.savez_compressed(f, data=test_dict)

    logger.info("Finish generating dataset.")


if __name__ == "__main__":
    dir_path = "Shakespeare/"
    generate_dataset(dir_path)

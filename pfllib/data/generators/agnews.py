import logging
import os
import random
import sys

import numpy as np
import torchtext

from pfllib.data.language_utils import tokenizer
from pfllib.data.utils import check, save_file, separate_data, split_data

logger = logging.getLogger(__name__)

random.seed(1)
np.random.seed(1)

max_len = 200
max_tokens = 32000


# Allocate data to users
def generate_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        return

    # Get AG_News data
    trainset, testset = torchtext.datasets.AG_NEWS(root=dir_path + "rawdata")

    trainlabel, traintext = list(zip(*trainset))
    testlabel, testtext = list(zip(*testset))

    dataset_text = []
    dataset_label = []

    dataset_text.extend(traintext)
    dataset_text.extend(testtext)
    dataset_label.extend(trainlabel)
    dataset_label.extend(testlabel)

    num_classes = len(set(dataset_label))
    logger.info(f"Number of classes: {num_classes}")

    vocab, text_list = tokenizer(dataset_text, max_len, max_tokens)
    label_pipeline = lambda x: int(x) - 1
    label_list = [label_pipeline(l) for l in dataset_label]

    text_lens = [len(text) for text in text_list]
    text_list = [(text, l) for text, l in zip(text_list, text_lens)]

    text_list = np.array(text_list, dtype=object)
    label_list = np.array(label_list)

    # dataset = []
    # for i in range(num_classes):
    #     idx = label_list == i
    #     dataset.append(text_list[idx])

    X, y, statistic = separate_data((text_list, label_list), num_clients, num_classes, niid, balance, partition)
    train_data, test_data = split_data(X, y)
    save_file(
        config_path,
        train_path,
        test_path,
        train_data,
        test_data,
        num_clients,
        num_classes,
        statistic,
        niid,
        balance,
        partition,
    )

    logger.info(f"The size of vocabulary: {len(vocab)}")


if __name__ == "__main__":
    dir_path = "AGNews/"
    num_clients = 20
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_dataset(dir_path, num_clients, niid, balance, partition)

import logging
import os
import random

import numpy as np
import torchvision.transforms as transforms
from wilds import get_dataset

from pfllib.data.utils import save_file, split_data

logger = logging.getLogger(__name__)

random.seed(1)
np.random.seed(1)

img_size = 64
least_samples = 100
least_class_per_client = 2


# Allocate data to users
def generate_dataset(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    # Get data
    dataset = get_dataset(dataset="iwildcam", root_dir=dir_path + "rawdata", download=True)
    logger.info(f"metadata columns: {dataset.metadata_fields}")

    transform = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

    dataset_train = dataset.get_subset("train")
    dataset_val = dataset.get_subset("val")
    dataset_test = dataset.get_subset("test")

    num_clients = 323
    X_o = [[] for _ in range(num_clients)]
    y_o = [[] for _ in range(num_clients)]
    statistic = []

    def reassign_data(dataset):
        for xx, yy, meta_data in dataset:
            camera_trap_id = meta_data[0].item()
            X_o[camera_trap_id].append(transform(xx).cpu().numpy())
            y_o[camera_trap_id].append(yy.item())

    reassign_data(dataset_train)
    reassign_data(dataset_val)
    reassign_data(dataset_test)

    X = []
    y = []

    class_map = {}
    class_cnt = 0

    for i in range(num_clients):
        if len(y_o[i]) > least_samples and len(set(y_o[i])) > least_class_per_client:
            X.append(X_o[i])
            for yy in y_o[i]:
                if yy not in class_map:
                    class_map[yy] = class_cnt
                    class_cnt += 1
            y.append([class_map[yy] for yy in y_o[i]])

    num_clients = len(y)
    logger.info(f"Number of clients: {num_clients}")
    logger.info(f"Number of classes: {class_cnt}")

    for i in range(num_clients):
        statistic.append([])
        y_arr = np.array(y[i])
        for yc in sorted(np.unique(y_arr)):
            statistic[-1].append((int(yc), int(sum(y_arr == yc))))

    for i in range(num_clients):
        logger.info(f"Client {i}\t Size of data: {len(X[i])}\t Labels: {np.unique(y[i])}")
        logger.info(f"\t\t Samples of labels: {[i for i in statistic[i]]}")
        logger.info("-" * 50)

    train_data, test_data = split_data(X, y)
    save_file(
        config_path, train_path, test_path, train_data, test_data, num_clients, class_cnt, statistic, None, None, None
    )


if __name__ == "__main__":
    dir_path = "iWildCam/"
    generate_dataset(dir_path)

from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
import torch


def read_data(
    dataset: str,
    idx: int,
    is_train: bool,
    data_dir: Union[str, Path] = "data",
) -> dict:
    split = "train" if is_train else "test"
    data_dir = Path(data_dir)
    file = data_dir / dataset / split / f"{idx}.npz"
    with open(file, "rb") as f:
        data = np.load(f, allow_pickle=True)["data"].tolist()
    return data


def read_client_data(
    dataset: str,
    idx: int,
    is_train: bool = True,
    few_shot: int = 0,
    data_dir: Union[str, Path] = "data",
) -> list:
    data = read_data(dataset, idx, is_train, data_dir)
    if "News" in dataset:
        data_list = process_text(data)
    elif "Shakespeare" in dataset:
        data_list = process_shakespeare(data)
    else:
        data_list = process_image(data)

    if is_train and few_shot > 0:
        shot_cnt_dict: dict[int, int] = defaultdict(int)
        data_list_new = []
        for data_item in data_list:
            label = data_item[1].item()
            if shot_cnt_dict[label] < few_shot:
                data_list_new.append(data_item)
                shot_cnt_dict[label] += 1
        data_list = data_list_new
    return data_list


def process_image(data: dict) -> list:
    X = torch.Tensor(data["x"]).type(torch.float32)
    y = torch.Tensor(data["y"]).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]


def process_text(data: dict) -> list:
    X_tuple, X_lens_tuple = list(zip(*data["x"]))
    y = data["y"]
    X_tensor = torch.Tensor(X_tuple).type(torch.int64)
    X_lens_tensor = torch.Tensor(X_lens_tuple).type(torch.int64)
    y_tensor = torch.Tensor(data["y"]).type(torch.int64)
    return [((x, lens), label) for x, lens, label in zip(X_tensor, X_lens_tensor, y_tensor)]


def process_shakespeare(data: dict) -> list:
    X = torch.Tensor(data["x"]).type(torch.int64)
    y = torch.Tensor(data["y"]).type(torch.int64)
    return [(x, y) for x, y in zip(X, y)]

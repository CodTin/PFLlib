import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)


def average_data(
    algorithm: str = "",
    dataset: str = "",
    goal: str = "",
    times: int = 10,
    results_dir: Path = Path("results"),
):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times, results_dir)

    max_accuracy = []
    for i in range(times):
        max_accuracy.append(test_acc[i].max())

    logger.info(f"std for best accuracy: {np.std(max_accuracy)}")
    logger.info(f"mean for best accuracy: {np.mean(max_accuracy)}")


def get_all_results_for_one_algo(
    algorithm: str = "",
    dataset: str = "",
    goal: str = "",
    times: int = 10,
    results_dir: Path = Path("results"),
):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, results_dir, delete=False)))

    return test_acc


def read_data_then_delete(file_name: str, results_dir: Path = Path("results"), delete: bool = False):
    file_path = results_dir / f"{file_name}.h5"

    with h5py.File(file_path, "r") as hf:
        rs_test_acc = np.array(hf.get("rs_test_acc"))

    if delete:
        file_path.unlink()
    logger.info(f"Length: {len(rs_test_acc)}")

    return rs_test_acc

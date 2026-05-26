import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

# ─── Config ───────────────────────────────────────────────────────────────────


class TestExperimentConfig:
    def test_default_values(self):
        from pfllib.config import ExperimentConfig

        config = ExperimentConfig()
        assert config.dataset == "MNIST"
        assert config.algorithm == "FedAvg"
        assert config.model == "CNN"
        assert config.num_classes == 10
        assert config.num_clients == 20
        assert config.global_rounds == 2000
        assert config.local_epochs == 1
        assert config.batch_size == 10
        assert config.local_learning_rate == 0.005
        assert config.device == "cuda"
        assert config.join_ratio == 1.0
        assert config.data_dir == Path("data")
        assert config.results_dir == Path("results")
        assert config.outputs_dir == Path("outputs")
        assert config.mu == 0.0
        assert config.alpha == 1.0

    def test_custom_values(self):
        from pfllib.config import ExperimentConfig

        config = ExperimentConfig(
            dataset="Cifar10",
            algorithm="FedProx",
            model="ResNet18",
            num_clients=50,
            global_rounds=500,
            mu=0.01,
        )
        assert config.dataset == "Cifar10"
        assert config.algorithm == "FedProx"
        assert config.model == "ResNet18"
        assert config.num_clients == 50
        assert config.global_rounds == 500
        assert config.mu == 0.01

    def test_extra_fields_allowed(self):
        from pfllib.config import ExperimentConfig

        config = ExperimentConfig(custom_param=42)
        assert config.custom_param == 42

    def test_path_fields_are_path_objects(self):
        from pfllib.config import ExperimentConfig

        config = ExperimentConfig()
        assert isinstance(config.data_dir, Path)
        assert isinstance(config.results_dir, Path)
        assert isinstance(config.outputs_dir, Path)

    def test_runtime_model_attachment(self):
        from pfllib.config import ExperimentConfig

        config = ExperimentConfig()
        model = nn.Linear(10, 2)
        config.model_instance = model
        assert config.model_instance is model

    def test_iteration(self):
        from pfllib.config import ExperimentConfig

        config = ExperimentConfig(dataset="Cifar10")
        fields = dict(config)
        assert fields["dataset"] == "Cifar10"
        assert "algorithm" in fields


# ─── Registry ─────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_register_and_get(self):
        from pfllib.registry import _ALGORITHM_REGISTRY, get_algorithm, register_algorithm

        class S:
            pass

        class C:
            pass

        register_algorithm("UnitTestAlgo", client_cls=C, uses_head_split=True)(S)
        s_cls, c_cls, head = get_algorithm("UnitTestAlgo")
        assert s_cls is S
        assert c_cls is C
        assert head is True
        _ALGORITHM_REGISTRY.pop("UnitTestAlgo", None)

    def test_register_model(self):
        from pfllib.registry import _MODEL_REGISTRY, get_model_builder, register_model

        class M:
            pass

        register_model("UnitTestModel")(M)
        assert get_model_builder("UnitTestModel") is M
        _MODEL_REGISTRY.pop("UnitTestModel", None)

    def test_unknown_algorithm_raises(self):
        from pfllib.registry import get_algorithm

        with pytest.raises(KeyError, match="Unknown algorithm"):
            get_algorithm("NoSuchAlgo")

    def test_unknown_model_raises(self):
        from pfllib.registry import get_model_builder

        with pytest.raises(KeyError, match="Unknown model"):
            get_model_builder("NoSuchModel")

    def test_list_algorithms(self):
        from pfllib.registry import _ALGORITHM_REGISTRY, list_algorithms, register_algorithm

        class S:
            pass

        class C:
            pass

        register_algorithm("ZzzListTest", client_cls=C, uses_head_split=False)(S)
        algos = list_algorithms()
        assert "ZzzListTest" in algos
        assert algos == sorted(algos)
        _ALGORITHM_REGISTRY.pop("ZzzListTest", None)

    def test_list_models(self):
        from pfllib.registry import _MODEL_REGISTRY, list_models, register_model

        class M:
            pass

        register_model("ZzzModelTest")(M)
        models = list_models()
        assert "ZzzModelTest" in models
        assert models == sorted(models)
        _MODEL_REGISTRY.pop("ZzzModelTest", None)

    def test_duplicate_registration_overwrites(self):
        from pfllib.registry import _ALGORITHM_REGISTRY, get_algorithm, register_algorithm

        class S1:
            pass

        class S2:
            pass

        class C:
            pass

        register_algorithm("DupTest", client_cls=C, uses_head_split=False)(S1)
        register_algorithm("DupTest", client_cls=C, uses_head_split=True)(S2)
        s_cls, _, head = get_algorithm("DupTest")
        assert s_cls is S2
        assert head is True
        _ALGORITHM_REGISTRY.pop("DupTest", None)


# ─── Logger ───────────────────────────────────────────────────────────────────


class TestLogger:
    def test_get_logger(self):
        from pfllib.logger import get_logger

        logger = get_logger("test.mod")
        assert logger.name == "test.mod"

    def test_setup_logger_level(self):
        import logging

        from pfllib.logger import setup_logger

        setup_logger(level="DEBUG")
        root = logging.getLogger("pfllib")
        assert root.level == logging.DEBUG

    def test_setup_logger_file_handler(self):
        import logging

        from pfllib.logger import setup_logger

        with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
            log_path = Path(f.name)
        try:
            setup_logger(level="INFO", log_file=log_path)
            logger = logging.getLogger("pfllib.test_file")
            logger.info("test message")
            has_file_handler = any(isinstance(h, logging.FileHandler) for h in logging.getLogger("pfllib").handlers)
            assert has_file_handler
        finally:
            for h in list(logging.getLogger("pfllib").handlers):
                if isinstance(h, logging.FileHandler):
                    h.close()
                    logging.getLogger("pfllib").removeHandler(h)
            log_path.unlink(missing_ok=True)


# ─── Data Utils ───────────────────────────────────────────────────────────────


class TestDataUtils:
    def test_split_data(self):
        from pfllib.data.utils import split_data

        X = [np.random.randn(100, 5) for _ in range(3)]
        y = [np.random.randint(0, 10, 100) for _ in range(3)]
        train_data, test_data = split_data(X, y, train_ratio=0.8)
        assert len(train_data) == 3
        assert len(test_data) == 3
        for i in range(3):
            assert train_data[i]["x"].shape[0] + test_data[i]["x"].shape[0] == 100
            assert len(train_data[i]["y"]) + len(test_data[i]["y"]) == 100

    def test_split_data_ratios(self):
        from pfllib.data.utils import split_data

        X = [np.random.randn(1000, 3) for _ in range(2)]
        y = [np.random.randint(0, 5, 1000) for _ in range(2)]
        train_data, test_data = split_data(X, y, train_ratio=0.9)
        for i in range(2):
            assert train_data[i]["x"].shape[0] == 900
            assert test_data[i]["x"].shape[0] == 100

    def test_check_nonexistent(self):
        from pfllib.data.utils import check

        with tempfile.TemporaryDirectory() as tmpdir:
            assert (
                check(
                    os.path.join(tmpdir, "config.json"),
                    os.path.join(tmpdir, "train/"),
                    os.path.join(tmpdir, "test/"),
                    num_clients=20,
                )
                is False
            )

    def test_check_existing_mismatch(self):
        import ujson

        from pfllib.data.utils import check

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "num_clients": 10,
                "non_iid": False,
                "balance": True,
                "partition": None,
                "alpha": 0.1,
                "batch_size": 10,
            }
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                ujson.dump(config, f)
            os.makedirs(os.path.join(tmpdir, "train"))
            os.makedirs(os.path.join(tmpdir, "test"))
            assert (
                check(
                    os.path.join(tmpdir, "config.json"),
                    os.path.join(tmpdir, "train/"),
                    os.path.join(tmpdir, "test/"),
                    num_clients=20,
                )
                is False
            )

    def test_check_existing_match(self):
        import ujson

        from pfllib.data.utils import check

        with tempfile.TemporaryDirectory() as tmpdir:
            config = {
                "num_clients": 20,
                "non_iid": True,
                "balance": False,
                "partition": "dir",
                "alpha": 0.5,
                "batch_size": 32,
            }
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                ujson.dump(config, f)
            os.makedirs(os.path.join(tmpdir, "train"))
            os.makedirs(os.path.join(tmpdir, "test"))
            assert (
                check(
                    os.path.join(tmpdir, "config.json"),
                    os.path.join(tmpdir, "train/"),
                    os.path.join(tmpdir, "test/"),
                    num_clients=20,
                    niid=True,
                    balance=False,
                    partition="dir",
                    alpha=0.5,
                    batch_size=32,
                )
                is True
            )

    def test_save_and_load_file(self):
        from pfllib.data.utils import save_file

        with tempfile.TemporaryDirectory() as tmpdir:
            train_data = [{"x": np.random.randn(50, 5), "y": np.random.randint(0, 3, 50)}]
            test_data = [{"x": np.random.randn(10, 5), "y": np.random.randint(0, 3, 10)}]
            config_path = os.path.join(tmpdir, "config.json")
            train_path = os.path.join(tmpdir, "train/")
            test_path = os.path.join(tmpdir, "test/")
            os.makedirs(train_path)
            os.makedirs(test_path)
            save_file(
                config_path,
                train_path,
                test_path,
                train_data,
                test_data,
                num_clients=1,
                num_classes=3,
                statistic=[[(0, 20), (1, 15), (2, 15)]],
            )
            assert os.path.exists(config_path)
            assert os.path.exists(os.path.join(train_path, "0.npz"))
            assert os.path.exists(os.path.join(test_path, "0.npz"))

    def test_image_dataset(self):
        import pandas as pd

        from pfllib.data.utils import ImageDataset

        df = pd.DataFrame({"file_name": ["img1.png"], "class": [0]})
        ds = ImageDataset(df, image_folder="/tmp", transform=None)
        assert len(ds) == 1


# ─── Data Reader ──────────────────────────────────────────────────────────────


class TestDataReader:
    def test_process_image(self):
        from pfllib.data.reader import process_image

        data = {"x": np.random.randn(10, 28, 28), "y": np.random.randint(0, 10, 10)}
        result = process_image(data)
        assert len(result) == 10
        assert isinstance(result[0][0], torch.Tensor)
        assert result[0][1].dim() == 0

    def test_process_shakespeare(self):
        from pfllib.data.reader import process_shakespeare

        data = {"x": np.random.randint(0, 80, (10, 100)), "y": np.random.randint(0, 80, (10, 100))}
        result = process_shakespeare(data)
        assert len(result) == 10
        assert isinstance(result[0][0], torch.Tensor)
        assert result[0][0].dtype == torch.int64

    def test_process_text(self):
        from pfllib.data.reader import process_text

        x_data = np.random.randint(0, 100, (10, 50))
        x_lens = np.random.randint(10, 50, 10)
        data = {"x": list(zip(x_data, x_lens)), "y": np.random.randint(0, 5, 10)}
        result = process_text(data)
        assert len(result) == 10
        assert isinstance(result[0][0], tuple)

    def test_read_data_file_not_found(self):
        from pfllib.data.reader import read_data

        with pytest.raises(FileNotFoundError):
            read_data("NonExistentDataset", 0, True, data_dir="/tmp/no_such_dir")


# ─── Optimizers ───────────────────────────────────────────────────────────────


class TestOptimizers:
    def test_per_avg_optimizer(self):
        from pfllib.optimizers.fed_optimizers import PerAvgOptimizer

        model = nn.Linear(10, 2)
        opt = PerAvgOptimizer(model.parameters(), lr=0.01)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        opt.step()

    def test_scaffold_optimizer(self):
        from pfllib.optimizers.fed_optimizers import SCAFFOLDOptimizer

        model = nn.Linear(10, 2)
        opt = SCAFFOLDOptimizer(model.parameters(), lr=0.01)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        server_cs = [torch.zeros_like(p) for p in model.parameters()]
        client_cs = [torch.zeros_like(p) for p in model.parameters()]
        opt.step(server_cs, client_cs)

    def test_pfedme_optimizer(self):
        from pfllib.optimizers.fed_optimizers import pFedMeOptimizer

        model = nn.Linear(10, 2)
        opt = pFedMeOptimizer(model.parameters(), lr=0.01, lamda=0.1, mu=0.001)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        local_model = [p.clone().detach() for p in model.parameters()]
        opt.step(local_model, device="cpu")

    def test_perturbed_gradient_descent(self):
        from pfllib.optimizers.fed_optimizers import PerturbedGradientDescent

        model = nn.Linear(10, 2)
        opt = PerturbedGradientDescent(model.parameters(), lr=0.01, mu=0.01)
        x = torch.randn(4, 10)
        loss = model(x).sum()
        loss.backward()
        global_params = [p.clone().detach() for p in model.parameters()]
        opt.step(global_params, device="cpu")


# ─── Package ──────────────────────────────────────────────────────────────────


class TestPackage:
    def test_version(self):
        import pfllib

        assert pfllib.__version__ == "0.1.0"

    def test_public_api_exports(self):
        import pfllib

        assert hasattr(pfllib, "ExperimentConfig")
        assert hasattr(pfllib, "register_algorithm")
        assert hasattr(pfllib, "get_algorithm")
        assert hasattr(pfllib, "list_algorithms")
        assert hasattr(pfllib, "register_model")
        assert hasattr(pfllib, "get_model_builder")
        assert hasattr(pfllib, "list_models")

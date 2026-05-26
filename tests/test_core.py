import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn


def test_config_creation():
    from pfllib.config import ExperimentConfig

    config = ExperimentConfig(
        dataset="MNIST",
        algorithm="FedAvg",
        model="CNN",
    )
    assert config.dataset == "MNIST"
    assert config.algorithm == "FedAvg"
    assert config.model == "CNN"
    assert config.num_clients == 20
    assert config.data_dir == Path("data")


def test_config_defaults():
    from pfllib.config import ExperimentConfig

    config = ExperimentConfig()
    assert config.device == "cuda"
    assert config.global_rounds == 2000
    assert config.local_epochs == 1
    assert config.batch_size == 10


def test_config_extra_fields():
    from pfllib.config import ExperimentConfig

    config = ExperimentConfig(extra_field="hello")
    assert config.extra_field == "hello"


def test_registry_register_and_get():
    from pfllib.registry import _ALGORITHM_REGISTRY, get_algorithm, register_algorithm

    class MockServer:
        pass

    class MockClient:
        pass

    register_algorithm("TestAlgo", client_cls=MockClient, uses_head_split=False)(MockServer)
    server_cls, client_cls, uses_head = get_algorithm("TestAlgo")
    assert server_cls is MockServer
    assert client_cls is MockClient
    assert uses_head is False
    _ALGORITHM_REGISTRY.pop("TestAlgo", None)


def test_registry_list():
    from pfllib.registry import _ALGORITHM_REGISTRY, list_algorithms, register_algorithm

    class S:
        pass

    class C:
        pass

    register_algorithm("ListTestAlgo", client_cls=C, uses_head_split=True)(S)
    algos = list_algorithms()
    assert "ListTestAlgo" in algos
    _ALGORITHM_REGISTRY.pop("ListTestAlgo", None)


def test_registry_unknown_algorithm():
    from pfllib.registry import get_algorithm

    with pytest.raises(KeyError):
        get_algorithm("NonExistentAlgo")


def test_logger_setup():
    from pfllib.logger import get_logger, setup_logger

    setup_logger(level="DEBUG")
    logger = get_logger("test_module")
    assert logger.name == "test_module"


def test_base_head_split():
    from pfllib.models.base import BaseHeadSplit

    base = nn.Linear(10, 5)
    head = nn.Linear(5, 2)
    model = BaseHeadSplit(base, head)
    x = torch.randn(3, 10)
    out = model(x)
    assert out.shape == (3, 2)


def test_cnn_models():
    from pfllib.models.cnn import DNN, FedAvgCNN, Mclr_Logistic

    model = FedAvgCNN(in_features=1, num_classes=10, dim=1024)
    x = torch.randn(2, 1, 28, 28)
    out = model(x)
    assert out.shape[0] == 2

    model2 = DNN(28 * 28, 100, 10)
    x2 = torch.randn(2, 28 * 28)
    out2 = model2(x2)
    assert out2.shape[0] == 2

    model3 = Mclr_Logistic(28 * 28, 10)
    out3 = model3(x2)
    assert out3.shape[0] == 2


def test_lstm_models():
    from pfllib.models.lstm import fastText

    model = fastText(hidden_dim=32, vocab_size=100, num_classes=5)
    x = torch.randint(0, 100, (2, 10))
    out = model(x)
    assert out.shape == (2, 5)


def test_resnet_models():
    from pfllib.models.resnet import resnet10

    model = resnet10(num_classes=10)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 10)


def test_client_base_init():
    from pfllib.clients.base import Client
    from pfllib.config import ExperimentConfig

    simple_model = nn.Linear(10, 2)
    config = ExperimentConfig(
        dataset="MNIST",
        algorithm="FedAvg",
        model="CNN",
        num_classes=2,
    )
    config.model = simple_model

    client = Client(
        args=config,
        id=0,
        train_samples=100,
        test_samples=50,
    )
    assert client.id == 0
    assert client.train_samples == 100
    assert client.test_samples == 50
    assert client.learning_rate == config.local_learning_rate


def test_server_base_init():
    from pfllib.config import ExperimentConfig
    from pfllib.servers.base import Server

    simple_model = nn.Linear(10, 2)
    config = ExperimentConfig(
        dataset="MNIST",
        algorithm="FedAvg",
        model="CNN",
        num_classes=2,
    )
    config.model = simple_model

    server = Server(config, times=0)
    assert server.num_clients == 20
    assert server.global_rounds == 2000


def test_data_utils_split_data():
    import numpy as np

    from pfllib.data.utils import split_data

    X = [np.random.randn(100, 5) for _ in range(3)]
    y = [np.random.randint(0, 10, 100) for _ in range(3)]
    train_data, test_data = split_data(X, y, train_ratio=0.8)
    assert len(train_data) == 3
    assert len(test_data) == 3
    assert train_data[0]["x"].shape[0] + test_data[0]["x"].shape[0] == 100


def test_data_utils_check_nonexistent():
    import tempfile

    from pfllib.data.utils import check

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = os.path.join(tmpdir, "config.json")
        train_path = os.path.join(tmpdir, "train/")
        test_path = os.path.join(tmpdir, "test/")
        result = check(config_path, train_path, test_path, num_clients=20)
        assert result is False


def test_cli_app_exists():
    from pfllib.cli import app

    assert app is not None


def test_package_version():
    import pfllib

    assert hasattr(pfllib, "__version__")
    assert pfllib.__version__ == "0.1.0"

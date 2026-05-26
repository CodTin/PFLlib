import copy
import os
import time

import torch
import torch.nn as nn

from pfllib.config import ExperimentConfig
from pfllib.logger import get_logger, setup_logger
from pfllib.models.base import BaseHeadSplit
from pfllib.registry import get_algorithm
from pfllib.utils.mem import MemReporter
from pfllib.utils.results import average_data

logger = get_logger(__name__)


def build_model(config: ExperimentConfig) -> nn.Module:
    import torchvision

    from pfllib.models.alexnet import alexnet
    from pfllib.models.bilstm import BiLSTM_TextClassification
    from pfllib.models.cnn import (
        DNN,
        HARCNN,
        AmazonMLP,
        Digit5CNN,
        FedAvgCNN,
        Mclr_Logistic,
    )
    from pfllib.models.lstm import LSTMNet, TextCNN, fastText
    from pfllib.models.mobilenet import mobilenet_v2
    from pfllib.models.resnet import resnet10
    from pfllib.models.transformer import TransformerModel

    model_str = config.model
    nc = config.num_classes
    dev = config.device

    if model_str == "MLR":
        if "MNIST" in config.dataset:
            return Mclr_Logistic(1 * 28 * 28, num_classes=nc).to(dev)
        elif "Cifar10" in config.dataset:
            return Mclr_Logistic(3 * 32 * 32, num_classes=nc).to(dev)
        else:
            return Mclr_Logistic(60, num_classes=nc).to(dev)

    elif model_str == "CNN":
        if "MNIST" in config.dataset:
            return FedAvgCNN(in_features=1, num_classes=nc, dim=1024).to(dev)
        elif "Cifar10" in config.dataset:
            return FedAvgCNN(in_features=3, num_classes=nc, dim=1600).to(dev)
        elif "Omniglot" in config.dataset:
            return FedAvgCNN(in_features=1, num_classes=nc, dim=33856).to(dev)
        elif "Digit5" in config.dataset:
            return Digit5CNN().to(dev)
        else:
            return FedAvgCNN(in_features=3, num_classes=nc, dim=10816).to(dev)

    elif model_str == "DNN":
        if "MNIST" in config.dataset:
            return DNN(1 * 28 * 28, 100, num_classes=nc).to(dev)
        elif "Cifar10" in config.dataset:
            return DNN(3 * 32 * 32, 100, num_classes=nc).to(dev)
        else:
            return DNN(60, 20, num_classes=nc).to(dev)

    elif model_str == "ResNet18":
        return torchvision.models.resnet18(pretrained=False, num_classes=nc).to(dev)

    elif model_str == "ResNet10":
        return resnet10(num_classes=nc).to(dev)

    elif model_str == "ResNet34":
        return torchvision.models.resnet34(pretrained=False, num_classes=nc).to(dev)

    elif model_str == "AlexNet":
        return alexnet(pretrained=False, num_classes=nc).to(dev)

    elif model_str == "GoogleNet":
        return torchvision.models.googlenet(pretrained=False, aux_logits=False, num_classes=nc).to(dev)

    elif model_str == "MobileNet":
        return mobilenet_v2(pretrained=False, num_classes=nc).to(dev)

    elif model_str == "LSTM":
        return LSTMNet(
            hidden_dim=config.feature_dim,
            vocab_size=config.vocab_size,
            num_classes=nc,
        ).to(dev)

    elif model_str == "BiLSTM":
        return BiLSTM_TextClassification(
            input_size=config.vocab_size,
            hidden_size=config.feature_dim,
            output_size=nc,
            num_layers=1,
            embedding_dropout=0,
            lstm_dropout=0,
            attention_dropout=0,
            embedding_length=config.feature_dim,
        ).to(dev)

    elif model_str == "fastText":
        return fastText(
            hidden_dim=config.feature_dim,
            vocab_size=config.vocab_size,
            num_classes=nc,
        ).to(dev)

    elif model_str == "TextCNN":
        return TextCNN(
            hidden_dim=config.feature_dim,
            max_len=config.max_len,
            vocab_size=config.vocab_size,
            num_classes=nc,
        ).to(dev)

    elif model_str == "Transformer":
        return TransformerModel(
            ntoken=config.vocab_size,
            d_model=config.feature_dim,
            nhead=8,
            nlayers=2,
            num_classes=nc,
            max_len=config.max_len,
        ).to(dev)

    elif model_str == "AmazonMLP":
        return AmazonMLP().to(dev)

    elif model_str == "HARCNN":
        if config.dataset == "HAR":
            return HARCNN(9, dim_hidden=1664, num_classes=nc, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(dev)
        elif config.dataset == "PAMAP2":
            return HARCNN(9, dim_hidden=3712, num_classes=nc, conv_kernel_size=(1, 9), pool_kernel_size=(1, 2)).to(dev)

    raise NotImplementedError(f"Unknown model: {model_str}")


def run_experiment(config: ExperimentConfig) -> None:
    setup_logger(level="INFO")

    os.environ["CUDA_VISIBLE_DEVICES"] = config.device_id
    if config.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA is not available, falling back to CPU")
        config.device = "cpu"

    logger.info("=" * 50)
    for field_name, value in config:
        logger.info(f"{field_name} = {value}")
    logger.info("=" * 50)

    reporter = MemReporter()
    time_list = []

    for i in range(config.prev, config.times):
        logger.info(f"\n============= Running time: {i}th =============")
        logger.info("Creating server and clients ...")
        start = time.time()

        model = build_model(config)
        logger.info(f"Model: {model}")

        server_cls, client_cls, uses_head_split = get_algorithm(config.algorithm)

        if uses_head_split:
            head = copy.deepcopy(model.fc)
            assert isinstance(head, nn.Module), "model.fc must be an nn.Module for head splitting"
            model.fc = nn.Identity()  # type: ignore[assignment]
            model = BaseHeadSplit(model, head)

        config.model = model  # type: ignore[assignment]
        server = server_cls(config, i)
        server.train()
        time_list.append(time.time() - start)

    avg_time = round(sum(time_list) / len(time_list), 2) if time_list else 0
    logger.info(f"Average time cost: {avg_time}s")

    average_data(
        dataset=config.dataset,
        algorithm=config.algorithm,
        goal=config.goal,
        times=config.times,
        results_dir=config.results_dir,
    )

    logger.info("All done!")
    reporter.report()

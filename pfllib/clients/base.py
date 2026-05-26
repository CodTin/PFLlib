import copy
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader

from pfllib.config import ExperimentConfig
from pfllib.data.reader import read_client_data
from pfllib.logger import get_logger

logger = get_logger(__name__)


class Client:
    def __init__(
        self,
        args: ExperimentConfig,
        id: int,
        train_samples: int,
        test_samples: int,
        train_slow: bool = False,
        send_slow: bool = False,
        **kwargs,
    ):
        torch.manual_seed(0)
        assert isinstance(args.model, nn.Module), "model must be an nn.Module instance"
        self.model: nn.Module = copy.deepcopy(args.model)
        self.args = args
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot

        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = train_slow
        self.send_slow = send_slow
        self.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}
        self.send_time_cost = {"num_rounds": 0, "total_cost": 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay

    def load_train_data(self, batch_size: Optional[int] = None) -> DataLoader:
        if batch_size is None:
            batch_size = self.batch_size
        train_data = read_client_data(
            self.dataset, self.id, is_train=True, few_shot=self.few_shot, data_dir=self.args.data_dir
        )
        return DataLoader(train_data, batch_size, drop_last=True, shuffle=True)  # type: ignore[arg-type]

    def load_test_data(self, batch_size: Optional[int] = None) -> DataLoader:
        if batch_size is None:
            batch_size = self.batch_size
        test_data = read_client_data(
            self.dataset, self.id, is_train=False, few_shot=self.few_shot, data_dir=self.args.data_dir
        )
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)  # type: ignore[arg-type]

    def set_parameters(self, model: nn.Module):
        for new_param, old_param in zip(model.parameters(), self.model.parameters()):
            old_param.data = new_param.data.clone()

    def clone_model(self, model: nn.Module, target: nn.Module):
        for param, target_param in zip(model.parameters(), target.parameters()):
            target_param.data = param.data.clone()

    def update_parameters(self, model: nn.Module, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self) -> tuple[int | float, int, float]:
        testloaderfull = self.load_test_data()
        self.model.eval()

        test_acc: int | float = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average="micro")

        return test_acc, test_num, auc

    def train_metrics(self) -> tuple[float, int]:
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def save_item(self, item, item_name: str, item_path: Optional[str] = None):
        if item_path is None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, f"client_{self.id}_{item_name}.pt"))

    def load_item(self, item_name: str, item_path: Optional[str] = None):
        if item_path is None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, f"client_{self.id}_{item_name}.pt"))

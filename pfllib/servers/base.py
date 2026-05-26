from __future__ import annotations

import copy
import random
import time
from typing import TYPE_CHECKING

import h5py
import numpy as np
import torch
import torch.nn as nn

from pfllib.config import ExperimentConfig
from pfllib.data.reader import read_client_data
from pfllib.logger import get_logger
from pfllib.privacy.dlg import DLG

if TYPE_CHECKING:
    from pfllib.clients.base import Client

logger = get_logger(__name__)


class Server:
    def __init__(self, args: ExperimentConfig, times: int):
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        assert isinstance(args.model, nn.Module), "model must be an nn.Module instance"
        self.global_model: nn.Module = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threshold = args.time_threshold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients: list[Client] = []
        self.selected_clients: list[Client] = []
        self.train_slow_clients: list[bool] = []
        self.send_slow_clients: list[bool] = []

        self.uploaded_weights: list[float] = []
        self.uploaded_ids: list[int] = []
        self.uploaded_models: list[nn.Module] = []

        self.rs_test_acc: list[float] = []
        self.rs_test_auc: list[float] = []
        self.rs_train_loss: list[float] = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients: list[Client] = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, client_cls):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(
                self.dataset, i, is_train=True, few_shot=self.few_shot, data_dir=self.args.data_dir
            )
            test_data = read_client_data(
                self.dataset, i, is_train=False, few_shot=self.few_shot, data_dir=self.args.data_dir
            )
            client = client_cls(
                self.args,
                id=i,
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_slow=train_slow,
                send_slow=send_slow,
            )
            self.clients.append(client)

    def select_slow_clients(self, slow_rate: float) -> list[bool]:
        slow_clients = [False for _ in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True
        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(
                range(self.num_join_clients, self.num_clients + 1), 1, replace=False
            )[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        return selected_clients

    def send_models(self):
        assert len(self.clients) > 0

        for client in self.clients:
            start_time = time.time()

            client.set_parameters(self.global_model)

            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients),
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = (
                    client.train_time_cost["total_cost"] / client.train_time_cost["num_rounds"]
                    + client.send_time_cost["total_cost"] / client.send_time_cost["num_rounds"]
                )
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threshold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert len(self.uploaded_models) > 0

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w: float, client_model: nn.Module):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def save_global_model(self):
        model_path = self.args.outputs_dir / self.dataset
        model_path.mkdir(parents=True, exist_ok=True)
        model_path = model_path / (self.algorithm + "_server.pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = self.args.outputs_dir / self.dataset / (self.algorithm + "_server.pt")
        assert model_path.exists()
        self.global_model = torch.load(model_path)

    def model_exists(self) -> bool:
        model_path = self.args.outputs_dir / self.dataset / (self.algorithm + "_server.pt")
        return model_path.exists()

    def save_results(self):
        algo = self.dataset + "_" + self.algorithm
        result_path = self.args.results_dir
        result_path.mkdir(parents=True, exist_ok=True)

        if len(self.rs_test_acc):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path / f"{algo}.h5"
            logger.info(f"File path: {file_path}")

            with h5py.File(file_path, "w") as hf:
                hf.create_dataset("rs_test_acc", data=self.rs_test_acc)
                hf.create_dataset("rs_test_auc", data=self.rs_test_auc)
                hf.create_dataset("rs_train_loss", data=self.rs_train_loss)

    def save_item(self, item, item_name: str):
        path = self.args.outputs_dir / self.save_folder_name
        path.mkdir(parents=True, exist_ok=True)
        torch.save(item, path / f"server_{item_name}.pt")

    def load_item(self, item_name: str):
        return torch.load(self.args.outputs_dir / self.save_folder_name / f"server_{item_name}.pt")

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()

        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]
        return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]

        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)

        ids = [c.id for c in self.clients]
        return ids, num_samples, losses

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2]) * 1.0 / sum(stats[1])
        test_auc = sum(stats[3]) * 1.0 / sum(stats[1])
        train_loss = sum(stats_train[2]) * 1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]

        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        logger.info(f"Averaged Train Loss: {train_loss:.4f}")
        logger.info(f"Averaged Test Accuracy: {test_acc:.4f}")
        logger.info(f"Averaged Test AUC: {test_auc:.4f}")
        logger.info(f"Std Test Accuracy: {np.std(accs):.4f}")
        logger.info(f"Std Test AUC: {np.std(aucs):.4f}")

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R: int):
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if isinstance(x, list):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1

        if cnt > 0:
            logger.info(f"PSNR value is {psnr_val / cnt:.2f} dB")
        else:
            logger.warning("PSNR error")

    def set_new_clients(self, client_cls):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(
                self.dataset, i, is_train=True, few_shot=self.few_shot, data_dir=self.args.data_dir
            )
            test_data = read_client_data(
                self.dataset, i, is_train=False, few_shot=self.few_shot, data_dir=self.args.data_dir
            )
            client = client_cls(
                self.args,
                id=i,
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_slow=False,
                send_slow=False,
            )
            self.new_clients.append(client)

    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if isinstance(x, list):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            ct, ns, auc = c.test_metrics()
            tot_correct.append(ct * 1.0)
            tot_auc.append(auc * ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]
        return ids, num_samples, tot_correct, tot_auc

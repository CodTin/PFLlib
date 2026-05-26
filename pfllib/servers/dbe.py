import time

from pfllib.clients.dbe import ClientDBE
from pfllib.logger import get_logger
from pfllib.servers.base import Server

logger = get_logger(__name__)


class FedDBE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()

        # initialization period
        self.set_clients(ClientDBE)
        self.selected_clients = self.clients
        for client in self.selected_clients:
            client.train()  # no DBE

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

        global_mean = 0
        for cid, w in zip(self.uploaded_ids, self.uploaded_weights):
            global_mean += self.clients[cid].running_mean * w
        logger.info(">>>> global_mean <<<<", global_mean)
        for client in self.selected_clients:
            client.global_mean = global_mean.data.clone()

        logger.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logger.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        logger.info("featrue map shape: ", self.clients[0].client_mean.shape)
        logger.info("featrue map numel: ", self.clients[0].client_mean.numel())

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if i % self.eval_gap == 0:
                logger.info(f"\n-------------Round number: {i}-------------")
                logger.info("\nEvaluate model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            logger.info("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        logger.info("\nBest accuracy.")
        logger.info(max(self.rs_test_acc))
        logger.info("\nAverage time cost per round.")
        logger.info(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

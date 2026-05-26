import time

from pfllib.clients.avg import ClientAVG
from pfllib.logger import get_logger
from pfllib.servers.base import Server

logger = get_logger(__name__)


class Local(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(ClientAVG)

        logger.info(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        logger.info("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                logger.info(f"\n-------------Round number: {i}-------------")
                logger.info("\nEvaluate personalized models")
                self.evaluate()

            self.selected_clients = self.select_clients()
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.Budget.append(time.time() - s_t)
            logger.info("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        logger.info("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        logger.info(max(self.rs_test_acc))
        logger.info("\nAverage time cost per round.")
        logger.info(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

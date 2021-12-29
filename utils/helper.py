from shutil import copyfile
from datetime import datetime
from tensorboardX import SummaryWriter

import torch
import random
import numpy as np
import os
import logging

from utils.utils import create_logger, output_process


class Helper:
    def __init__(self, params, name, seed):
        # BASE
        self.current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
        self.writer = None
        self.logger = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = None
        self.name = name
        self.seed = seed
        self.best_result = (0, 0, 0)

        # DATA
        self.dataset = {'train': None, 'test': None}
        self.test_type = 'server'
        self.data_name = params.get('data_name', 'cifar')
        self.data_path = params.get('data_path', None)

        # MODEL
        self.global_model = None
        self.model_name = params.get('model_name', None)
        self.model_depth = params.get('model_depth', None)

        # TRAINING STRATEGY
        self.optimizer = params.get('optimizer', None)
        self.lr_scheduler = params.get('lr_scheduler', None)
        self.lr = params.get('lr', None)
        self.lr_decay = params.get('lr_decay', None)
        self.lr_decay_interval = params.get('lr_decay_interval', None)
        self.weight_decay = params.get('weight_decay', None)
        self.momentum = params.get('momentum', None)
        self.save_rounds = params.get('save_rounds', [1500])
        self.batch_size = params.get('batch_size', None)
        self.num_workers = params.get('num_workers', 0)
        self.resumed_model = params.get('resumed_model', False)
        self.resumed_reset = params.get('resumed_reset', False)
        self.only_eval = params.get('only_eval', False)

        # FEDERATED LEARNING PARAMS
        self.start_round = params.get('start_round', 1)
        self.total_round = params.get('total_round', 0)
        self.local_round = params.get('local_round', 1)
        self.aggregation_type = params.get('aggregation_type', 'fedavg')
        self.num_clients = params.get('num_clients', 100)
        self.sampled_ratio = params.get('sampled_ratio', 0.1)

        # SAVE PATH
        self.repo_path = os.getcwd()
        self.save_folder = f'{self.repo_path}/results/{self.data_name}_{self.model_name}_{self.model_depth}_' \
                           f'{self.seed}/{self.name}'

        output_process(self.save_folder)  # create folder or not
        self._init_log()  # get log and writer
        self.fix_random(self.seed)

    def _init_log(self):
        self.writer = SummaryWriter(log_dir=self.save_folder)
        self.logger = create_logger()
        fh = logging.FileHandler(filename=f'{self.save_folder}/log.txt')
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def save_opts(self):
        opts = vars(self)
        with open(f"{self.save_folder}/opts.txt", 'w') as f:
            for k, v in opts.items():
                f.write(str(k) + ": " + str(v) + '\n')

    def save_model(self, cur_result=None):
        """

        :param cur_result: (top1, round) (top1, top5, round)
        :return:
        """
        assert isinstance(cur_result, tuple), 'Best result should be tuple.'
        model = self.global_model
        self.logger.info("saving model")
        if cur_result[-1] in self.save_rounds:
            model_name = f"{self.save_folder}/checkpoint_{cur_result[-1]}.pth.tar"
        else:
            model_name = f"{self.save_folder}/checkpoint.pth.tar"
        saved_dict = {'state_dict': model.state_dict(), 'round': cur_result[-1], 'lr': self.lr}
        if cur_result[0] > self.best_result[0]:
            self.best_result = cur_result
            self.save_checkpoint(saved_dict, True, model_name)
        else:
            self.save_checkpoint(saved_dict, False, model_name)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            model_name = f"{self.save_folder}/model_best.pth.tar"
            copyfile(filename, model_name)

    @staticmethod
    def fix_random(seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return True

from datetime import datetime
from tensorboardX import SummaryWriter

import os
import logging

from utils.utils import create_logger, output_process, fix_random

from torchutils import output_directory, logger, summary_writer

class BaseTrainer(object):
    def __init__(self, experimental_name='debug', seed=None):
        # BASE
        self.current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
        self.writer = None
        self.logger = None
        self.experimental_name = experimental_name
        self.seed = seed

        # SAVE PATH
        # self.repo_path = os.getcwd()
        # self.save_folder = f'{self.repo_path}/results/{experimental_name}'
        self.save_folder = output_directory
        self.logger = logger
        self.writer = summary_writer

        # output_process(self.save_folder)  # create folder or not
        # self._init_log()  # get log and writer
        # if seed is not None:
            # fix_random(seed)

    # def _init_log(self):
    #     self.writer = SummaryWriter(log_dir=self.save_folder)
    #     self.logger = create_logger()
    #     fh = logging.FileHandler(filename=f'{self.save_folder}/log.txt')
    #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     fh.setFormatter(formatter)
    #     self.logger.addHandler(fh)

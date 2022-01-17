from datetime import datetime
from torchutils import output_directory, logger, summary_writer


class BaseTrainer(object):
    def __init__(self, experimental_name='debug', seed=None):
        # BASE
        self.current_time = datetime.now().strftime('%b.%d_%H.%M.%S')
        self.writer = None
        self.logger = None
        self.experimental_name = experimental_name
        self.seed = seed

        self.save_folder = output_directory
        self.logger = logger
        self.writer = summary_writer


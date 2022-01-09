import numpy as np
import prettytable as pt
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from matplotlib.pyplot import MultipleLocator
import matplotlib as mpl
import matplotlib.ticker as ticker
mpl.rcParams['font.sans-serif'] = ['SimSun']
mpl.rcParams['axes.unicode_minus'] = False
import torch
from collections import defaultdict


class FederatedDataset(object):
    """
    Create federated dataset.
    We consider image classification dataset by default. (e.g., MNIST、CIFAR-10、IMAGENET.)
     Args:
         dataset (Dataset): The whole Dataset.
         num_clients (int): Number of total clients.
         partition_type (str): Type of data partition. (IID, N-class Non-IID, Dirichlet)
         batch_size (int): Batch size.
         num_workers (int): Number of threads to load data.
         server_ratio (float): The case that server owns some data.
         indices (dict(list)): indices of dataset for client or server.
     """
    def __init__(self, dataset, num_clients, partition_type, batch_size=50, num_workers=0, server_ratio=0, data_name='cifar'):
        self.dataset = dataset
        self.num_clients = num_clients
        self.partition_type = partition_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.server_ratio = server_ratio
        self.data_name = data_name

        self.indices = defaultdict(list)
        if self.dataset is not None:
            self.indices['all_clients'] = list(range(len(self.dataset)))

        self.run()

    def run(self):
        # self.get_server_dataset_indices()
        # if self.server_ratio < 1:
        self.get_clients_dataset_indices()

    def __str__(self):
        tb = pt.PrettyTable()
        tb.field_names = ['name', 'value']
        tb.add_row(['Number of Clients', self.num_clients])
        tb.add_row(['Partition Type', self.partition_type])
        return tb.__str__()

    def get_clients_dataset_indices(self):
        """
          To get indices of dataset in each client, which satisfy some federated distribution.
          That is, fill in 'self.indices[client_id]'.
        """
        raise NotImplementedError

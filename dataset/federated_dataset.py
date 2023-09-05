import numpy as np

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
        self.get_clients_dataset_indices()

    def get_clients_dataset_indices(self):
        """
          To get indices of dataset in each client, which satisfy some federated distribution.
          That is, fill in 'self.indices[client_id]'.
        """
        raise NotImplementedError

    def get_reverse_label_distribution(self):
        num_classes = len(self.dataset.classes)
        distribution = np.zeros((self.num_clients, num_classes))
        targets = self.dataset.targets
        for client_id in range(self.num_clients):
            for idx in self.indices[client_id]:
                target = targets[idx]
                distribution[client_id][target] += 1
        weights_distribution = np.zeros_like(distribution)
        for i in range(len(weights_distribution)):
            for j in range(len(weights_distribution[i])):
                weights_distribution[i][j] = distribution[i][j] / np.sum(distribution[i])
        
        # get reverse distribution
        weights_distribution = 1 - weights_distribution
        # normalization
        weights_distribution = weights_distribution / (num_classes - 1)
        
        return weights_distribution

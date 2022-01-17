import numpy as np
import random

from dataset.federated_dataset import FederatedDataset

PARTITION_TYPE = 'DIRICHLET'


class FederatedDatasetDirichlet(FederatedDataset):
    """
    Create federated dataset for dirichlet sampling setting.
    """

    def __init__(self, dataset, num_clients, batch_size=50, num_workers=0, server_ratio=0, dirichlet_alpha=0.5,
                 balance=True, seed=7777, dirichlet_threshold=0, data_name='cifar', **kwargs):
        self.alpha = dirichlet_alpha
        self.balance = balance
        self.seed = seed
        self.threshold = dirichlet_threshold
        super().__init__(dataset, num_clients, PARTITION_TYPE, batch_size, num_workers, server_ratio, data_name)

    def get_clients_dataset_indices(self):
        np.random.seed(self.seed)  # To guarantee training dataset and testing dataset stay same distribution
        classes = {}
        targets = self.dataset.targets
        if not isinstance(targets, list):
            targets = targets.numpy().tolist()
        for ind in self.indices['all_clients']:
            label = targets[ind]
            if label in classes:
                classes[label].append(ind)
            else:
                classes[label] = [ind]
        no_classes = len(classes.keys())
        datasize = np.zeros(self.num_clients)
        if self.balance:
            num_per_client = int(len(self.indices['all_clients']) / self.num_clients)
            while min(datasize) < num_per_client - self.threshold:
                for user in range(self.num_clients):
                    num_this_client = num_per_client - datasize[user]
                    sampled_probabilities = num_this_client * np.random.dirichlet(
                        np.array(no_classes * [self.alpha]))
                    for n in range(no_classes):
                        no_imgs = int(round(sampled_probabilities[n]))
                        no_imgs = min(len(classes[n]), no_imgs)
                        no_imgs = min(int(num_this_client), no_imgs)
                        datasize[user] += no_imgs
                        sampled_list = classes[n][:no_imgs]
                        self.indices[user].extend(sampled_list)
                        classes[n] = classes[n][no_imgs:]
                        num_this_client = num_per_client - datasize[user]
                # find the last two min
                sort_array = sorted(np.unique(datasize))
                if sort_array[1] >= num_per_client - self.threshold:
                    for n in range(no_classes):
                        sampled_list = classes[n]
                        self.indices[user].extend(sampled_list)
                    break

        else:
            for n in range(no_classes):
                random.shuffle(classes[n])
                sampled_probabilities = len(classes[n]) * np.random.dirichlet(
                    np.array(self.num_clients * [self.alpha]))
                # to avoid some client do not have data
                if self.alpha < 0.1 or self.num_clients > 200:
                    sampled_probabilities = np.clip(sampled_probabilities, self.threshold,
                                                    len(classes[n]) - self.threshold)
                for user in range(self.num_clients):
                    no_imgs = int(round(sampled_probabilities[user]))
                    no_imgs = min(len(classes[n]), no_imgs)
                    datasize[user] += no_imgs
                    sampled_list = classes[n][:no_imgs]
                    self.indices[user].extend(sampled_list)
                    classes[n] = classes[n][no_imgs:]


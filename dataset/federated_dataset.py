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

    # def get_server_dataset_indices(self):
    #     """
    #     Consider simple case: randomly send part of original training dataset into server.
    #     """
    #     num_server_data = int(len(self.indices['all_clients']) * self.server_ratio)
    #     server_indices = np.random.choice(self.indices['all_clients'], num_server_data, replace=False)
    #     self.indices['server'] = list(server_indices)
    #     self.indices['all_clients'] = list(set(self.indices['all_clients']) - set(server_indices))
    #     return server_indices

    def get_clients_dataset_indices(self):
        """
          To get indices of dataset in each client, which satisfy some federated distribution.
          That is, fill in 'self.indices[client_id]'.
        """
        raise NotImplementedError

    def plot_distribution_histogram(self):
        """
        Plot data distribution for classification dataset.
        """
        # num_classes = len(self.dataset.classes)
        num_classes = self.num_classes
        distribution = np.zeros((num_classes, self.num_clients))
        targets = self.dataset.targets
        for client_id in range(self.num_clients):
            for idx in self.indices[client_id]:
                target = targets[idx]
                distribution[target][client_id] += 1
        weights_distribution = np.zeros_like(distribution)
        for i in range(len(weights_distribution)):
            for j in range(len(weights_distribution[i])):
                weights_distribution[i][j] = distribution[i][j] / np.sum(distribution[i])

        print(weights_distribution)
        plt.figure()
        label_list = range(self.num_clients)
        for i in range(num_classes):
            plt.bar(label_list, distribution[i], bottom=np.sum(distribution[:i], axis=0), label=f'class-{i}')
        plt.legend(loc=2, bbox_to_anchor=(1.0, 1.0))
        # savefig() should be called before show().
        # if save_path:
        plt.savefig('dirichlet0.05.png')
        plt.show()

    def plot_label_distribution_histogram(self, save_dir):
        """
        Plot data distribution for classification dataset.
        """
        num_classes = len(self.dataset.classes)
        distribution = np.zeros((num_classes, self.num_clients))
        global_distribution = [0] * num_classes
        targets = self.dataset.targets
        print("length: ", len(targets))
        targets_length = len(self.dataset.targets)
        global_sum = 0
        for client_id in range(self.num_clients):
            client_sum = len(self.indices[client_id]);
            global_sum += client_sum
            client_distribution = [0] * num_classes
            for idx in self.indices[client_id]:
                target = targets[idx]
                client_distribution[target] += 1
                global_distribution[target] += 1
            # if client_id == 67 or client_id == 88:
            max_distribution = 0.0
            for c in range(num_classes):
                client_distribution[c] = client_distribution[c] / (client_sum * 1.0)
                # if client_id == 67 and (c in [1, 3, 8]):
                #     client_distribution[c] = 0
                if client_distribution[c] > max_distribution:
                    max_distribution = client_distribution[c]
            # plt.figure()
            plt.figure(figsize=(4,3))
            plt.grid(linewidth=0.1)
            y_major_locator = MultipleLocator(0.15)
            ax = plt.gca()
            ax.yaxis.set_major_locator(y_major_locator)
            plt.ylim(0, 0.312)
            my_x_ticks = np.arange(0, 10, 1)
            my_y_ticks = np.arange(0, 0.312, 0.15)
            plt.xticks(my_x_ticks, fontproperties = 'Times New Roman')
            plt.yticks(my_y_ticks, fontproperties='Times New Roman')
            plt.tick_params(top=False, bottom=False, left=False, right=False, labelsize=10, pad=0)
            class_list = range(num_classes)
            plt.bar(class_list, client_distribution, width=0.8, facecolor="#1F75B3", edgecolor='black', linewidth=0.7)
            font_label = {
                'weight': 'bold',
                'size': 12,
            }
            plt.xlabel(u'类别', font_label)
            plt.ylabel(u'概率', font_label)
            save_path = save_dir +  "/train_direchlet_" + str(client_id) + ".png"
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            # plt.show()

        for c in range(num_classes):
            global_distribution[c] /= (global_sum * 1.0)
        plt.figure(figsize=(4,3))
        plt.grid(linewidth=0.1)
        y_major_locator = MultipleLocator(0.05)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim((0, 0.104))
        my_x_ticks = np.arange(0, 10, 1)
        my_y_ticks = np.arange(0, 0.104, 0.05)
        plt.xticks(my_x_ticks, fontproperties='Times New Roman')
        plt.yticks(my_y_ticks, fontproperties='Times New Roman')
        plt.tick_params(top=False, bottom=False, left=False, right=False, labelsize=10, pad=0)
        class_list = range(num_classes)
        plt.bar(class_list, global_distribution, width=0.8, facecolor="#1F75B3", edgecolor='black', linewidth=0.7)
        font_label = {
            'weight': 'bold',
            'size': 12,
        }
        plt.xlabel(u'类别', font_label)
        plt.ylabel(u'概率', font_label)
        global_save_path = save_dir +  "/train_direchlet_global.png"
        if global_save_path:
            plt.savefig(global_save_path, dpi=300, bbox_inches='tight')
        # plt.show()

    # def get_label_distribution(self):
    #     num_classes = len(self.dataset.classes)
    #     distribution = torch.zeros((self.num_clients, num_classes))
    #     targets = self.dataset.targets
    #     for client_id in range(self.num_clients):
    #         for idx in self.indices[client_id]:
    #             target = targets[idx]
    #             distribution[client_id][target] += 1
    #     weights_distribution = torch.zeros_like(distribution)
    #     for i in range(len(weights_distribution)):
    #         for j in range(len(weights_distribution[i])):
    #             weights_distribution[i][j] = distribution[i][j] / torch.sum(distribution[i])
    #     return weights_distribution

    def get_label_distribution(self):
        if self.data_name != 'inaturalist':
            num_classes = len(self.dataset.classes)
            # num_classes = self.num_classes
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
        else:
            num_classes = 1010
            distribution = np.zeros((self.num_clients, num_classes))
            for client_id in range(self.num_clients):
                for idx in self.indices[client_id]:
                    target = idx[2]
                    distribution[client_id][target] += 1
            weights_distribution = np.zeros_like(distribution)
            for i in range(len(weights_distribution)):
                for j in range(len(weights_distribution[i])):
                    weights_distribution[i][j] = distribution[i][j] / np.sum(distribution[i])
        return weights_distribution

    def get_reverse_label_distribution(self):
        if self.data_name != 'inaturalist':
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
        else:
            num_classes = 1010
            distribution = np.zeros((self.num_clients, num_classes))
            for client_id in range(self.num_clients):
                for idx in self.indices[client_id]:
                    target = idx[2]
                    distribution[client_id][target] += 1
            weights_distribution = np.zeros_like(distribution)
            for i in range(len(weights_distribution)):
                for j in range(len(weights_distribution[i])):
                    weights_distribution[i][j] = distribution[i][j] / np.sum(distribution[i])
        # get reverse distribution
        weights_distribution = 1 - weights_distribution
        # normalization
        weights_distribution = weights_distribution / (num_classes - 1)
        # reverse_distribution = np.zeros_like(distribution)
        # for i in range(len(weights_distribution)):
        #     for j in range(len(weights_distribution[i])):
        #         reverse_distribution[i][j] = np.max(distribution[i]) - distribution[i][j]
        #     for j in range(len(weights_distribution[i])):
        #         weights_distribution[i][j] = reverse_distribution[i][j] / np.sum(reverse_distribution[i])
        return weights_distribution

    def get_sample_label_distribution(self):
        num_classes = len(self.dataset.classes)
        distribution = np.zeros((self.num_clients, num_classes))
        targets = self.dataset.targets
        for client_id in range(self.num_clients):
            for idx in self.indices[client_id]:
                target = targets[idx]
                distribution[client_id][target] += 1
        return distribution

    def get_reverse_gen_num(self):
        num_classes = len(self.dataset.classes)
        distribution = np.zeros((self.num_clients, num_classes))
        targets = self.dataset.targets
        for client_id in range(self.num_clients):
            for idx in self.indices[client_id]:
                target = targets[idx]
                distribution[client_id][target] += 1
        # weights_distribution = np.zeros_like(distribution)
        # for i in range(len(weights_distribution)):
        #     for j in range(len(weights_distribution[i])):
        #         weights_distribution[i][j] = distribution[i][j] / np.sum(distribution[i])

        # # get reverse distribution
        reverse_distribution = np.zeros_like(distribution)
        for i in range(len(distribution)):
            for j in range(len(distribution[i])):
                reverse_distribution[i][j] = np.max(distribution[i]) - distribution[i][j]
        return reverse_distribution
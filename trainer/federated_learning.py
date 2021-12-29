# import torch

# from torch.utils.data import DataLoader, Subset

# from trainer.base_trainer import BaseTrainer

# from utils.utils import AverageMeter, accuracy
# from utils.loss_function import loss_cross_entropy


# class FederatedLearning(BaseTrainer):
#     """
#     A common federated learning training includes the following processes:
#     1. Client selection 2. Broadcast 3. Client Computation 4. Aggregation 5. Model update
#     'Advances and Open Problems in Federated Learning'.
#     https://arxiv.org/abs/1912.04977
#     Args:
#         num_clients (int): Number of clients.
#         sampled_ratio (float): Ratio of sampled clients in each round. 0 < sampled_ratio <= 1
#         total_round (int): Number of total communication rounds.
#         local_round (int): How many rounds the client device trains.
#         global_model: The global model shared for all clients.
#         dataset (dict): Federated dataset.
#         test_type (str): Commonly, we test model in a shared test dataset ('server')
#                     or in local private datasets ('client').
#         lr (float): Learning rate.
#         momentum (float): Momentum.
#         weight_decay (float): Weight decay.
#         batch_size (int): Batch size.
#         num_workers (int): Number of threads to load data.
#         device (str): cpu or cuda.
#     """

#     def __init__(self, params):
#         self.params = params
#         self.multi_gpus = params.get('multi_gpus', False)
#         self.num_clients = params.get('num_clients', 1)
#         self.num_classes = params.get('num_classes', 10)

#         self.sampled_ratio = params.get('sampled_ratio', 1)
#         self.start_round = params.get('start_round', 1)
#         self.total_round = params.get('total_round', 0)
#         self.local_round = params.get('local_round', 0)
#         self.global_model = params.get('global_model')
#         self.dataset = params.get('dataset')
#         self.test_type = params.get('test_type', 'server')
#         self.lr = params.get('lr', 0.1)
#         self.lr_finetune = params.get('lr_finetune', 0.01)

#         self.momentum = params.get('momentum', 0.9)
#         self.weight_decay = params.get('weight_decay', 0.0005)
#         self.batch_size = params.get('batch_size', 64)
#         self.num_workers = params.get('num_workers', 0)
#         self.device = params.get('device', 'cuda')

#         self.model_name = params.get('model_name', '')
#         self.model_depth = params.get('model_depth', '')
#         self.data_name = params.get('data_name', '')
#         self.experimental_name = params.get('name', 'debug')
#         self.local_dataset_test = params.get('local_dataset_test', False)
#         self.test_ensemble_model = params.get('test_ensemble_model', False)
#         self.resumed_model = params.get('resumed_model', False)
#         self.resumed_reset = params.get('resumed_reset', False)
#         seed = params.get('seed', 7777)

#         experimental_name = f'{self.params["aggregation_type"]}_{self.data_name}_{self.params["partition_type"]}_' \
#                             f'{self.model_name}{self.model_depth}_{seed}/{self.experimental_name}'
#         super().__init__(experimental_name, seed)
#         self.selected_clients = []
#         self.local_models = [None] * self.num_clients
#         # self.generators = [None] * self.num_clients
#         self.client_train_loader = {}
#         self.client_val_loader = {}
#         self.client_test_loader = {}
#         self.test_loader = None

#         self.get_data_loaders()

#     def run(self):
#         """
#         Start federated training.
#         :return:
#         """
#         raise NotImplementedError

#     def client_selection(self):
#         """
#         The server sample from a set of clients meeting eligibility requirements.
#         :return:
#         """
#         raise NotImplementedError

#     def broadcast(self, hook_flag=True):
#         """
#         The selected clients download the current model (e.g. weights) from the server.
#         :return:
#         """
#         raise NotImplementedError

#     def client_computation(self, round=0):
#         """
#         Each selected device locally computes an update to the model by executing the training program,
#         e.g., run SGD on the local data.
#         :return:
#         """
#         raise NotImplementedError

#     def model_update(self, round=0):
#         """
#         The server locally updates the shared model based on the aggregated update computed from the clients
#         that participated in the current round.
#         :return:
#         """
#         raise NotImplementedError

#     def model_evaluation(self):
#         """
#         Commonly, there exists two evaluation methods.
#         1. Evaluate the global model on a shared testing dataset.
#         2. Clients evaluate their local models on their own private data,
#         the server calculate the final performance by some methods. (e.g., averaging accuracy)
#         """
#         raise NotImplementedError

#     def calculate_communication_cost(self):
#         """
#         Model propagation between the sever and local device will incur communication costs.
#         We calculate total parameter propagated for all rounds by default.
#         :return:
#         """
#         model_parameter = sum(map(lambda p: p.numel(), self.global_model.parameters()))
#         total_cost = model_parameter * self.total_round * 2
#         return total_cost

#     def get_data_loaders(self):
#         for i in range(self.num_clients):
#             dataset = self.dataset['train']
#             len_samples = len(dataset.indices[i])
#             # To avoid "ValueError('Expected more than 1 value per channel when training,
#             # got input size {}'.format(size))" when using batchnorm.
#             if len_samples % self.batch_size == 1:
#                 dataset.indices[i] = dataset.indices[i][:-1]
#             if len_samples > 0:
#                 subset = Subset(dataset.dataset, dataset.indices[i])
#                 data_loader = DataLoader(subset, batch_size=self.batch_size,
#                                          num_workers=self.num_workers,
#                                          shuffle=True)
#                 self.client_train_loader[i] = data_loader
#                 val_subset = Subset(self.dataset['val'], dataset.indices[i])
#                 self.client_val_loader[i] = DataLoader(val_subset, batch_size=self.batch_size,
#                                               num_workers=self.num_workers,
#                                               shuffle=False)

#         self.test_loader = DataLoader(self.dataset['test'].dataset, batch_size=self.batch_size,
#                                       num_workers=self.num_workers,
#                                       shuffle=False)

#     @staticmethod
#     def train(model, optimizer, data_loader, device):
#         model.train()
#         obj = AverageMeter()
#         total_top1 = AverageMeter()
#         total_top5 = AverageMeter()
#         for batch_id, (data, targets) in enumerate(data_loader):
#             data = data.to(device)
#             targets = targets.to(device)
#             optimizer.zero_grad()
#             output = model(data)
#             loss = loss_cross_entropy(output, targets)
#             top1, top5 = accuracy(output, targets, topk=(1, 5))
#             loss.backward()
#             optimizer.step()
#             obj.update(loss.item(), data.size(0))
#             total_top1.update(top1.item(), data.size(0))
#             total_top5.update(top5.item(), data.size(0))

#         return obj, total_top1, total_top5

#     @staticmethod
#     def evaluation(model, data_loader, device):
#         model.eval()
#         obj = AverageMeter()
#         total_top1 = AverageMeter()
#         total_top5 = AverageMeter()
#         with torch.no_grad():
#             for batch_id, (data, targets) in enumerate(data_loader):
#                 data = data.to(device)
#                 targets = targets.to(device)
#                 output = model(data)
#                 loss = loss_cross_entropy(output, targets)
#                 top1, top5 = accuracy(output, targets, topk=(1, 5))
#                 obj.update(loss.item(), data.size(0))
#                 total_top1.update(top1.item(), data.size(0))
#                 total_top5.update(top5.item(), data.size(0))
#         return obj, total_top1, total_top5

#     def get_clients_weight(self):
#         """
#         Get the weights of selected clients for each round.
#         :return:
#         """
#         clients_weight = {}
#         total_length = 0
#         for client in self.selected_clients:
#             total_length += len(self.dataset['train'].indices[client])
#         for client in self.selected_clients:
#             clients_weight[client] = torch.tensor(len(self.dataset['train'].indices[client]) / total_length)
#         return clients_weight

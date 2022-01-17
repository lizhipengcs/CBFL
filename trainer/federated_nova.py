import random
import copy
import torch
from utils.utils import TimeRecorder, save_checkpoint
from utils.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from trainer.base_trainer import BaseTrainer
from utils.loss_function import loss_cross_entropy

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from torchutils.metrics import AverageMetric, AccuracyMetric
from torchutils.distributed import is_master, rank, world_size, local_rank
import torch.distributed as tdist
from utils.utils import st_all_reduce, coff_all_reduce, st_broadcast, partition, st_copy
from collections import OrderedDict


class FederatedNova(BaseTrainer):

    def __init__(self, params):
        self.params = params
        self.num_clients = params.get('num_clients', 1)
        self.num_classes = params.get('num_classes', 10)

        self.sampled_ratio = params.get('sampled_ratio', 1)
        self.start_round = params.get('start_round', 1)
        self.total_round = params.get('total_round', 0)
        self.local_round = params.get('local_round', 0)
        self.global_model = params.get('global_model')
        self.dataset = params.get('dataset')
        self.lr = params.get('lr', 0.1)

        self.momentum = params.get('momentum', 0.9)
        self.weight_decay = params.get('weight_decay', 0.0005)
        self.batch_size = params.get('batch_size', 64)
        self.num_workers = params.get('num_workers', 2)
        self.device = params.get('device', 'cuda')

        self.model_name = params.get('model_name', '')
        self.data_name = params.get('data_name', '')
        self.data_path = params.get('data_path', '')
        self.experimental_name = params.get('name', 'debug')
        self.resumed_model = params.get('resumed_model', False)
        self.resumed_reset = params.get('resumed_reset', False)
        self.save_interval = params.get('save_interval', 50)
        seed = params.get('seed', 7777)

        experimental_name = f'{self.params["aggregation_type"]}_{self.data_name}_{self.params["partition_type"]}_' \
                            f'{self.model_name}_{seed}/{self.experimental_name}'
        super().__init__(experimental_name, seed)
        self.selected_clients = []
        self.local_model = None
        self.client_train_loader = {}
        self.client_test_loader = {}
        self.test_loader = None

        self.get_data_loaders()

    def run(self):
        self.logger.info('federated training...')

        # recode training time
        time_recoder = TimeRecorder(self.start_round, self.total_round, self.logger)

        # adjust lr
        lr_scheduler = StepLR(self.lr, self.params['lr_decay'], self.params['lr_decay_interval'])

        best_acc = 0
        best_round = -1

        global_delta_weight = None
        coeff = None

        model = copy.deepcopy(self.global_model)

        for federated_round in range(self.start_round, self.total_round + 1):

            st_broadcast(self.global_model.state_dict())

            self.client_selection()
            assert len(self.selected_clients) > 0

            clients_weight = self.get_clients_weight()
            if global_delta_weight is None:
                global_delta_weight = OrderedDict()
                for name, data in self.global_model.state_dict().items():
                    if 'num_batches_tracked' in name:
                        continue
                    global_delta_weight[name] = torch.zeros_like(data)
            else:
                for name, data in global_delta_weight.items():
                    data.data.zero_()
            if coeff is None:
                coeff = torch.tensor([0.0]).cuda()
            else:
                coeff.zero_()
            n_clients = int(self.num_clients * self.sampled_ratio)
            task_ids = partition(list(range(n_clients)), world_size())

            client_ids = [self.selected_clients[task_id] for task_id in task_ids[rank()]]

            for task_id, client_id in zip(task_ids[rank()], client_ids):

                st_copy(src=self.global_model.state_dict(), dst=model.state_dict())
                optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum,
                                            weight_decay=self.weight_decay)
                data_iterator = self.client_train_loader[client_id]
                tau = 0
                for internal_epoch in range(1, self.local_round + 1):
                    model.train()
                    for batch_id, (data, targets) in enumerate(data_iterator):
                        data = data.to(self.device)
                        targets = targets.to(self.device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = loss_cross_entropy(output, targets)
                        loss.backward()
                        optimizer.step()
                        tau += 1
                a_i = (tau - self.momentum * (1 - pow(self.momentum, tau)) / (1 - self.momentum)) / (1 - self.momentum)

                local_state_dict = model.state_dict()
                global_model_para = self.global_model.state_dict()
                for name in global_delta_weight.keys():
                    if 'num_batches_tracked' in name:
                        continue
                    if 'bn' in name:
                        global_delta_weight[name].add_((global_model_para[name] - local_state_dict[name]) * clients_weight[client_id])
                    else:
                        global_delta_weight[name].add_(torch.true_divide(global_model_para[name] - local_state_dict[name], a_i)* clients_weight[client_id])

                coeff += a_i * clients_weight[client_id]

            tdist.barrier()
            st_all_reduce(global_delta_weight)
            coff_all_reduce(coeff)

            updated_model = self.global_model.state_dict()
            for key in updated_model:
                if 'num_batches_tracked' in key:
                    continue
                elif 'bn' in key:
                    updated_model[key] -= global_delta_weight[key]
                else:
                    updated_model[key] -= coeff.item() * global_delta_weight[key]

            st_copy(src=updated_model, dst=self.global_model.state_dict())

            round_loss, round_acc = self.distributed_evaluation(
                self.global_model, self.dataset["test"].dataset, self.device
            )

            self.lr = lr_scheduler.update(federated_round)

            self.writer.add_scalar(f'test/round_loss', round_loss, federated_round)
            self.writer.add_scalar(f'test/round_acc', round_acc/100.0, federated_round)

            state_dict = {'state_dict': self.global_model.state_dict(), 'round': federated_round, 'lr': self.lr}
            if is_master():
                if round_acc > best_acc:
                    best_acc, best_round = round_acc, federated_round
                    save_checkpoint(state_dict, True, self.save_folder)
                else:
                    save_checkpoint(state_dict, False, self.save_folder)
                if federated_round % self.save_interval == 0:
                    checkpoint_name = f"{self.save_folder}/checkpoint_{federated_round}.pth.tar"
                    torch.save(state_dict, checkpoint_name)
            time_recoder.update()
            self.logger.info(
                f"Round {federated_round}, lr:{self.lr}, acc:{round_acc:.2f}%,"
                f"@Best:({best_acc:.2f}%, {best_round})")

    def client_selection(self):
        """
        Randomly sample clients.
        """
        num_sampled = int(self.num_clients * self.sampled_ratio)
        self.selected_clients = random.sample(range(self.num_clients), num_sampled)

    def aggregation(self):
        """
        By default, we consider there are no stragglers and other techniques.
        """
        pass

    def get_data_loaders(self):
        for i in range(self.num_clients):
            for split in ['train', 'test']:
                dataset = self.dataset[split]
                len_samples = len(dataset.indices[i])
                # To avoid "ValueError('Expected more than 1 value per channel when training,
                # got input size {}'.format(size))" when using batchnorm.
                if len_samples % self.batch_size == 1:
                    dataset.indices[i] = dataset.indices[i][:-1]
                if len_samples > 0:
                    subset = Subset(dataset.dataset, dataset.indices[i])
                    data_loader = DataLoader(subset, batch_size=self.batch_size,
                                             num_workers=self.num_workers,
                                             shuffle=True if split == 'train' else False)
                    if split == 'train':
                        self.client_train_loader[i] = data_loader
                    else:
                        self.client_test_loader[i] = data_loader

        self.test_loader = DataLoader(self.dataset['test'].dataset, batch_size=self.batch_size,
                                      num_workers=self.num_workers,
                                      shuffle=False)

    def get_client_datasets(self):
        for i in range(self.num_clients):
            dataset = self.dataset['train']
            len_samples = len(dataset.indices[i])
            # To avoid "ValueError('Expected more than 1 value per channel when training,
            # got input size {}'.format(size))" when using batchnorm.
            if len_samples % self.batch_size == 1:
                dataset.indices[i] = dataset.indices[i][:-1]
            if len_samples > 0 and len_samples != 1:
                subset = Subset(dataset.dataset, dataset.indices[i])
                self.client_train_loader[i] = subset

    # @staticmethod
    def train(self, model, optimizer, data_loader, device):
        model.train()
        for batch_id, (data, targets) in enumerate(data_loader):
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_cross_entropy(output, targets)
            loss.backward()
            optimizer.step()

    def get_clients_weight(self):
        """
        Get the weights of selected clients for each round.
        :return:
        """
        clients_weight = {}
        total_length = 0
        for client in self.selected_clients:
            total_length += len(self.dataset['train'].indices[client])
        for client in self.selected_clients:
            clients_weight[client] = torch.tensor(len(self.dataset['train'].indices[client]) / total_length)
        return clients_weight

    def distributed_evaluation(self, model, dataset, device):
        dist_model = DDP(model, device_ids=[local_rank()])
        dist_model.eval()
        dist_data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, pin_memory=False,
                sampler=DistributedSampler(dataset, shuffle=False), num_workers=self.num_workers)
        loss_metric = AverageMetric()
        accuracy_metric = AccuracyMetric(topk=(1,))

        critirion = torch.nn.CrossEntropyLoss()
        for iter_, (inputs, targets) in enumerate(dist_data_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                logits = dist_model(inputs)
                loss = critirion(logits, targets)

            loss_metric.update(loss)
            accuracy_metric.update(logits, targets)
        return loss_metric.compute(), accuracy_metric.at(1).rate*100
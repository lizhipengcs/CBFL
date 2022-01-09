import random
import copy
import torch
from utils.utils import TimeRecorder, save_checkpoint
from utils.lr_scheduler import StepLR, CosinLR
from torch.utils.data import DataLoader, Subset, Dataset
from trainer.base_trainer import BaseTrainer
from utils.utils import AverageMeter, accuracy
from utils.loss_function import loss_cross_entropy, criterion_prox


from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# from .base_trainer import SHARED_TASK_FILE

from torchutils.metrics import AverageMetric, AccuracyMetric
from torchutils.distributed import is_master, rank, world_size, local_rank
# from utils.task_allocate import find_first_available_task, reset_task_file
import torch.distributed as tdist
from torchutils.metrics import _all_reduce
from utils.utils import st_all_reduce, st_broadcast, partition, st_copy
from collections import OrderedDict


class FederatedProx(BaseTrainer):

    def __init__(self, params):
        self.params = params
        # self.multi_gpus = torch.cuda.device_count() > 1
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
        self.mu = params.get('mu', 0.001)
        self.global_model.eval()
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

        best_top1 = 0
        best_top5 = 0
        best_round = -1

        global_weight = None

        model = copy.deepcopy(self.global_model)

        for federated_round in range(self.start_round, self.total_round + 1):

            st_broadcast(self.global_model.state_dict())

            self.client_selection()
            assert len(self.selected_clients) > 0

            clients_weight = self.get_clients_weight()
            if global_weight is None:
                global_weight = OrderedDict()
                for name, data in self.global_model.state_dict().items():
                    if 'num_batches_tracked' in name:
                        continue
                    global_weight[name] = torch.zeros_like(data)
            else:
                for name, data in global_weight.items():
                    data.data.zero_()

            n_clients = int(self.num_clients * self.sampled_ratio)
            task_ids = partition(list(range(n_clients)), world_size())

            client_ids = [self.selected_clients[task_id] for task_id in task_ids[rank()]]

            for task_id, client_id in zip(task_ids[rank()], client_ids):
                st_copy(src=self.global_model.state_dict(), dst=model.state_dict())

                optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum,
                                            weight_decay=self.weight_decay)

                data_iterator = self.client_train_loader[client_id]
                for internal_epoch in range(1, self.local_round + 1):
                    self.local_train(model, optimizer, data_iterator, self.device)
                local_state_dict = model.state_dict()
                for name in global_weight.keys():
                    global_weight[name].add_(local_state_dict[name] * clients_weight[client_id])

            tdist.barrier()
            st_all_reduce(global_weight)

            st_copy(src=global_weight, dst=self.global_model.state_dict())

            round_loss, round_top1, round_top5 = self.distributed_evaluation(
                self.global_model, self.dataset["test"].dataset, self.device
            )

            if lr_scheduler is not None:
                self.lr = lr_scheduler.update(federated_round)

            self.writer.add_scalar(f'test/round_loss', round_loss, federated_round)
            self.writer.add_scalar(f'test/round_top1', round_top1/100.0, federated_round)
            self.writer.add_scalar(f'test/round_top5', round_top5/100.0, federated_round)

            state_dict = {'state_dict': self.global_model.state_dict(), 'round': federated_round, 'lr': self.lr}
            if is_master():
                if round_top1 > best_top1:
                    best_top1, best_top5, best_round = round_top1, round_top5, federated_round
                    save_checkpoint(state_dict, True, self.save_folder)
                else:
                    save_checkpoint(state_dict, False, self.save_folder)
                if federated_round % self.save_interval == 0:
                    checkpoint_name = f"{self.save_folder}/checkpoint_{federated_round}.pth.tar"
                    torch.save(state_dict, checkpoint_name)
            time_recoder.update()
            self.logger.info(
                f"Round {federated_round}, lr:{self.lr}, top1:{round_top1:.2f}%, top5:{round_top5:.2f}%, "
                f"@Best:({best_top1:.2f}%, {best_top5:.2f}%, {best_round})")

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

    def calculate_communication_cost(self):
        """
        Model propagation between the sever and local device will incur communication costs.
        We calculate total parameter propagated for all rounds by default.
        :return:
        """
        model_parameter = sum(map(lambda p: p.numel(), self.global_model.parameters()))
        total_cost = model_parameter * self.total_round * 2
        return total_cost

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

    def local_train(self, model, optimizer, data_loader, device):
        model.train()
        obj = AverageMeter()
        total_top1 = AverageMeter()
        total_top5 = AverageMeter()
        for batch_id, (data, targets) in enumerate(data_loader):
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_cross_entropy(output, targets) + criterion_prox(self.global_model, model, mu=self.mu)
            top1, top5 = accuracy(output, targets, topk=(1, 5))
            loss.backward()
            optimizer.step()
            obj.update(loss.item(), data.size(0))
            total_top1.update(top1.item(), data.size(0))
            total_top5.update(top5.item(), data.size(0))

        return obj, total_top1, total_top5

    @staticmethod
    def evaluation(model, data_loader, device):
        model.eval()
        obj = AverageMeter()
        total_top1 = AverageMeter()
        total_top5 = AverageMeter()
        with torch.no_grad():
            for batch_id, (data, targets) in enumerate(data_loader):
                data = data.to(device)
                targets = targets.to(device)
                output = model(data)
                loss = loss_cross_entropy(output, targets)
                top1, top5 = accuracy(output, targets, topk=(1, 5))
                obj.update(loss.item(), data.size(0))
                total_top1.update(top1.item(), data.size(0))
                total_top5.update(top5.item(), data.size(0))
        return obj, total_top1, total_top5

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
        # dataset = data_loader.dataset
        dist_data_loader = torch.utils.data.DataLoader(
                dataset, batch_size=self.batch_size, pin_memory=False,
                sampler=DistributedSampler(dataset, shuffle=False), num_workers=self.num_workers)
        loss_metric = AverageMetric()
        accuracy_metric = AccuracyMetric(topk=(1, 5))

        critirion = torch.nn.CrossEntropyLoss()
        for iter_, (inputs, targets) in enumerate(dist_data_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                logits = dist_model(inputs)
                loss = critirion(logits, targets)

            loss_metric.update(loss)
            accuracy_metric.update(logits, targets)
        return loss_metric.compute(), accuracy_metric.at(1).rate*100, accuracy_metric.at(5).rate*100
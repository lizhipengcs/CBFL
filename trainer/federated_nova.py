# from math import dist, floor
import random
import copy
import torch
from tqdm import tqdm
from utils.utils import TimeRecorder, save_checkpoint
from utils.lr_scheduler import StepLR, CosinLR
from torch.utils.data import DataLoader, Subset, Dataset
from trainer.base_trainer import BaseTrainer
from utils.utils import AverageMeter, accuracy
from utils.loss_function import loss_cross_entropy

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# from .base_trainer import SHARED_TASK_FILE

from torchutils.metrics import AverageMetric, AccuracyMetric
from torchutils.distributed import is_master, rank, world_size, local_rank
# from utils.task_allocate import find_first_available_task, reset_task_file
import torch.distributed as tdist
from torchutils.metrics import _all_reduce
from utils.utils import st_all_reduce, coff_all_reduce, st_broadcast, partition, st_copy
from collections import OrderedDict

from PIL import Image
from torchvision import transforms
import os


def data_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.Resize((64,64)),
        # transforms.Scale(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    return transform_train, transform_test


class InaturalistDataset(Dataset):
    def __init__(self, root, data_pairs, transforms):
        self.root = root
        self.data_pairs = data_pairs
        self.transforms = transforms

    def __getitem__(self, index):
        data_path, label_name, label = self.data_pairs[index]
        image = Image.open(os.path.join(
            self.root, data_path)).convert('RGB')

        image = self.transforms(image)

        return image, label

    def __len__(self):
        return len(self.data_pairs)


class FederatedNovaImageNet(BaseTrainer):
    """
    The Federated Averaging algorithm.
    "Communication-Efficient Learning of Deep Networks from Decentralized Data".
    https://arxiv.org/abs/1602.05629
    """

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
        self.test_type = params.get('test_type', 'server')
        self.lr = params.get('lr', 0.1)
        self.lr_finetune = params.get('lr_finetune', 0.01)

        self.momentum = params.get('momentum', 0.9)
        self.weight_decay = params.get('weight_decay', 0.0005)
        self.batch_size = params.get('batch_size', 64)
        self.num_workers = params.get('num_workers', 2)
        self.device = params.get('device', 'cuda')

        self.model_name = params.get('model_name', '')
        self.model_depth = params.get('model_depth', '')
        self.data_name = params.get('data_name', '')
        self.data_path = params.get('data_path', '')
        self.experimental_name = params.get('name', 'debug')
        self.resumed_model = params.get('resumed_model', False)
        self.resumed_reset = params.get('resumed_reset', False)
        self.save_interval = params.get('save_interval', 50)
        self.optimizer_type = params.get('optimizer_type', 'SGD')
        seed = params.get('seed', 7777)

        experimental_name = f'{self.params["aggregation_type"]}_{self.data_name}_{self.params["partition_type"]}_' \
                            f'{self.model_name}{self.model_depth}_{seed}/{self.experimental_name}'
        super().__init__(experimental_name, seed)
        self.selected_clients = []
        self.local_model = None
        self.client_train_loader = {}
        self.client_val_loader = {}
        self.client_test_loader = {}
        self.test_loader = None
        self.local_model_global_test = params.get('local_model_global_test', False)
        self.local_model_local_test = params.get('local_model_local_test', False)

        self.get_data_loaders()

    def run(self):
        self.logger.info('federated training...')
        # round_loss, round_top1, round_top5 = self.model_evaluation()
        # f"lr:{self.lr}, top1:{round_top1:.2f}, top5:{round_top5:.2f}"

        # recode training time
        time_recoder = TimeRecorder(self.start_round, self.total_round, self.logger)

        # adjust lr
        if self.params['lr_scheduler'] == 'decay':
            lr_scheduler = StepLR(self.lr, self.params['lr_decay'], self.params['lr_decay_interval'])
        elif self.params['lr_scheduler'] == 'cosin':
            lr_scheduler = CosinLR(self.total_round, self.lr, 0.0001)
        else:
            lr_scheduler = None

        writer_list = ['round_top1', 'round_top5', 'round_loss']
        best_top1 = 0
        best_top5 = 0
        best_round = -1

        global_weight = None
        coeff = None

        model = copy.deepcopy(self.global_model)

        # from line_profiler import LineProfiler
        # lp = LineProfiler()
        # lp_wrapper = lp(self.train)

        for federated_round in range(self.start_round, self.total_round + 1):
            # sync global parameter
            # print(f"rank={rank()}, start to sync global model.")

            st_broadcast(self.global_model.state_dict())

            # if self.resumed_model and not self.resumed_reset:
            #     self.lr = lr_scheduler.update(federated_round)

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
            if coeff is None:
                coeff = torch.tensor([0.0]).cuda()
            else:
                coeff.zero_()
            n_clients = int(self.num_clients * self.sampled_ratio)
            task_ids = partition(list(range(n_clients)), world_size())
            # for i in range(world_size()):
            # print(f"rank={rank()}, task_ids={task_ids[rank()]}")
            client_ids = [self.selected_clients[task_id] for task_id in task_ids[rank()]]
            # if is_master():
                # reset_task_file(n=int(self.num_clients * self.sampled_ratio))
            
            # task_id = find_first_available_task(rank=rank())
            # client_id = self.selected_clients[task_id]
            # while client_id != None:

            if self.local_model_global_test:
                total_global_obj, total_global_top1, total_global_top5 = 0, 0, 0
            if self.local_model_local_test:
                total_local_obj, total_local_top1, total_local_top5 = 0, 0, 0

            for task_id, client_id in zip(task_ids[rank()], client_ids):
                # print(f"rank={rank()}, task_id={task_id}, cliend_id={client_id}")
            # for client_id in self.selected_clients:
                st_copy(src=self.global_model.state_dict(), dst=model.state_dict())
                if self.optimizer_type == 'SGD':
                    optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum,
                                                weight_decay=self.weight_decay)
                else:
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.lr,
                                                 weight_decay=self.weight_decay, amsgrad=True)
                data_iterator = self.client_train_loader[client_id]
                tau = 0
                for internal_epoch in range(1, self.local_round + 1):
                    # lp_wrapper(model, optimizer, data_iterator, self.device)
                    # self.train(model, optimizer, data_iterator, self.device)
                    self.global_model.eval()
                    model.train()
                    obj = AverageMeter()
                    total_top1 = AverageMeter()
                    total_top5 = AverageMeter()
                    for batch_id, (data, targets) in enumerate(data_iterator):
                        data = data.to(self.device)
                        targets = targets.to(self.device)
                        optimizer.zero_grad()
                        output = model(data)
                        loss = loss_cross_entropy(output, targets)
                        top1, top5 = accuracy(output, targets, topk=(1, 5))
                        loss.backward()
                        optimizer.step()
                        obj.update(loss.item(), data.size(0))
                        total_top1.update(top1.item(), data.size(0))
                        total_top5.update(top5.item(), data.size(0))
                        tau += 1
                a_i = (tau - self.momentum * (1 - pow(self.momentum, tau)) / (1 - self.momentum)) / (1 - self.momentum)
                # a_i = 1
                # print('a_i', a_i, 'w_id:', clients_weight[client_id])
                local_state_dict = model.state_dict()
                global_model_para = self.global_model.state_dict()
                for name in global_weight.keys():
                    if 'num_batches_tracked' in name:
                        continue
                    # if 'running_mean' in name or 'running_var' in name:
                    #     global_weight[name].add_(
                    #         (global_model_para[name] - local_state_dict[name]) * clients_weight[client_id])

                    if 'bn' in name:
                        global_weight[name].add_((global_model_para[name] - local_state_dict[name]) * clients_weight[client_id])
                    else:
                        global_weight[name].add_(torch.true_divide(global_model_para[name] - local_state_dict[name], a_i)* clients_weight[client_id])

                    # global_weight[name].add_(local_state_dict[name] * clients_weight[client_id])
                coeff += a_i * clients_weight[client_id]
                # test local model
                if self.local_model_global_test:
                    local_obj, local_top1, local_top5 = self.evaluation(model,
                                                                        self.test_loader, self.device)
                    total_global_obj += local_obj.avg * self.sampled_ratio
                    total_global_top1 += local_top1.avg * self.sampled_ratio
                    total_global_top5 += local_top5.avg * self.sampled_ratio
                    self.logger.info(f"global_obj:{local_obj.avg:.2f}, global_top1:{local_top1.avg:.2f}, "
                                     f"global_top5:{local_top5.avg:.2f}")

                if self.local_model_local_test:
                    local_obj, local_top1, local_top5 = self.evaluation(model,
                                                                        self.client_test_loader[client_id], self.device)
                    total_local_obj += local_obj.avg * self.sampled_ratio
                    total_local_top1 += local_top1.avg * self.sampled_ratio
                    total_local_top5 += local_top5.avg * self.sampled_ratio
                    self.logger.info(f"local_obj:{local_obj.avg:.2f}, local_top1:{local_top1.avg:.2f}, "
                                     f"local_top5:{local_top5.avg:.2f}")

                # task_id = find_first_available_task(rank=rank())
                # if task_id is None:
                    # break
                # client_id = self.selected_clients[task_id]
            tdist.barrier()
            # print(f"rank={rank()}, start to reduce local weight.")
            st_all_reduce(global_weight)
            coff_all_reduce(coeff)
            # print('coeff', coeff)
            # print(f"rank={rank()}, complete to reduce local weight.")
            if self.local_model_global_test:
                global_obj, global_top1, global_top5 = _all_reduce(total_global_obj, total_global_top1, total_global_top5)
                info_str = f"Epoch: {federated_round}, total_global_obj: {global_obj}, total_global_top1: {global_top1}, " \
                           f"total_global_top5: {global_top5}"
                self.logger.info(info_str)
            if self.local_model_local_test:
                local_obj, local_top1, local_top5 = _all_reduce(total_local_obj, total_local_top1, total_local_top5)
                info_str = f"Epoch: {federated_round}, total_local_obj: {local_obj}, total_local_top1: {local_top1}, " \
                           f"total_local_top5: {local_top5}"
                self.logger.info(info_str)

            updated_model = self.global_model.state_dict()
            for key in updated_model:
                # #print(updated_model[key])
                # if updated_model[key].type() == 'torch.LongTensor':
                #     updated_model[key] -= (coeff * d_total_round[key]).type(torch.LongTensor)
                # elif updated_model[key].type() == 'torch.cuda.LongTensor':
                #     updated_model[key] -= (coeff * d_total_round[key]).type(torch.cuda.LongTensor)
                # else:
                #     #print(updated_model[key].type())
                #     #print((coeff*d_total_round[key].type()))

                if 'num_batches_tracked' in key:
                    continue
                # elif 'running_mean' in key or 'running_var' in key:
                #     updated_model[key] -= global_weight[key]
                #

                elif 'bn' in key:
                    updated_model[key] -= global_weight[key]
                else:
                    updated_model[key] -= coeff.item() * global_weight[key]
                # updated_model[key] -= 1 * global_weight[key]
            # global_model.load_state_dict(updated_model)

            st_copy(src=updated_model, dst=self.global_model.state_dict())
            # self.global_model.load_state_dict(global_weight)
            # print(self.global_model.state_dict())
            round_loss, round_top1, round_top5 = self.distributed_evaluation(
                self.global_model, self.dataset["test"].dataset, self.device
            )

            if lr_scheduler is not None:
                self.lr = lr_scheduler.update(federated_round)

            # for wl in writer_list:
            self.writer.add_scalar(f'test/round_loss', round_loss, federated_round)
            self.writer.add_scalar(f'test/round_top1', round_top1/100.0, federated_round)
            self.writer.add_scalar(f'test/round_top5', round_top5/100.0, federated_round)
            if self.local_model_global_test:
                self.writer.add_scalar(f'local_test/round_loss', global_obj, federated_round)
                self.writer.add_scalar(f'local_test/round_top1', global_top1 / 100.0, federated_round)
                self.writer.add_scalar(f'local_test/round_top5', global_top5 / 100.0, federated_round)
            if self.local_model_local_test:
                self.writer.add_scalar(f'local_test/round_loss', local_obj, federated_round)
                self.writer.add_scalar(f'local_test/round_top1', local_top1 / 100.0, federated_round)
                self.writer.add_scalar(f'local_test/round_top5', local_top5 / 100.0, federated_round)

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
            # if lr_scheduler is not None:
            #     self.lr = lr_scheduler.update(federated_round)
        # lp.print_stats()

    def client_selection(self):
        """
        Randomly sample clients.
        """
        num_sampled = int(self.num_clients * self.sampled_ratio)
        self.selected_clients = random.sample(range(self.num_clients), num_sampled)

    # def broadcast(self, hook_flag=True):
    #     """
    #     The server broadcasts the model weight to selected clients.
    #     """
    #     assert len(self.selected_clients) > 0
    #     for client_id in self.selected_clients:
    #         if hook_flag:
    #             self.local_models[client_id] = copy.deepcopy(self.global_model)
    #         else:
    #             self.local_models[client_id].load_state_dict(self.global_model.state_dict())

    # def client_computation(self, round=0):
    #     """
    #     Selected devices locally run SGD on their private data.
    #     """
    #     # Local training
    #     # TODO: How to apply DistributedDataParallel to federated training.
    #     for client in self.selected_clients:
    #         model = self.local_models[client]
    #         if torch.cuda.device_count() > 1:
    #             model = torch.nn.DataParallel(model).cuda()
    #         else:
    #             model = model.to(self.device)
    #         optimizer = torch.optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum,
    #                                     weight_decay=self.weight_decay)
    #         data_iterator = self.client_train_loader[client]
    #         for internal_epoch in range(1, self.local_round + 1):
    #             self.train(model, optimizer, data_iterator, self.device)

    def aggregation(self):
        """
        By default, we consider there are no stragglers and other techniques.
        """
        pass

    # def model_update(self, round=0):
    #     """
    #     The server updates global model by weighted averaging the model weight of selected clients.
    #     """
    #     assert len(self.selected_clients) > 0
    #     clients_weight = self.get_clients_weight()
    #     global_weight = dict()
    #     for name, data in self.global_model.state_dict().items():
    #         if 'num_batches_tracked' in name:
    #             continue
    #         global_weight[name] = torch.zeros_like(data)
    #     for client_id in self.selected_clients:
    #         local_state_dict = self.local_models[client_id].state_dict()
    #         for name in global_weight.keys():
    #             global_weight[name].add_(local_state_dict[name] * clients_weight[client_id])
    #     self.global_model.load_state_dict(global_weight)

    # def model_evaluation(self):
        # """
        # There exist two evaluation methods.
        # 1. Evaluate the global model on a shared testing dataset.
        # 2. We average the performance of all clients, in which client evaluates the model on their private dataset.
        # """

        # obj, top1, top5 = self.distributed_evaluation(self.global_model, self.test_loader, self.device)
        # return obj, top1, top5
        # final_obj, final_top1, final_top5 = obj.avg, top1.avg, top5.avg

        # return final_obj, final_top1, final_top5

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
        if self.data_name != 'inaturalist':
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
        else:
            transform_train, transform_test = data_transform()
            for i in range(self.num_clients):
                for split in ['train']:
                    dataset = self.dataset[split]
                    # len_samples = len(dataset.indices[i])
                    subset = InaturalistDataset(self.data_path, dataset.indices[i], transform_train)
                    data_loader = DataLoader(subset, batch_size=self.batch_size,
                                             num_workers=self.num_workers,
                                             shuffle=True if split == 'train' else False)
                    # if split == 'train':
                    self.client_train_loader[i] = data_loader
                    # else:
                    #     self.client_test_loader[i] = data_loader
            test_result = []
            for k in self.dataset['test'].indices.keys():
                test_result.extend(self.dataset['test'].indices[k])
            testset = InaturalistDataset(self.data_path, test_result, transform_test)
            self.dataset['test'].dataset = testset
            self.test_loader = DataLoader(testset, batch_size=self.batch_size,
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
        self.global_model.eval()
        model.train()
        obj = AverageMeter()
        total_top1 = AverageMeter()
        total_top5 = AverageMeter()
        for batch_id, (data, targets) in enumerate(data_loader):
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_cross_entropy(output, targets)
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
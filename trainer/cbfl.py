import random
import copy
import torch
from models.generator import Conditional_Generator
from trainer.federated_averaging import FederatedAveraging
from utils.lr_scheduler import StepLR
import torch.nn as nn
from utils.utils import AverageMeter, accuracy, save_checkpoint, TimeRecorder, ModelHook
from utils.loss_function import *
import numpy as np
from collections import OrderedDict


from torch.nn.parallel import DistributedDataParallel as DDP
from torchutils.distributed import is_master, rank, world_size
import torch.distributed as tdist

from torchutils.metrics import _all_reduce

from utils.utils import st_all_reduce, st_broadcast, partition, st_copy


class CBFL(FederatedAveraging):

    def __init__(self, params):
        super().__init__(params)
        # generator
        self.img_size = params.get('img_size', 32)
        self.img_channels = params.get('img_channels', 1)
        self.latent_dim = params.get('latent_dim', 100)
        self.lr_G = params.get('lr_G', 0.001)
        self.iterations = params.get('iterations', 50)
        self.first_iterations = params.get('first_iterations', 50)
        self.gen_interval = params.get('gen_interval', 3)
        self.gen_batch_size = params.get('gen_batch_size', 256)
        self.fake_batch_size = params.get('fake_batch_size', 50)
        self.warmup_round = params.get('warmup_round', 50)
        self.alpha_fake = params.get('alpha_fake', 1)
        self.AT_w = params.get('AT_w', 400)
        self.BNS_w = params.get('BNS_w', 10)
        self.KD_w = params.get('KD_w', 1)
        self.gen_CE_w = params.get('gen_CE_w', 1)
        self.temperature = params.get('temperature', 5)

        self.first_iter_flag = True
        self.generator = None
        self.optimizer_G = None
        self.reverse_label_distribution = None

        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []
        self.set_generator()
        self.pre_global_model = copy.deepcopy(self.global_model)
        self.local_model = copy.deepcopy(self.global_model)
        self.hook_global = None
        self.hook_local = None

    def set_generator(self):
        generator = Conditional_Generator(
            self.num_classes, self.latent_dim, self.img_size, self.img_channels)
        generator = generator.to(self.device)
        self.generator = generator
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(), lr=self.lr_G, betas=(0.5, 0.999))

    def hook_fn_forward(self, module, input, output):
        input = input[0]
        mean = input.mean([0, 2, 3])
        var = input.var([0, 2, 3], unbiased=False)
        self.mean_list.append(mean)
        self.var_list.append(var)
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)

    def run(self):
        self.logger.info('federated training...')
        # recode training time
        time_recoder = TimeRecorder(
            self.start_round, self.total_round, self.logger)

        # adjust lr
        lr_scheduler = StepLR(
            self.lr, self.params['lr_decay'], self.params['lr_decay_interval'])

        best_round = -1
        best_acc = 0

        self.reverse_label_distribution = self.dataset['train'].get_reverse_label_distribution(
        )

        # set hook for BN layers
        for m in self.pre_global_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_forward_hook(self.hook_fn_forward)
        # set hook to get feature
        self.hook_global = ModelHook(
            self.pre_global_model, self.model_name, self.num_classes)
        self.hook_local = ModelHook(
            self.local_model, self.model_name, self.num_classes)

        global_weight = None

        for federated_round in range(self.start_round, self.total_round + 1):
            st_broadcast(self.global_model.state_dict())

            self.client_selection()
            assert len(self.selected_clients) > 0
            clients_weight = self.get_clients_weight()

            if (federated_round >= self.warmup_round) and ((federated_round-self.warmup_round) % self.gen_interval == 0):
                perform = True
            else:
                perform = False

            if perform:
                st_copy(src=self.global_model.state_dict(),
                        dst=self.pre_global_model.state_dict())
                self.set_generator()
                self.train_conditional_generator(federated_round)

            if global_weight is None:
                global_weight = OrderedDict()
                for name, data in self.global_model.state_dict().items():
                    if 'num_batches_tracked' in name:
                        continue
                    global_weight[name] = torch.zeros_like(data)
            else:
                for name, data in global_weight.items():
                    data.data.zero_()

            total_loss_true, total_loss_KD, total_loss_at = 0, 0, 0

            n_clients = int(self.num_clients * self.sampled_ratio)
            task_ids = partition(list(range(n_clients)), world_size())

            client_ids = [self.selected_clients[task_id]
                          for task_id in task_ids[rank()]]

            for task_id, client_id in zip(task_ids[rank()], client_ids):
                st_copy(src=self.global_model.state_dict(),
                        dst=self.local_model.state_dict())
                optimizer = torch.optim.SGD(self.local_model.parameters(), lr=self.lr, momentum=self.momentum,
                                            weight_decay=self.weight_decay)
                data_iterator = self.client_train_loader[client_id]
                for internal_epoch in range(1, self.local_round + 1):
                    if federated_round > self.warmup_round:
                        loss_true, loss_KD, loss_at = self.train_with_conditon_fake_data_concat(
                            self.local_model, optimizer, data_iterator, self.device,
                            self.reverse_label_distribution[client_id],
                            client_id)
                        total_loss_true += loss_true.avg * \
                            clients_weight[client_id]
                        total_loss_KD += loss_KD.avg * \
                            clients_weight[client_id]
                        total_loss_at += loss_at.avg * \
                            clients_weight[client_id]
                    else:
                        obj = self.local_train(
                            self.local_model, optimizer, data_iterator, self.device, client_id)
                        total_loss_true += obj.avg * clients_weight[client_id]

                local_state_dict = self.local_model.state_dict()
                for name in global_weight.keys():
                    global_weight[name].add_(
                        local_state_dict[name] * clients_weight[client_id])
            tdist.barrier()
            st_all_reduce(global_weight)
            total_loss_true, total_loss_KD, total_loss_at = \
                _all_reduce(total_loss_true, total_loss_KD, total_loss_at)

            st_copy(src=global_weight, dst=self.global_model.state_dict())

            info_str = f"Epoch: {federated_round}, loss_true: {total_loss_true}, " \
                       f"loss_KD: {total_loss_KD}, loss_at: {total_loss_at}"
            self.logger.info(info_str)

            round_loss, round_acc = self.model_evaluation()

            self.writer.add_scalar(
                f'test/round_loss', round_loss, federated_round)
            self.writer.add_scalar(
                f'test/round_acc', round_acc/100.0, federated_round)

            self.lr = lr_scheduler.update(federated_round)

            state_dict = {'state_dict': self.global_model.state_dict(),
                          'generator': self.generator.state_dict(),
                          'optimizer_G': self.optimizer_G.state_dict(),
                          'round': federated_round,
                          'lr': self.lr}
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
        self.selected_clients = random.sample(
            range(self.num_clients), num_sampled)

    def train_with_conditon_fake_data_concat(self, model, optimizer, data_loader, device, reverse_label_distribution, client):
        model.train()
        self.generator.eval()
        self.pre_global_model.eval()
        total_loss_true = AverageMeter()
        total_loss_KD = AverageMeter()
        total_loss_at = AverageMeter()

        for batch_id, (data, targets) in enumerate(data_loader):
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            client_label = np.random.choice(a=self.num_classes, size=self.fake_batch_size, replace=True,
                                            p=reverse_label_distribution)
            client_label = torch.Tensor(client_label)
            labels = torch.ones(self.fake_batch_size) * client_label
            labels = labels.long().to(self.device)
            labels = labels.contiguous()

            z = torch.randn(self.fake_batch_size,
                            self.latent_dim).to(self.device)
            z = z.contiguous()
            gen_imgs = self.generator(z, labels).detach()

            total_input = torch.cat([data, gen_imgs])

            output = model(total_input)
            loss_true = loss_cross_entropy(output[:data.size(0)], targets)

            global_output = self.pre_global_model(gen_imgs)
            loss_KD = loss_kd(output[data.size(0):],
                              global_output, T=self.temperature)

            global_layer_outputs = self.hook_global.get_layer_outputs()
            local_layer_outputs = self.hook_local.get_layer_outputs()

            loss_at = sum(
                AT(p=2)(local_layer_outputs[j][data.size(
                    0):], global_layer_outputs[j].detach())
                for j in range(len(global_layer_outputs)))

            loss = loss_true + self.alpha_fake * \
                (self.KD_w * loss_KD + loss_at * self.AT_w)

            loss.backward()
            optimizer.step()
            total_loss_true.update(loss_true.item(), data.size(0))
            total_loss_KD.update(loss_KD.item(), self.fake_batch_size)
            total_loss_at.update(loss_at, self.fake_batch_size)

            self.mean_list.clear()
            self.var_list.clear()
            self.teacher_running_mean.clear()
            self.teacher_running_var.clear()
            self.hook_global.layer_outputs.clear()
            self.hook_local.layer_outputs.clear()

        return total_loss_true, total_loss_KD, total_loss_at

    def aggregation(self):
        """
        By default, we consider there are no stragglers and other techniques.
        """
        pass

    def train_conditional_generator(self, round, select_local_id=0):
        loss_list = {'total_generator_loss': AverageMeter(),
                     'total_ce_loss': AverageMeter(),
                     'total_BNS_loss': AverageMeter()}

        self.pre_global_model.eval()
        self.generator.train()
        gen_batch_size = self.gen_batch_size

        if self.first_iter_flag:
            iterations = self.first_iterations
            self.first_iter_flag = False
        else:
            iterations = self.iterations

        for iter in range(iterations):
            self.mean_list.clear()
            self.var_list.clear()
            self.teacher_running_mean.clear()
            self.teacher_running_var.clear()

            z = torch.randn(gen_batch_size, self.latent_dim).to(self.device)
            z = z.contiguous()
            labels = torch.randint(0, self.num_classes,
                                   (gen_batch_size,)).to(self.device)
            labels = labels.contiguous()
            gen_imgs = self.generator(z, labels)

            outputs_T = self.pre_global_model(gen_imgs, out_feature=False)

            # CE loss
            ce_loss = torch.nn.CrossEntropyLoss()(outputs_T, labels)

            # BN statistic loss
            eps = 1e-6
            BNS_loss = torch.zeros(1).to(self.device)
            for num in range(len(self.mean_list)):
                BNS_loss += torch.sum(
                    torch.log(torch.sqrt(
                        self.teacher_running_var[num]) / torch.sqrt(self.var_list[num]))
                    + (self.var_list[num] + torch.pow(self.mean_list[num] - self.teacher_running_mean[num], 2)) / (
                        self.teacher_running_var[num] * 2+eps) - 0.5)
            BNS_loss = BNS_loss / len(self.mean_list)

            self.optimizer_G.zero_grad()
            generator_loss = self.gen_CE_w * ce_loss + self.BNS_w * BNS_loss
            generator_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.generators[idx].parameters(), 5)
            self.optimizer_G.step()

            loss_list['total_generator_loss'].update(
                generator_loss.item(), gen_batch_size)
            loss_list['total_ce_loss'].update(ce_loss.item(), gen_batch_size)
            loss_list['total_BNS_loss'].update(BNS_loss.item(), gen_batch_size)

            self.mean_list.clear()
            self.var_list.clear()
            self.teacher_running_mean.clear()
            self.teacher_running_var.clear()
            self.hook_global.layer_outputs.clear()

        info_str = f"[Epoch {round}] "
        for name, lss in loss_list.items():
            tmp_lss = name.replace('total_', '')
            self.writer.add_scalar(f'train/{tmp_lss}', lss.avg, round)
            info_str += f"[{tmp_lss} {lss.avg} ]"
        self.logger.info(info_str)

    def model_evaluation(self):
        obj, acc = self.distributed_evaluation(
            self.global_model, self.dataset["test"].dataset, self.device)
        return obj, acc

    def get_clients_weight(self):
        """
        Get the weights of selected clients for each round.
        """
        clients_weight = {}
        total_length = 0
        for client in self.selected_clients:
            total_length += len(self.dataset['train'].indices[client])
        for client in self.selected_clients:
            clients_weight[client] = len(
                self.dataset['train'].indices[client]) / total_length
        return clients_weight

    def local_train(self, model, optimizer, data_loader, device, client):
        model.train()
        obj = AverageMeter()
        for batch_id, (data, targets) in enumerate(data_loader):
            data = data.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_cross_entropy(output, targets)
            loss.backward()
            optimizer.step()
            obj.update(loss.item(), data.size(0))
            self.hook_local.layer_outputs.clear()
        return obj

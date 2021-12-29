from utils.helper import Helper
from models import *
from dataset import *
from utils.utils import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torchvision.models import resnet18, mobilenet_v2


class ImageHelper(Helper):
    def __init__(self, params, name, seed):
        super().__init__(params, name, seed)
        self.partition_type = params.get('partition_type', 'dirichlet')
        # Dirichlet
        self.balance = params.get('balance', False)
        self.dirichlet_alpha = params.get('dirichlet_alpha', 0.5)
        self.dirichlet_threshold = params.get('dirichlet_threshold', 0)
        # N-class Non-IID
        self.n_class = params.get('n_class', 1)
        # Server data
        self.server_ratio = params.get('server_ratio', 0)
        self.server_lr = params.get('server_lr', 0.1)
        self.server_batch_size = params.get('server_batch_size', 64)
        self.server_epoch = params.get('server_epoch', 600)
        # show distribution histogram
        self.show_distribution = params.get('show_distribution', False)

    def run(self):
        self.load_data()  # load data
        self.create_model()  # load model
        self.save_opts()  # save parameters
        save_code(self.repo_path, f"{self.save_folder}/code", ['results'])  # save code

    def create_model(self):
        self.logger.info('Loading model...')
        if self.data_name.lower() == 'mnist':
            if self.model_name.lower() == 'lenet':
                global_model = CNNMnist()
            else:
                raise NotImplementedError(f'No considering {self.model_name}')
        elif self.data_name.lower() == 'cifar':
            if self.model_name.lower() == 'lenet':
                global_model = CNNCifar()
            elif self.model_name.lower() == 'resnet':
                # global_model = eval(f"resnet{self.model_depth}()")
                global_model = PreResNet(self.model_depth)
            elif self.model_name.lower() == 'mobilenetv2':
                global_model = mobilenet_v2()
            else:
                raise NotImplementedError(f'No considering {self.model_name}')
        else:
            raise NotImplementedError(f'No considering {self.data_name}')

        if self.resumed_model:
            self.logger.info('load parameters')
            self.resumed_model = f"{self.repo_path}/{self.resumed_model}"
            loaded_params = torch.load(self.resumed_model)
            global_model.load_state_dict(loaded_params['state_dict'])
            if not self.resumed_reset:
                self.start_round = loaded_params['round']
                self.lr = loaded_params['lr']
                self.logger.info(f"Loaded parameters from saved model: LR is"
                                 f" {self.lr} and current round is {self.start_round}")

        global_model = global_model.to(self.device)
        self.global_model = global_model

    def load_data(self):
        self.logger.info('Loading data...')

        self.dataset['train'], self.dataset['test'] = build_dataset(self.partition_type,
                                                                    data_name=self.data_name,
                                                                    data_path=self.data_path,
                                                                    num_clients=self.num_clients,
                                                                    batch_size=self.batch_size,
                                                                    num_workers=self.num_workers,
                                                                    server_ratio=self.server_ratio,
                                                                    alpha=self.dirichlet_alpha,
                                                                    balance=self.balance,
                                                                    seed=self.seed,
                                                                    threshold=self.dirichlet_threshold,
                                                                    n_class=self.n_class
                                                                    )
        if self.show_distribution:
            self.dataset['train'].plot_distribution_histogram()
            self.dataset['test'].plot_distribution_histogram()

    def server_training(self, trainer):

        time_recoder = TimeRecorder(1, self.server_epoch, self.logger)
        dataset = self.dataset['train'].dataset
        indices = self.dataset['train'].indices['server']
        server_data_loader = DataLoader(Subset(dataset, indices), batch_size=self.server_batch_size,
                                        num_workers=2, shuffle=True)
        optimizer = torch.optim.SGD(self.global_model.parameters(), lr=self.server_lr, momentum=0.9,
                                    weight_decay=0.0005)
        server_lr_scheduler = CosineAnnealingLR(optimizer, self.server_epoch, eta_min=0.0001)
        writer_list = ['obj', 'total_top1', 'total_top5']
        best_top1 = 0
        best_epoch = -1
        for epoch in range(1, self.server_epoch + 1):
            self.global_model.train()
            print(id(self.global_model))
            print(id(trainer.global_model))
            trainer.train(self.global_model, optimizer, server_data_loader, self.device)
            obj, total_top1, total_top5 = trainer.model_evaluation()

            for wl in writer_list:
                self.writer.add_scalar(f'server_test/{wl}', eval(wl), epoch)

            if total_top1 > best_top1:
                best_top1 = total_top1
                best_epoch = epoch
                self.save_checkpoint(self.global_model.state_dict(), True,
                                     filename=f'{self.save_folder}/server_best_model.pth.tar')
            else:
                self.save_checkpoint(self.global_model.state_dict(), False,
                                     filename=f'{self.save_folder}/server_checkpoint.pth.tar')
            self.logger.info(
                f"Round {epoch}, lr:{server_lr_scheduler.get_lr()}, top1:{total_top1:.2f}, top5:{total_top5:.2f}"
                f"@Best:{best_top1}({best_epoch})")
            server_lr_scheduler.step()
            time_recoder.update()
        return best_top1

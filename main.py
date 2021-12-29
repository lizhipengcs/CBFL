import argparse
import yaml

from utils.utils import *
from trainer import *
from models import *
from dataset import build_dataset
# from torchvision.models import resnet18
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.utils import fix_random

from torch.nn.parallel import DistributedDataParallel as DDP

from torchutils import logger, output_directory
from torchutils.distributed import is_dist_avail_and_init, local_rank, is_master

# from utils.task_allocate import init_task_file, read_task_file, reset_task_file, clean_task_file

def main():
    parser = argparse.ArgumentParser(description='FederatedLearning')
    parser.add_argument('--params', dest='params', default='./configs/cifar/fed_gen_avg_cifar100_local5.yaml')
    parser.add_argument('--name', dest='name', default='debug')
    parser.add_argument('--seed', type=int, default=7777)
    # parser.add_argument('--gpus', type=str, default='7')

    args,_ = parser.parse_known_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus



    torch.cuda.set_device(local_rank())

    fix_random(args.seed)

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    # init_task_file()

    # if is_master():
        # reset_task_file(n=int(params.get('num_clients', 1)*params.get('sampled_ratio', 1)))
    params['name'] = args.name
    params['seed'] = args.seed
    params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_data(params)
    model = create_model(params['model_name'], params['model_depth'], params['data_name'], params['num_classes'])
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model).cuda()
    # else:
    #     model = model.to(params['device'])
    if params['resumed_model']:
        resumed_reset = params.get('resumed_reset', False)
        if not resumed_reset:
            start_round, lr = resume_model(model, params['resumed_model'], resumed_reset)
            params['start_round'], params['lr'] = start_round, lr
        else:
            resume_model(model, params['resumed_model'], resumed_reset)
    
    # global_model = model.cuda()
    # global_model.
    params['global_model'] = model.cuda()
    params['dataset'] = dataset
    trainer = build_trainer(**params)

    if is_master():
        save_opts(params, trainer.save_folder)
        # save_code(trainer.repo_path, f"{trainer.save_folder}/code", ['results'])

    if not params['only_eval']:
        # if is_master():
            
        #     from line_profiler import LineProfiler
        #     lp = LineProfiler()
        #     lp_wrapper = lp(trainer.run)
        #     lp_wrapper()
        #     lp.print_stats()
        # else:
        trainer.run()
    else:
        trainer.logger.info("Testing...")
        round_loss, round_top1, round_top5 = trainer.model_evaluation()
        trainer.logger.info(f"top1:{round_top1:.2f}, top5:{round_top5:.2f}, loss:{round_loss:.2f}")
    # clean_task_file()


def create_model(model_name, model_depth, data_name, num_classes):
    if data_name.lower() == 'mnist':
        if model_name.lower() == 'lenet':
            global_model = LeNet5()
        else:
            raise NotImplementedError(f'No considering {model_name}')
    elif data_name.lower() == 'cifar':
        if model_name.lower() == 'resnet':
            if num_classes == 10:
                global_model = resnet_cifar.resnet20_cifar10(10)
            elif num_classes == 100:
                global_model = resnet_cifar.resnet20_cifar100(100)
        elif model_name.lower() == 'preresnet':
            global_model = PreResNet(model_depth)
        elif model_name.lower() == 'vgg':
            global_model = vgg.VGG_CIFAR(depth=16, num_classes=num_classes)
        elif model_name.lower() == 'mobilenetv2':
            global_model = MobileNetV2Cifar(num_classes=num_classes)
        else:
            raise NotImplementedError(f'No considering {model_name}')
    elif data_name.lower() == 'imagenet':
        global_model = resnet.resnet18()
    elif data_name.lower() == 'inaturalist':
        if model_name.lower() == 'resnet':
            global_model = resnet.resnet18()
            global_model.fc = torch.nn.Linear(512, 1010)
        elif model_name.lower() == 'mobilenetv2':
            global_model = MobileNetV2Inaturalist(num_classes=1010)
        else:
            raise NotImplementedError(f'No considering {model_name}')
    elif data_name.lower() == 'cinic':
        if model_name.lower() == 'resnet':
            global_model = resnet_cifar.resnet20_cifar10(10)
        elif model_name.lower() == 'vgg':
            global_model = vgg.VGG_CIFAR(depth=16, num_classes=10)
        elif model_name.lower() == 'mobilenetv2':
            global_model = MobileNetV2Cifar(num_classes=10)
    else:
        raise NotImplementedError(f'No considering {data_name}')
    return global_model


def resume_model(model, resumed_path, resumed_reset=False):
    loaded_params = torch.load(resumed_path, map_location='cpu')
    state_dict = loaded_params['state_dict']
    model.load_state_dict(state_dict)
    # base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(state_dict.items())}
    # model.load_state_dict(base_dict)
    if not resumed_reset:
        start_round = loaded_params['round'] + 1
        lr = loaded_params['lr']
        return start_round, lr


def load_data(params):
    dataset = dict()
    dataset['train'], dataset['test'] = build_dataset(**params)
    if params['show_distribution']:
        # dataset['train'].plot_distribution_histogram()
        # dataset['test'].plot_distribution_histogram()
        temp = "/home/lizhipeng/project/FL_figures/cifar10"
        print(temp + "/train_" + str(params['dirichlet_alpha']))
        dataset['train'].plot_label_distribution_histogram(temp)
    return dataset


# def server_training(trainer, model, dataset, params):
#     logger = trainer.logger
#     writer = trainer.writer
#     logger.info('server training...')
#     time_recoder = TimeRecorder(1, params['server_epoch'], logger)
#     train_dataset = dataset['train'].dataset
#     indices = dataset['train'].indices['server']
#     server_data_loader = DataLoader(Subset(train_dataset, indices), batch_size=params['server_batch_size'],
#                                     num_workers=2, shuffle=True)
#     optimizer = torch.optim.SGD(model.parameters(), lr=params['server_lr'], momentum=0.9,
#                                 weight_decay=0.0005)
#     server_lr_scheduler = CosineAnnealingLR(optimizer, params['server_epoch'], eta_min=0.0001)
#     # load model
#     resumed_server_model = params.get('resumed_server_model', False)
#     if resumed_server_model:
#         loaded_params = torch.load(params['resumed_model'])
#         model.load_state_dict(loaded_params['state_dict'])
#         optimizer.load_state_dict(loaded_params['optimizer'])
#         server_lr_scheduler.load_state_dict(loaded_params['scheduler'])

#     writer_list = ['obj', 'total_top1', 'total_top5']
#     best_top1 = 0
#     best_epoch = -1
#     for epoch in range(1, params['server_epoch'] + 1):
#         trainer.train(model, optimizer, server_data_loader, params['device'])
#         obj, total_top1, total_top5 = trainer.model_evaluation()

#         for wl in writer_list:
#             writer.add_scalar(f'server_test/{wl}', eval(wl), epoch)
#         state_dict = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
#                       'scheduler': server_lr_scheduler.state_dict()}
#         if total_top1 > best_top1:
#             best_top1 = total_top1
#             best_epoch = epoch
#             save_checkpoint(state_dict, True, f"{trainer.save_folder}/server_training")
#         else:
#             save_checkpoint(state_dict, False, f"{trainer.save_folder}/server_training")
#         logger.info(
#             f"Round {epoch}, lr:{server_lr_scheduler.get_lr()}, top1:{total_top1:.2f}, top5:{total_top5:.2f}"
#             f"@Best:{best_top1}({best_epoch})")
#         server_lr_scheduler.step()
#         time_recoder.update()
#     return best_top1


if __name__ == '__main__':
    main()

import argparse
import yaml

from utils.utils import *
from trainer import *
from models import *
from dataset import build_dataset
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.utils import fix_random

from torch.nn.parallel import DistributedDataParallel as DDP

from torchutils import logger, output_directory
from torchutils.distributed import is_dist_avail_and_init, local_rank, is_master


def main():
    parser = argparse.ArgumentParser(description='FederatedLearning')
    parser.add_argument('--params', dest='params', default='./configs/cifar/fedavg_cifar100.yaml')
    parser.add_argument('--name', dest='name', default='debug')
    parser.add_argument('--seed', type=int, default=7777)

    args,_ = parser.parse_known_args()

    torch.cuda.set_device(local_rank())

    fix_random(args.seed)

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['name'] = args.name
    params['seed'] = args.seed
    params['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = load_data(params)
    model = create_model(params['model_name'], params['data_name'], params['num_classes'])

    if params['resumed_model']:
        resumed_reset = params.get('resumed_reset', False)
        if not resumed_reset:
            start_round, lr = resume_model(model, params['resumed_model'], resumed_reset)
            params['start_round'], params['lr'] = start_round, lr
        else:
            resume_model(model, params['resumed_model'], resumed_reset)

    params['global_model'] = model.cuda()
    params['dataset'] = dataset
    trainer = build_trainer(**params)

    if is_master():
        save_opts(params, trainer.save_folder)

    if not params['only_eval']:
        trainer.run()
    else:
        trainer.logger.info("Testing...")
        round_loss, round_top1, round_top5 = trainer.model_evaluation()
        trainer.logger.info(f"top1:{round_top1:.2f}, top5:{round_top5:.2f}, loss:{round_loss:.2f}")


def create_model(model_name, data_name, num_classes):
    if data_name.lower() == 'cifar':
        if model_name.lower() == 'resnet':
            if num_classes == 10:
                global_model = resnet_cifar.resnet20_cifar10(10)
            elif num_classes == 100:
                global_model = resnet_cifar.resnet20_cifar100(100)
        elif model_name.lower() == 'mobilenetv2':
            global_model = MobileNetV2Cifar(num_classes=num_classes)
        else:
            raise NotImplementedError(f'No considering {model_name}')
    else:
        raise NotImplementedError(f'No considering {data_name}')
    return global_model


def resume_model(model, resumed_path, resumed_reset=False):
    loaded_params = torch.load(resumed_path, map_location='cpu')
    state_dict = loaded_params['state_dict']
    model.load_state_dict(state_dict)
    if not resumed_reset:
        start_round = loaded_params['round'] + 1
        lr = loaded_params['lr']
        return start_round, lr


def load_data(params):
    dataset = dict()
    dataset['train'], dataset['test'] = build_dataset(**params)
    return dataset


if __name__ == '__main__':
    main()

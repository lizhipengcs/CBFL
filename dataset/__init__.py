from dataset.federated_dataset import FederatedDataset
from dataset.federated_dataset_dirichlet import FederatedDatasetDirichlet, FederatedInaturalist
import torch.nn.functional as F
import os
from torchvision import transforms
from torchvision import datasets

maps = dict(dirichlet=FederatedDatasetDirichlet,
            inaturalist_partition=FederatedInaturalist)


def get_dataset(data_name, data_path, num_classes):
    """
    Get dataset according to data name and data path.
    """
    transform_train, transform_test = data_transform(data_name)
    if data_name.lower() == 'mnist':
        train_dataset = datasets.MNIST(data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.MNIST(data_path, train=False, download=False, transform=transform_test)
    elif data_name.lower() == 'cifar':
        if num_classes == 10:
            train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform_train)
            # val_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform_test)
            test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform_test)
        elif num_classes == 100:
            train_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform_train)
            # val_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform_test)
            test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transform_test)
    elif data_name.lower() == 'imagenet':
        traindir = os.path.join(data_path, "train")
        testdir = os.path.join(data_path, "val")
        train_dataset = datasets.ImageFolder(traindir, transform_train)
        # val_dataset = datasets.ImageFolder(traindir, transform_test)
        test_dataset = datasets.ImageFolder(testdir, transform_test)
    elif data_name.lower() == "cinic":
        traindir = os.path.join(data_path, "train")
        testdir = os.path.join(data_path, "test")
        train_dataset = datasets.ImageFolder(traindir, transform=transform_train)
        # val_dataset = datasets.ImageFolder(traindir, transform_test)
        test_dataset = datasets.ImageFolder(testdir, transform_test)
    elif data_name.lower() == "inaturalist":
        train_dataset = None
        test_dataset = None
    else:
        raise NotImplementedError(f'No considering {data_name}')
    return train_dataset, test_dataset
    # return train_dataset, test_dataset, val_datase


def build_dataset(**kwargs):
    train_dataset, test_dataset = get_dataset(kwargs['data_name'], kwargs['data_path'], kwargs['num_classes'])
    if kwargs['data_name'] == 'inaturalist':
        train_fd = maps[kwargs['partition_type']](dataset=train_dataset, csvFile=kwargs['train_csv'], **kwargs)
        kwargs['server_ratio'] = 0  # Do not spilt test dataset.
        test_fd = maps[kwargs['partition_type']](dataset=test_dataset, csvFile=kwargs['test_csv'], **kwargs)
    else:
        train_fd = maps[kwargs['partition_type']](dataset=train_dataset, **kwargs)
        kwargs['server_ratio'] = 0  # Do not spilt test dataset.
        test_fd = maps[kwargs['partition_type']](dataset=test_dataset, **kwargs)

    return train_fd, test_fd




def data_transform(data_name):
    transform_train, transform_test = None, None
    if data_name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return transform, transform
    elif data_name.lower() == 'cifar':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif data_name.lower() == 'imagenet':
        # follow GDFQ
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            # transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
        ])

        # # Centralized training
        # transform_train = transforms.Compose([
        #     transforms.RandomResizedCrop(224),
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(
        #         brightness=0.4,
        #         contrast=0.4,
        #         saturation=0.4,
        #         hue=0.2),
        #     transforms.ToTensor(),
        #     normalize,
        # ])
        # transform_test = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     normalize,
        # ])
    elif data_name.lower() == "cinic":
        cinic_mean = [0.47889522, 0.47227842, 0.43047404]
        cinic_std = [0.24205776, 0.23828046, 0.25874835]
        transform_train = transforms.Compose([transforms.ToTensor(),
                                              transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                                (4, 4, 4, 4),mode='reflect').data.squeeze()),
                                              transforms.ToPILImage(),
                                              transforms.RandomCrop(32),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=cinic_mean, std=cinic_std)
                                              ])

        transform_test=transforms.Compose([transforms.ToTensor(),
                                           transforms.Lambda(lambda x: F.pad(x.unsqueeze(0),
                                                             (4, 4, 4, 4), mode='reflect').data.squeeze()),
                                           transforms.ToPILImage(),
                                           transforms.RandomCrop(32),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=cinic_mean, std=cinic_std)
                                           ])

    return transform_train, transform_test

from dataset.federated_dataset import FederatedDataset
from dataset.federated_dataset_dirichlet import FederatedDatasetDirichlet
from torchvision import transforms
from torchvision import datasets

maps = dict(dirichlet=FederatedDatasetDirichlet)


def get_dataset(data_name, data_path, num_classes):
    """
    Get dataset according to data name and data path.
    """
    transform_train, transform_test = data_transform(data_name)
    if data_name.lower() == 'cifar':
        if num_classes == 10:
            train_dataset = datasets.CIFAR10(data_path, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR10(data_path, train=False, download=True, transform=transform_test)
        elif num_classes == 100:
            train_dataset = datasets.CIFAR100(data_path, train=True, download=True, transform=transform_train)
            test_dataset = datasets.CIFAR100(data_path, train=False, download=True, transform=transform_test)
    else:
        raise NotImplementedError(f'No considering {data_name}')
    return train_dataset, test_dataset


def build_dataset(**kwargs):
    train_dataset, test_dataset = get_dataset(kwargs['data_name'], kwargs['data_path'], kwargs['num_classes'])
    train_fd = maps[kwargs['partition_type']](dataset=train_dataset, **kwargs)
    kwargs['server_ratio'] = 0  # Do not spilt test dataset.
    test_fd = maps[kwargs['partition_type']](dataset=test_dataset, **kwargs)

    return train_fd, test_fd


def data_transform(data_name):
    transform_train, transform_test = None, None
    if data_name.lower() == 'cifar':
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

    else:
        raise NotImplementedError(f'No considering {data_name}')
    return transform_train, transform_test

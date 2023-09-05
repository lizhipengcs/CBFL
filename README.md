# Code for CBFL(类别均衡联邦学习)

## 环境准备

```
所需的安装包在requirements.txt
```

## 配置文件

​		配置文件是configs文件夹下的yaml文件，在运行代码前需修改各数据集的路径参数：`data_path`

## 运行

- CBFL on CIFAR-10

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=22333 main.py --params configs/cifar/cbfl_c10.yaml -o exp_output/cbfl_cifar10
```

- CBFL on CIFAR-100

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=23334 main.py --params configs/cifar/cbfl_c100.yaml -o exp_output/cbfl_cifar100
```

- FedAvg on CIFAR-10

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=20664 main.py --params configs/cifar/fedavg_c10.yaml -o exp_output/fedavg_cifar10
```

- FedAvg on CIFAR-100


```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=15693 main.py --params configs/cifar/fedavg_c100.yaml -o exp_output/fedavg_cifar100
```

- FedProx on CIFAR-10


```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=30930 main.py --params configs/cifar/fedprox_c10.yaml -o exp_output/fedprox_cifar10
```

- FedProx on CIFAR-100


```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=39650 main.py --params configs/cifar/fedprox_c100.yaml -o exp_output/fedprox_cifar100
```

- SCAFFOLD on CIFAR-10


```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=10987 main.py --params configs/cifar/scaffold_c10.yaml -o exp_output/scaffold_cifar10
```

- SCAFFOLD on CIFAR-100


```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=29753 main.py --params configs/cifar/scaffold_c100.yaml -o exp_output/scaffold_cifar100
```

- FedNova on CIFAR-10


```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=13018 main.py --params configs/cifar/fednova_c10.yaml -o exp_output/fednova_cifar10
```

- FedNova on CIFAR-100


```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=33750 main.py --params configs/cifar/fednova_c100.yaml -o exp_output/fednova_cifar100
```


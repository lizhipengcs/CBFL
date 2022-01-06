# Code for Reproducing CBFL(类别均衡联邦学习)

目前，本仓库开源了baseline对比方法（FedAvg、FedProx、SCAFFOLD、FedNova）的复现代码，所提出的CBFL算法代码将在论文被接收后开源。

环境准备:

pip install -r requirment.txt

FedAvg on CIFAR-10:

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=22333 main.py --params configs/cifar/fedavg_c10.yaml -o exp_output/fedavg_cifar10

FedAvg on CIFAR-100:

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=22333 main.py --params configs/cifar/fedavg_c100.yaml -o exp_output/fedavg_cifar100

FedProx on CIFAR-10:

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=22333 main.py --params configs/cifar/fedprox_c10.yaml -o exp_output/fedprox_cifar10

FedProx on CIFAR-100:

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=22333 main.py --params configs/cifar/fedprox_c100.yaml -o exp_output/fedprox_cifar100

SCAFFOLD on CIFAR-10:

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=22333 main.py --params configs/cifar/scaffold_c10.yaml -o exp_output/scaffold_cifar10

SCAFFOLD on CIFAR-100:

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=22333 main.py --params configs/cifar/scaffold_c100.yaml -o exp_output/scaffold_cifar100

FedNova on CIFAR-10:

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=22333 main.py --params configs/cifar/fednova_c10.yaml -o exp_output/fednova_cifar10

FedNova on CIFAR-100:

CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=22333 main.py --params configs/cifar/fednova_c100.yaml -o exp_output/fednova_cifar100
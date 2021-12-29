CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=22333 main.py --params configs/cifar/fedavg_c10.yaml -o exp_output/fedavg_cifar10

python -m torch.distributed.launch --nproc_per_node=4 --master_port=22333 main.py --params configs/cifar/fedavg_c10.yaml -o output/dist_test01

kernprof -l launch.py --nproc_per_node=8 --master_port=22333 main.py --params configs/cifar/fedavg_c10.yaml -o t/31


python -m torch.distributed.launch --nproc_per_node=2 --master_port=22334 a.py


python -m torch.distributed.launch --nproc_per_node=8 --master_port=22333 main.py --params configs/cifar/fedavg_c10_debug.yaml -o output/dist_test10
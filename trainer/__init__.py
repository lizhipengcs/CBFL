# from trainer.federated_learning import FederatedLearning
# from trainer.federated_averaging import FederatedAveraging
# from trainer.federated_generative_averaging import FederatedGenerativeAveraging
from trainer.federated_averaging_imagenet import FederatedAveragingImageNet
from trainer.federated_generative_averaging_imagenet import FederatedGenerativeAveragingImageNet
from trainer.federated_prox_imagenet import FederatedProxImageNet
from trainer.federated_averaging_mutual_imagenet import FederatedAveragingMutualImageNet
from trainer.federated_dcgan_imagenet import FederatedDcganImageNet
from trainer.federated_multi_generative_averaging_imagenet import FederatedMultiGenerativeAveragingImageNet
from trainer.federated_generative_balanced_averaging_imagenet import FederatedGenerativeBalancedAveragingImageNet
from trainer.federated_nova_imagenet import FederatedNovaImageNet
from trainer.scaffold_imagenet import SCAFFOLDImageNet

def build_trainer(**kwargs):
    maps = dict(
        # fedavg=FederatedAveraging,
        # fed_gen_avg=FederatedGenerativeAveraging,
        fedavg_imagenet=FederatedAveragingImageNet,
        fed_gen_avg_imagenet=FederatedGenerativeAveragingImageNet,
        fedprox_imagenet=FederatedProxImageNet,
        fedavg_mutual=FederatedAveragingMutualImageNet,
        fed_dcgan_imagenet=FederatedDcganImageNet,
        fed_multi_gen_avg_imagenet=FederatedMultiGenerativeAveragingImageNet,
        fed_gen_bal_avg_imagenet=FederatedGenerativeBalancedAveragingImageNet,
        fednova_imagenet=FederatedNovaImageNet,
        scaffold_imagenet=SCAFFOLDImageNet
        # fedprox=FederatedProx
    )
    return maps[kwargs['aggregation_type']](kwargs)

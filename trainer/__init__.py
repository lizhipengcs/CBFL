from trainer.federated_averaging import FederatedAveraging
from trainer.federated_prox import FederatedProx
from trainer.federated_nova import FederatedNova
from trainer.scaffold import SCAFFOLD

def build_trainer(**kwargs):
    maps = dict(
        fedavg=FederatedAveraging,
        fedprox=FederatedProx,
        fednova=FederatedNova,
        scaffold=SCAFFOLD
    )
    return maps[kwargs['aggregation_type']](kwargs)

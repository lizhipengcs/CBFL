from trainer.federated_averaging import FederatedAveraging
from trainer.federated_prox import FederatedProx
from trainer.federated_nova import FederatedNova
from trainer.scaffold import SCAFFOLD
from trainer.cbfl import CBFL


def build_trainer(**kwargs):
    maps = dict(
        fedavg=FederatedAveraging,
        fedprox=FederatedProx,
        fednova=FederatedNova,
        scaffold=SCAFFOLD,
        cbfl=CBFL,
    )
    return maps[kwargs['aggregation_type']](kwargs)

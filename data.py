import numpy as np
import torch_geometric as geom
from torch.utils.data import Subset

MOLECULES = ["benzene", "malonaldehyde", "ethanol", "toluene"]


def get_dataset(molecule):
    assert molecule in MOLECULES, "Molecule must have CCSD(T) level of theory"
    return geom.datasets.MD17(root="data/MD17", name=molecule)


def download_all():
    for molecule in MOLECULES:
        get_dataset(molecule)


def split_dataset(dataset, n_train, n_val, seed):
    np.random.seed(seed)
    idxs = list(range(len(dataset)))
    np.random.shuffle(idxs)
    train_idx = idxs[:n_train]
    val_idx = idxs[n_train : n_train + n_val]

    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)

    return train_dataset, val_dataset


if __name__ == "__main__":
    dataset = get_dataset("benzene")

    train_dataset, val_dataset = split_dataset(dataset, 900, 60, 1337)

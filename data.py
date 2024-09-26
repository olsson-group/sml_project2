import numpy as np
import torch_geometric as geom
from torch.utils.data import Subset

MOLECULES = ["benzene", "malonaldehyde", "ethanol", "toluene"]


def get_dataset(molecule, path="data/MD17"):
    assert molecule in MOLECULES, "Molecule must have CCSD(T) level of theory"
    return geom.datasets.MD17(root=path, name=molecule)


def download_all(path):
    for molecule in MOLECULES:
        get_dataset(molecule, path)


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

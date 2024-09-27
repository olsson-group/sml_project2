import numpy as np
import torch_geometric as geom
from torch.utils.data import Subset

MOLECULES = ["benzene", "malonaldehyde", "ethanol", "toluene"]

ATOMIC_ENERGIES = {
    1: -0.501392,
    6: -37.8450,
    7: -54.5834,
    8: -75.0645,
}


def get_dataset(molecule, path="data/MD17"):
    assert molecule in MOLECULES, "Molecule must have CCSD(T) level of theory"
    return geom.datasets.MD17(root=path, name=molecule)


def download_all(path):
    for molecule in MOLECULES:
        get_dataset(molecule, path)


def get_ground_state_energy(dataset):
    energies = [data.energy.item() for data in dataset]
    ground_state_energy = min(energies)
    return ground_state_energy


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
    for molecule in MOLECULES:
        print(molecule)
        dataset = get_dataset(molecule)
        atomic_energy = get_atomic_energy(dataset)

        print(dataset[0].energy.item())
        print(atomic_energy)

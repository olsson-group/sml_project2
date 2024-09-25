import torch_geometric as geom


def download_data(molecule):
    geom.datasets.MD17(root="data/MD17", name=molecule)


def download_all():
    for molecule in ["benzene", "malonaldehyde", "ethanol", "toluene"]:
        download_data(molecule)

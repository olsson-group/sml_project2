import torch_geometric as geom


def get_data(molecule):
    return geom.datasets.MD17(root="data/MD17", name=molecule)


def download_all():
    for molecule in ["benzene", "malonaldehyde", "ethanol", "toluene"]:
        get_data(molecule)

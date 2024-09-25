import torch_geometric as geom


def download_data():
    for molecule in ["benzene", "malonaldehyde", "ethanol", "toluene"]:
        geom.datasets.MD17(root="data/MD17", name=molecule)


download_data()

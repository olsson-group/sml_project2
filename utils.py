import mdtraj as md
import torch

ELEMENTS = {
    1: md.element.hydrogen,
    2: md.element.helium,
    3: md.element.lithium,
    4: md.element.beryllium,
    5: md.element.boron,
    6: md.element.carbon,
    7: md.element.nitrogen,
    8: md.element.oxygen,
    9: md.element.fluorine,
    10: md.element.neon,
    11: md.element.sodium,
    12: md.element.magnesium,
    13: md.element.aluminum,
    14: md.element.silicon,
    15: md.element.phosphorus,
    16: md.element.sulfur,
}


def get_mdtraj(traj, atoms):
    atoms = atoms.cpu().numpy()
    traj = traj.cpu().numpy()

    topology = get_topology(atoms)
    traj = traj.reshape(-1, *traj.shape[-2:])
    traj = md.Trajectory(traj, topology)
    return traj


def get_topology(atom_numbers):
    topology = md.Topology()
    chain = topology.add_chain()
    residue = topology.add_residue("RES", chain)

    for i, atom_number in enumerate(atom_numbers):
        e = ELEMENTS[int(atom_number)]
        name = f"{e}{i}"
        topology.add_atom(name, e, residue)

    return topology


def get_msd(y, y_hat):
    return ((y - y_hat) ** 2).sum(-1)


def get_rmsd(y, y_hat):
    return torch.sqrt(get_msd(y, y_hat))


if __name__ == "__main__":
    x = torch.randn(9, 3)
    y = torch.zeros(9, 3)
    assert torch.allclose(get_msd(x, y), torch.norm(x, dim=-1) ** 2)

import MDAnalysis as mda
import pickle


def defect_loader(base_names=['no_eq'], base_dir_prefix='output_data/', filenames=None):
    if filenames is None:
        filenames = [
            ('PLacyl.gro', 'PLacyl.xtc'),
            ('TGacyl.gro', 'TGacyl.xtc'),
            ('TGglyc.gro', 'TGglyc.xtc')
        ]

    universes = {}
    for base_name in base_names:
        base_dir = f'{base_dir_prefix}{base_name}'
        for topo, traj in filenames:
            topo_path = f'{base_dir}/{topo}'
            traj_path = f'{base_dir}/{traj}'
            key = (base_name, topo, traj)
            universes[key] = mda.Universe(topo_path, traj_path)

    return universes


def save_defects_to_pickle(defects_dict, filename):
    with open(filename, 'wb') as f:
        pickle.dump(defects_dict, f)

def load_defects_from_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

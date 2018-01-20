import json
import os

import pandas as pd
import numpy as np
import pymatgen as mg
from joblib import dump, Parallel, delayed


def Coumloumb_Matrix_descriptor(structure, N_CM=80):
    natom = len(structure.sites)
    list_Z = [speicies.common_oxidation_states[0] for speicies in structure.species]

    M = np.diag([0.5 * np.abs(Z) ** 2.4 for Z in list_Z])
    for i in np.arange(natom):
        for j in np.arange(i):
            M[i, j] = list_Z[i] * list_Z[j] / structure.get_distance(i, j)
            M[j, i] = M[i, j]

    w = np.linalg.eigvalsh(M)
    w_sorted = sorted(w, key=lambda x: -np.abs(x))
    ret = np.zeros(N_CM)
    ret[:natom] = w

    return ret


def load_structure(row, file_dir):
    # import pdb; pdb.set_trace()
    filename = os.path.join(file_dir, str(int(row['id'])), 'geometry.json')
    with open(filename, 'r') as f:
        d = json.load(f)
        structure = mg.core.Structure.from_dict(d)

    return structure


def load_train_structure(row):
    return load_structure(row, '../data/train')


def load_test_structure(row):
    return load_structure(row, '../data/test')


if __name__ == '__main__':
    df_train = pd.read_csv('../data/train.csv')
    df_test = pd.read_csv('../data/test.csv')

    structures_train = df_train.apply(load_train_structure, axis=1)
    structures_test = df_test.apply(load_test_structure, axis=1)

    CM_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(Coumloumb_Matrix_descriptor)(structure) for structure in structures_train)
    CM_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(Coumloumb_Matrix_descriptor)(structure) for structure in structures_test)

    CM_train = np.array(CM_train_tmp)
    CM_test = np.array(CM_test_tmp)

    dump(CM_train, '../data/CM_mat.train')
    dump(CM_test, '../data/CM_mat.test')

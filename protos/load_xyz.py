import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump

from load_data import load_train_data, load_test_data


# https://www.kaggle.com/tonyyy/how-to-get-atomic-coordinates
def get_xyz_data(filename):
    pos_data = []
    lat_data = []
    with open(filename) as f:
        for line in f.readlines():
            x = line.split()
            if x[0] == 'atom':
                pos_data.append([np.array(x[1:4], dtype=np.float), x[4]])
            elif x[0] == 'lattice_vector':
                lat_data.append(np.array(x[1:4], dtype=np.float))
    return pos_data, np.array(lat_data)


def get_shortest_distances(reduced_coords, amat):
    natom = len(reduced_coords)
    dists = np.zeros((natom, natom))
    Rij_min = np.zeros((natom, natom, 3))

    for i in range(natom):
        for j in range(i):
            rij = reduced_coords[i][0] - reduced_coords[j][0]
            d_min = np.inf
            R_min = np.zeros(3)
            for l in range(-1, 2):
                for m in range(-1, 2):
                    for n in range(-1, 2):
                        r = rij + np.array([l, m, n])
                        R = np.matmul(amat, r)
                        d = np.linalg.norm(R)
                        if d < d_min:
                            d_min = d
                            R_min = R
            dists[i, j] = d_min
            dists[j, i] = dists[i, j]
            Rij_min[i, j] = R_min
            Rij_min[j, i] = -Rij_min[i, j]
    return dists, Rij_min


def get_min_length(distances, A_atoms, B_atoms):
    A_B_length = np.inf
    for i in A_atoms:
        for j in B_atoms:
            d = distances[i, j]
            if d > 1e-8 and d < A_B_length:
                A_B_length = d

    return A_B_length


def get_inv_min_length(index, kind):
    filename = '../data/{}/{}/geometry.xyz'.format(kind, index)
    crystal_xyz, crystal_lat = get_xyz_data(filename)

    A = np.transpose(crystal_lat)
    B = np.linalg.inv(A)

    crystal_red = [[np.dot(B, R), symbol] for (R, symbol) in crystal_xyz]
    crystal_dist, crystal_Rij = get_shortest_distances(crystal_red, A)

    natom = len(crystal_red)
    al_atoms = [i for i in range(natom) if crystal_red[i][1] == 'Al']
    ga_atoms = [i for i in range(natom) if crystal_red[i][1] == 'Ga']
    in_atoms = [i for i in range(natom) if crystal_red[i][1] == 'In']
    o_atoms = [i for i in range(natom) if crystal_red[i][1] == 'O']

    inv_dmin_Al_O, inv_dmin_Ga_O, inv_dmin_In_O = 0, 0, 0

    if len(al_atoms):
        inv_dmin_Al_O = 1.0 / get_min_length(crystal_dist, al_atoms, o_atoms)
    if len(ga_atoms):
        inv_dmin_Ga_O = 1.0 / get_min_length(crystal_dist, ga_atoms, o_atoms)
    if len(in_atoms):
        inv_dmin_In_O = 1.0 / get_min_length(crystal_dist, in_atoms, o_atoms)

    return inv_dmin_Al_O, inv_dmin_Ga_O, inv_dmin_In_O


if __name__ == '__main__':
    df_train = load_train_data()
    df_test = load_test_data()

    inv_train_tmp = Parallel(n_jobs=-1)(delayed(get_inv_min_length)(index, 'train') for index in df_train['id'])
    inv_test_tmp = Parallel(n_jobs=-1)(delayed(get_inv_min_length)(index, 'test') for index in df_test['id'])

    inv_train = pd.DataFrame(np.array(inv_train_tmp), index=df_train.index)
    inv_train['id'] = df_train['id']
    inv_test = pd.DataFrame(np.array(inv_test_tmp), index=df_test.index)
    inv_test['id'] = df_test['id']

    dump(inv_train, 'inv_dmin.train')
    dump(inv_test, 'inv_dmin.test')

import json
import os

import pandas as pd
import numpy as np
import pymatgen as mg
from joblib import dump, Parallel, delayed


def get_metal_oxygen_PRDF(structure, cutoff=10., dr=0.1):
    list_metal = ['Al', 'Ga', 'In']
    structure_element2 = structure.copy()
    structure_element2.remove_species(list_metal)
    neighbors = [structure_element2.get_sites_in_sphere(site.coords, cutoff)
                 for site, species in zip(structure.sites, structure.species)
                 if species.name != 'O']
    bins = int(cutoff / dr)
    hist, _ = np.histogram([dist for e in neighbors for _, dist in e], bins=bins, range=(0, cutoff))
    prdf = hist / (4 * np.pi * dr * (np.linspace(dr, cutoff, num=bins) ** 2) * len(neighbors))
    return prdf


def calc_PRDF_descriptor(structure, cutoff, dr):
    prdf = get_metal_oxygen_PRDF(structure, cutoff, dr)
    oxygen_ave = len([1 for species in structure.species if species.name == 'O']) / structure.volume
    dsc = prdf - oxygen_ave
    return dsc


def calc_PRDF_power_spectrum_descriptor(structure, cutoff, dr, num):
    prdf = get_metal_oxygen_PRDF(structure, cutoff, dr)
    oxygen_ave = len([1 for species in structure.species if species.name == 'O']) / structure.volume

    spectrum = np.fft.fft(prdf - oxygen_ave)
    ps = np.abs(spectrum) ** 2
    freq = np.fft.fftfreq(prdf.shape[0], d=dr)

    idx = np.argsort(-ps)
    freq_sorted = freq[idx]

    dsc = freq_sorted[freq_sorted > 0][:num]

    return dsc


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

    cutoff = 20.0
    dr = 0.01
    num = 10

    PRDF_train_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_PRDF_power_spectrum_descriptor)(structure, cutoff, dr, num) for structure in structures_train)
    PRDF_test_tmp = Parallel(n_jobs=-1, verbose=1)(delayed(calc_PRDF_power_spectrum_descriptor)(structure, cutoff, dr, num) for structure in structures_test)

    PRDF_train = np.array(PRDF_train_tmp)
    PRDF_test = np.array(PRDF_test_tmp)

    dump(PRDF_train, '../data/PRDF_ps_mat.train')
    dump(PRDF_test, '../data/PRDF_ps_mat.test')

import json
import os

import pandas as pd
import numpy as np
import pymatgen as mg
from pymatgen.analysis.local_env import JMolNN
from joblib import Parallel, delayed


def get_anion_polyhedron(structure, cation_sym):
    nn = JMolNN()
    coo = [[e['site_index'] for e in nn.get_nn_info(structure, i) if e['site'].specie.name == 'O'] for i, spc in enumerate(structure.species) if spc.name == cation_sym]
    return coo


def get_cation_polyhedron(structure, anion_sym='O'):
    nn = JMolNN()
    coo = [[e['site_index'] for e in nn.get_nn_info(structure, i) if e['site'].specie.name != 'O'] for i, spc in enumerate(structure.species) if spc.name == anion_sym]
    return coo


def load_structure(row, file_dir):
    filename = os.path.join(file_dir, str(int(row['id'])), 'geometry.json')
    with open(filename, 'r') as f:
        d = json.load(f)
        structure = mg.core.Structure.from_dict(d)

    return structure


def load_train_structure(row):
    return load_structure(row, '../data/train')


def load_test_structure(row):
    return load_structure(row, '../data/test')

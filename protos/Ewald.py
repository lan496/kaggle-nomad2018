import json

import pandas as pd
import numpy as np
import pymatgen as mg
from scipy.special import erfc


class EwaldSummationMatrix(object):

    def __init__(self, structure, real_space_cut=None, recip_space_cut=None,
                 eta=None, acc_factor=12.):
        self._s = structure
        self._vol = structure.volume

        if eta:
            self._eta = eta
            self._sqrt_eta = np.sqrt(eta)
        else:
            self._sqrt_eta = np.sqrt(np.pi) * ((0.01 * len(structure) / self._vol) ** (1 / 6))
            self._eta = self._sqrt_eta ** 2

        self._acc_factor = float(acc_factor)
        self._accf = np.sqrt(np.log(10 ** self._acc_factor))

        # automatically determine the optimal real and reciprocal space cutoff
        if real_space_cut:
            self._rmax = real_space_cut
        else:
            self._rmax = self._accf / self._sqrt_eta

        if recip_space_cut:
            self._gmax = recip_space_cut
        else:
            self._gmax = 2. * self._sqrt_eta * self._accf

        # precomute
        self._oxi_states = [specie.common_oxidation_states[0] for specie in structure.species]
        self._coords = np.array(self._s.cart_coords)

    def get_ewald_summation_matrix(self):
        ereal = self._calc_real()
        erecip = self._calc_recip()
        econstant = self._calc_constant()

        etotal = ereal + erecip + econstant

        return etotal

    def _calc_real(self):
        numsites = self._s.num_sites
        fcoords = self._s.frac_coords
        coords = self._coords  # (numsites, 3)

        oxistates = np.array(self._oxi_states)
        zizj = oxistates[np.newaxis, :] * oxistates[:, np.newaxis]  # (numsites, numsites)

        real_nn = self._s.lattice.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], self._rmax)
        frac_coords = [fcoords for fcoords, dist, i in real_nn if dist != 0]
        ls = self._s.lattice.get_cartesian_coords(frac_coords)  # (numpoints, 3)

        lrij = coords[np.newaxis, :, np.newaxis, :] - coords[np.newaxis, np.newaxis, :, :] + ls[:, np.newaxis, np.newaxis, :]  # (numpoints, numsites, numsites, 3)
        lrij_norm = np.sum(lrij, axis=3)  # (numpoints, numsites, numsites)

        erfcvals = erfc(self._sqrt_eta * lrij_norm)  # (numpoints, numsites, numsites)
        inds = lrij_norm > 1e-8
        inv_lrij_norm = np.zeros_like(lrij_norm)  # (numpoints, numsites, numsites)
        inv_lrij_norm[~inds] = 0

        ereal = zizj * np.sum(erfcvals * inv_lrij_norm, axis=0)  # (numsites, numsites)
        ereal[np.diag_indices(numsites)] *= 0.5

        return ereal

    def _calc_recip(self):
        numsites = self._s.num_sites
        coords = self._coords  # (numsites, 3)
        rcp_latt = self._s.lattice.reciprocal_lattice
        recip_nn = rcp_latt.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], self._gmax)

        frac_coords = [fcoords for fcoords, dist, i in recip_nn if dist != 0]
        gs = rcp_latt.get_cartesian_coords(frac_coords)  # (numkpoints, 3)
        g2s = np.sum(gs ** 2, axis=1)  # (numkpoints, )
        expvals = np.exp(-g2s / (4 * self._sqrt_eta))  # (numkpoints, )
        grs = np.sum(gs[:, np.newaxis] * coords[np.newaxis, :], axis=2)  # (numkpoints, numsites)
        cosgr = np.cos(grs[:, :, np.newaxis] - grs[:, np.newaxis, :])  # (numkpoints, numsites, numsites)

        oxistates = np.array(self._oxi_states)

        zizj = oxistates[np.newaxis, :] * oxistates[:, np.newaxis]  # (numsites, numsites)

        erecip = zizj / (np.pi * self._vol) * np.sum(expvals[:, np.newaxis, np.newaxis] / g2s[:, np.newaxis, np.newaxis] * cosgr, axis=0)
        erecip[np.diag_indices(numsites)] *= 0.5

        return erecip

    def _calc_constant(self):
        numsites = self._s.num_sites
        oxistates = np.array(self._oxi_states)
        zizi = oxistates ** 2

        econstant = - self._sqrt_eta / np.sqrt(np.pi) * (zizi[:, np.newaxis] + zizi[np.newaxis, :]) - np.pi / (2. * self._vol * self._eta) * ((oxistates[:, np.newaxis] + oxistates[np.newaxis, :]) ** 2)
        econstant[np.diag_indices(numsites)] *= 0.5

        return econstant


def ewald_descriptor(structure, acc_factor=12.):
    ewald = EwaldSummationMatrix(structure, acc_factor=acc_factor)
    ewald_matrix = ewald.get_ewald_summation_matrix()

    eigvals = np.linalg.eigvalsh(ewald_matrix)

    eigvals_sorted = eigvals[np.argsort(-np.abs(eigvals))]

    return eigvals_sorted


if __name__ == '__main__':
    with open('../data/train/116/geometry.json', 'r') as f:
        d = json.load(f)
        structure = mg.core.Structure.from_dict(d)

    acc_factor = 30.

    dsc = ewald_descriptor(structure, acc_factor)
    print(dsc)

    structure2 = structure.copy()
    structure2.make_supercell(2)
    dsc2 = ewald_descriptor(structure2, acc_factor)
    print(dsc2)

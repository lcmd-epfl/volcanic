#!/usr/bin/env python

import numpy as np
import scipy.constants as sc


def calc_tof(array, Delta_G_reaction, T, coeff, exact=True, verb=0):
    coeff = np.array(coeff)
    array = np.array(array)
    assert array.size == coeff.size
    # coeff = list(coeff)
    h = sc.value("Planck constant")
    k_b = sc.value("Boltzmann constant")
    R = sc.value("molar gas constant")
    n_TS = np.count_nonzero(coeff)
    n_I = np.count_nonzero(coeff == 0)
    if verb > 2:
        print(f"Number of intermediates taken into account is {n_I}")
        print(f"Number of TS taken into account is {n_TS}")
    X_TOF = np.zeros((n_I - 1, 2))
    if exact:
        matrix_T_I = np.zeros((n_I - 1, 2))
        j = 0
        for i in range(0, len(array) - 1):
            if coeff[i] == 1:
                matrix_T_I[(j, 1)] = array[i]
                j += 1
            else:
                if coeff[i + 1] == 0:
                    matrix_T_I[(j, 0)] = array[i]
                    matrix_T_I[(j, 1)] = array[i]
                    j += 1
                else:
                    matrix_T_I[(j, 0)] = array[i]
        sum_span = 0
        for i in range(0, n_I - 1):
            for j in range(0, n_I - 1):
                if i > j:
                    sum_span += np.exp(
                        (
                            (matrix_T_I[(i, 1)] - matrix_T_I[(j, 0)] - Delta_G_reaction)
                            * 4184
                        )
                        / (R * T)
                    )
                else:
                    sum_span += np.exp(
                        ((matrix_T_I[(i, 1)] - matrix_T_I[(j, 0)]) * 4184) / (R * T)
                    )
        TOF = ((k_b * T) / h) * (
            (np.exp((-Delta_G_reaction * 4184) / (R * T))) / sum_span
        )
        for i in range(0, n_I - 1):
            sum_e = 0
            for j in range(0, n_I - 1):
                if i > j:
                    sum_e += np.exp(
                        (
                            (matrix_T_I[(i, 1)] - matrix_T_I[(j, 0)] - Delta_G_reaction)
                            * 4184
                        )
                        / (R * T)
                    )
                else:
                    sum_e += np.exp(
                        ((matrix_T_I[(i, 1)] - matrix_T_I[(j, 0)]) * 4184) / (R * T)
                    )
            X_TOF[(i, 1)] = round(sum_e / sum_span, 2)
        for j in range(0, n_I - 1):
            sum_e = 0
            for i in range(0, n_I - 1):
                if i > j:
                    sum_e += np.exp(
                        (
                            (matrix_T_I[(i, 1)] - matrix_T_I[(j, 0)] - Delta_G_reaction)
                            * 4184
                        )
                        / (R * T)
                    )
                else:
                    sum_e += np.exp(
                        ((matrix_T_I[(i, 1)] - matrix_T_I[(j, 0)]) * 4184) / (R * T)
                    )
            X_TOF[(j, 0)] = round(sum_e / sum_span, 2)
    else:
        delta_E_matrix = np.zeros((len(array), len(array)))
        for i in range(0, len(array) - 1):
            for j in range(0, len(array) - 1):
                if not (coeff[i] == coeff[j]):
                    if i < j:
                        delta_E_matrix[i, j] = array[i] - array[j] + Delta_G_reaction
                    else:
                        delta_E_matrix[i, j] = array[i] - array[j]
        Energy_Span = np.amax(delta_E_matrix)
        TOF = ((k_b * T) / h) * np.exp((-(Energy_Span * 4184) / (R * T)))  # s^-1
    return TOF, X_TOF


if __name__ == "__main__":
    a = np.array([0.0, 8.59756198, 7.1439459, 12.47470641, -27.48101312])
    dgr = -5.8
    T = 298.15
    coeff = [0, 1, 0, 1, 0]
    print(calc_tof(a, dgr, T, coeff, exact=True, verb=3)[0])
    print(calc_tof(a, dgr, T, coeff, exact=False, verb=3)[0])

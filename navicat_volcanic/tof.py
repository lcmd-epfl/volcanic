#!/usr/bin/env python

import copy

import numpy as np


def calc_atof(es, dgr, T, verb=0):
    h = 6.62607015e-34
    k_b = 1.380649e-23
    R = 8.314462618
    TOF = ((k_b * T) / h) * np.exp((-(es * 4184) / (R * T)))
    return TOF


def calc_tof(array, dgr, T, coeff, exact=True, verb=0):
    """Function to compute TOF using the energy span model.
    Reproduces results from the AUTOF code (https://doi.org/10.1002/jcc.21669).
    Adapted from the AUTOF implementation with contribution by Pit Steinbach."""

    coeff = np.array(coeff)
    array = np.array(array)
    h = 6.62607015e-34
    k_b = 1.380649e-23
    R = 8.314462618
    n_S = array.size
    n_TS = np.count_nonzero(coeff)
    n_I = np.count_nonzero(coeff == 0)
    if verb > 1:
        print(f"Number of intermediates taken into account is {n_I}")
        print(f"Number of TS taken into account is {n_TS}")
    try:
        assert array.size == coeff.size
    except AssertionError:
        print(
            f"WARNING: The species number {n_S} does not seem to match the identified intermediates ({n_I}) plus TS ({n_TS})."
        )
    X_TOF = np.zeros((n_I, 2))
    matrix_T_I = np.zeros((n_I, 2))
    j = 0
    for i in range(n_S):
        if coeff[i] == 0:
            matrix_T_I[j, 0] = array[i]
            if i < n_S - 1:
                if coeff[i + 1] == 1:
                    matrix_T_I[j, 1] = array[i + 1]
                if coeff[i + 1] == 0:
                    if array[i + 1] > array[i]:
                        matrix_T_I[j, 1] = array[i + 1]
                    else:
                        matrix_T_I[j, 1] = array[i]
                j += 1
            if i == n_S - 1:
                if dgr > array[i]:
                    matrix_T_I[j, 1] = dgr
                else:
                    matrix_T_I[j, 1] = array[i]
    if verb > 3:
        print(f"From profile {array}, \n the reaction step matrix is: \n{matrix_T_I}")
    if exact:
        sum_span = 0
        for i in range(n_I):
            for j in range(n_I):
                if i >= j:
                    sum_span += np.exp(
                        ((matrix_T_I[i, 1] - matrix_T_I[j, 0] - dgr) * 4184) / (R * T)
                    )
                if i < j:
                    sum_span += np.exp(
                        ((matrix_T_I[i, 1] - matrix_T_I[j, 0]) * 4184) / (R * T)
                    )
        TOF = ((k_b * T) / h) * ((np.exp((-dgr * 4184) / (R * T))) / sum_span)
        for i in range(n_I):
            sum_e = 0
            for j in range(n_I):
                if i >= j:
                    sum_e += np.exp(
                        ((matrix_T_I[i, 1] - matrix_T_I[j, 0] - dgr) * 4184) / (R * T)
                    )
                if i < j:
                    sum_e += np.exp(
                        ((matrix_T_I[i, 1] - matrix_T_I[j, 0]) * 4184) / (R * T)
                    )
            X_TOF[i, 1] = np.round(sum_e / sum_span, 4)
        for j in range(n_I):
            sum_e = 0
            for i in range(n_I):
                if i >= j:
                    sum_e += np.exp(
                        ((matrix_T_I[i, 1] - matrix_T_I[j, 0] - dgr) * 4184) / (R * T)
                    )
                if i < j:
                    sum_e += np.exp(
                        ((matrix_T_I[i, 1] - matrix_T_I[j, 0]) * 4184) / (R * T)
                    )
            X_TOF[j, 0] = np.round(sum_e / sum_span, 4)
    else:
        dE = np.zeros((n_I, n_I))
        for i in range(n_I):
            for j in range(n_I):
                if i >= j:
                    dE[i, j] = matrix_T_I[i, 1] - matrix_T_I[j, 0]
                if i < j:
                    dE[i, j] = matrix_T_I[i, 1] - matrix_T_I[j, 0] + dgr
        Energy_Span = np.amax(dE)
        sum_span = None
        if verb > 2:
            print(f"Energy Span computed : {Energy_Span} kcal/mol.")
        TOF = ((k_b * T) / h) * np.exp((-(Energy_Span * 4184) / (R * T)))
        if verb > 2:
            print(f"TOF computed : {TOF} s^-1.")
        i, j = np.unravel_index(np.argmax(dE, axis=None), dE.shape)
        X_TOF[i, 1] = 1.0
        X_TOF[j, 0] = 1.0
    return TOF, X_TOF, sum_span


def calc_es(profile, dgr, esp=True, chemical_sense=False):
    es1 = -np.inf
    imax = 0
    imin = 1
    for i, lower in enumerate(profile):
        view = copy.deepcopy(profile)
        view[:i] += dgr
        j = np.argmax(view)
        upper = view[j]
        es2 = upper - lower
        if es2 > es1:
            diff = es2 - es1
            es1 = es2
            imax = j
            imin = i
    return [-es1, imax, imin, diff]


def calc_s_es(profile, dgr, esp=True, chemical_sense=False):
    es1 = -np.inf
    imax = 0
    imin = 1
    for i, lower in enumerate(profile):
        if i < len(profile) - 1:
            es2 = profile[i + 1] - lower
        else:
            es2 = dgr - lower
        if es2 > es1:
            diff = es2 - es1
            es1 = es2
            imax = i + 1
            imin = i
    return [-es1, imax, imin, diff]


def test_tof():
    # Test 1
    T = 298.15
    a = np.array([0.0, 8.59756198, 7.1439459, 12.47470641, -27.48101312])
    dgr_a = -5.8
    coeff_a = [0, 1, 0, 1, 0]
    etof, xetof, _ = calc_tof(a, dgr_a, T, coeff_a, exact=True, verb=0)
    assert np.isclose(etof, 5.71e-13, 4)
    atof, xatof, _ = calc_tof(a, dgr_a, T, coeff_a, exact=False, verb=0)

    T = 298.15
    c = np.array([0.0, -1.2, 3.7, 18.9, 3.8, 7.4, -3.8, -1.0, 6.0, 20.1])
    dgr_c = -6.5
    coeff_c = [0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    etof, xetof, _ = calc_tof(c, dgr_c, T, coeff_c, exact=True, verb=0)
    assert np.isclose(etof, 1.84e-05, 4)
    atof, xatof, _ = calc_tof(c, dgr_c, T, coeff_c, exact=False, verb=0)


def test_aryl_ether_cleavage():
    T = 383.15
    coeff = [0, 0, 1, 0, 1, 0, 0, 1]
    dgr = -28.1

    # Ni(PMe3)
    a = np.array([0.0, 0.2, 27.4, 10.5, 23.9, -19.9, -27.5, -0.5])
    # Ni(SIMes3)
    b = np.array([0.0, 6.34, 28.6, 11.42, 22.81, -17.96, -40.31, -0.07])
    # Ni(PBu3)
    c = np.array([0.0, 4.68, 27.95, 13.85, 25.95, -14.93, -24.80, 2.07])

    etof, xetof, _ = calc_tof(a, dgr, T, coeff, exact=True, verb=0)
    assert np.isclose(etof, 6.91e-04, 4)
    etof, xetof, _ = calc_tof(b, dgr, T, coeff, exact=True, verb=0)
    assert np.isclose(etof, 2.89e-11, 4)
    etof, xetof, _ = calc_tof(c, dgr, T, coeff, exact=True, verb=0)
    assert np.isclose(etof, 6.85e-04, 4)


if __name__ == "__main__":
    test_tof()
    test_aryl_ether_cleavage()

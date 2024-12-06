#!/usr/bin/env python

import numpy as np
import sklearn as sk
import sklearn.linear_model

from navicat_volcanic.exceptions import InputError

rng = np.random.default_rng()


def call_imputer(a, b, imputer_strat="iterative"):
    if imputer_strat == "iterative":
        try:
            from sklearn.experimental import enable_iterative_imputer
            from sklearn.impute import IterativeImputer
        except ModuleNotFoundError as err:
            return a
        imputer = IterativeImputer(max_iter=25)
        newa = imputer.fit(b).transform(a.reshape(1, -1)).flatten()
        return newa
    elif imputer_strat == "simple":
        try:
            from sklearn.impute import SimpleImputer
        except ModuleNotFoundError as err:
            return a
        imputer = SimpleImputer()
        newa = imputer.fit_transform(a.reshape(-1, 1)).flatten()
        return newa
    elif imputer_strat == "knn":
        try:
            from sklearn.impute import KNNImputer
        except ModuleNotFoundError as err:
            return a
        imputer = KNNImputer(n_neighbors=2)
        newa = imputer.fit(b).transform(a.reshape(1, -1)).flatten()
        return newa
    elif imputer_strat == "none":
        return a
    else:
        return a


def curate_d(d, regress, cb, ms, tags, imputer_strat="none", nstds=5, verb=0):
    assert isinstance(d, np.ndarray)
    dit = d[:, regress]
    tagsit = tags[:]
    curated_d = np.zeros_like(dit)
    for i in range(dit.shape[1]):
        mean = dit[:, i].mean()
        std = dit[:, i].std()
        moe = nstds * std
        if verb > 2:
            print(f"We assume a margin of error of {moe}.")
        maxtol = np.abs(mean) + moe
        mintol = np.abs(mean) - moe
        absd = np.abs(dit[:, i])
        if any(absd > maxtol):
            outlier = np.where(absd > maxtol)
            if verb > 1:
                print(
                    f"Among data series {tagsit[i]} some outliers (very large values) were detected: {dit[outlier,i].flatten()} and will be skipped."
                )
            dit[outlier, i] = np.nan
        if any(absd < mintol):
            outlier = np.where(absd < mintol)
            if verb > 1:
                print(
                    f"Among data series {tagsit[i]} some outliers (very small values) were detected: {dit[outlier,i].flatten()} and will be skipped."
                )
            dit[outlier, i] = np.nan
    for j in range(dit.shape[0]):
        n_nans = np.count_nonzero(np.isnan(dit[j, :]))
        if n_nans > 0:
            tofix = dit[j, :]
            if verb > 1:
                print(f"Using the imputer strategy, converted\n {tofix}.")
            toref = dit[np.arange(dit.shape[0]) != j, :]
            dit[j, :] = call_imputer(tofix, toref, imputer_strat)
            if verb > 1:
                print(f"to\n {dit[j,:]}.")
        curated_d[j, :] = dit[j, :]
    incomplete = np.ones_like(curated_d[:, 0], dtype=bool)
    for i in range(curated_d.shape[0]):
        n_nans = np.count_nonzero(np.isnan(curated_d[i, :]))
        if n_nans > 0:
            if verb > 1:
                print(
                    f"Some of your reaction profiles contain {n_nans} undefined values and will not be considered:\n {curated_d[i,:]}"
                )
            incomplete[i] = False
    curated_cb = cb[incomplete]
    curated_ms = ms[incomplete]
    d[:, regress] = curated_d
    d = d[incomplete, :]
    return d, curated_cb, curated_ms


def find_1_dv(d, tags, coeff, regress, verb=0):
    assert isinstance(d, np.ndarray)
    assert len(tags) == len(coeff) == len(regress)
    valid_d = np.copy(regress)
    try:
        assert np.isclose(d[:, 0].std(), 0)
    except AssertionError as m:
        raise InputError(
            "The first field of every profile should be the same (reference state). Exit."
        )
    if np.isclose(d[:, -1].std(), 0):
        valid_d[-1] = False
        if verb > 0:
            print(f"\nReaction energy is constant. Assuming substrates are constant.")
    tags = tags[1:]
    coeff = coeff[1:]
    valid_d = valid_d[1:]
    d = d[:, 1:]
    lnsteps = range(d.shape[1])
    regsteps = range(d[:, valid_d].shape[1])
    # Regression diagnostics
    maes = np.ones(d.shape[1])
    r2s = np.ones(d.shape[1])
    maps = np.ones(d.shape[1])
    for i in lnsteps:
        if verb > 0:
            print(f"\nTrying {tags[i]} as descriptor variable:")
        imaes = []
        imaps = []
        ir2s = []
        for j in regsteps:
            Y = d[:, valid_d][:, j]
            XY = np.vstack([d[:, i], d[:, j]]).T
            XY = XY[~np.isnan(XY).any(axis=1)]
            X = XY[:, 0].reshape(-1, 1)
            Y = XY[:, 1]
            reg = sk.linear_model.LinearRegression().fit(X, Y)
            imaes.append(sk.metrics.mean_absolute_error(Y, reg.predict(X)))
            imaps.append(sk.metrics.mean_absolute_percentage_error(Y, reg.predict(X)))
            ir2s.append(reg.score(X, Y))
            if verb > 1:
                print(
                    f"With {tags[i]} as descriptor, regressed {tags[j]} with r2 : {np.round(ir2s[-1],2)} and MAE: {np.round(imaes[-1],2)}"
                )
        if verb > 2:
            print(
                f"\nWith {tags[i]} as descriptor the following r2 values were obtained : {np.round(ir2s,2)}"
            )
        maes[i] = np.around(np.array(imaes).mean(), 4)
        r2s[i] = np.around(np.array(ir2s).mean(), 4)
        maps[i] = np.around(np.array(imaps).mean(), 4)
        if verb > 0:
            print(
                f"\nWith {tags[i]} as descriptor,\n the mean r2 is : {np.round(r2s[i],2)},\n the mean MAE is :  {np.round(maes[i],2)}\n the std MAPE is : {np.round(maps[i],2)}\n"
            )
    criteria = []
    criteria.append(np.squeeze(np.where(r2s == np.max(r2s[~coeff]))))
    criteria.append(np.squeeze(np.where(maes == np.min(maes[~coeff]))))
    criteria.append(np.squeeze(np.where(maps == np.min(maps[~coeff]))))
    for i, criterion in enumerate(criteria):
        if isinstance(criterion, (np.ndarray)):
            if any(criterion.shape):
                criterion = [idx for idx in criterion if ~coeff[idx]]
                criteria[i] = rng.choice(criterion, size=1)
    a = criteria[0]
    b = criteria[1]
    c = criteria[2]
    dvs = []
    if a == b:
        if a == c:
            if verb >= 0:
                print(f"All indicators agree: best descriptor is {tags[a]}")
            dvs.append(a)
        else:
            if verb >= 0:
                print(
                    f"Disagreement: best descriptors is either \n{tags[a]} or \n{tags[c]}"
                )
            dvs = [a, c]
    elif a == c:
        if verb >= 0:
            print(
                f"Disagreement: best descriptors is either \n{tags[a]} or \n{tags[b]}"
            )
        dvs = [a, b]
    elif b == c:
        if verb >= 0:
            print(
                f"Disagreement: best descriptors is either \n{tags[a]} or \n{tags[b]}"
            )
        dvs = [a, b]
    else:
        if verb >= 0:
            print(
                f"Total disagreement: best descriptors is either \n{tags[a]} or \n{tags[b]} or \n{tags[c]}"
            )
        dvs = [a, b, c]
    r2 = [r2s[i] for i in dvs]
    dvs = [i + 1 for i in dvs]  # Recover the removed step of the reaction
    return dvs, r2


def test_dv1():
    a = np.array(
        [
            0,
            -11.34,
            2.66,
            -14.78,
            0.14,
            -18.22,
            -13.81,
            -20.98,
            -22.26,
            -53.98,
            -43.19,
        ]
    )
    b = np.array(
        [
            0,
            -11.24,
            3.66,
            -16.78,
            0.54,
            -18.52,
            -4.81,
            -21.98,
            -23.26,
            -52.98,
            -43.19,
        ]
    )
    c = np.array(
        [
            0,
            -14.24,
            0.66,
            -14.78,
            0.94,
            -14.52,
            -1.81,
            -20.98,
            -24.26,
            -54.98,
            -43.19,
        ]
    )
    dgr = -43.19
    coeff = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=int)
    tags = np.array(
        [
            "Reactants",
            "Struc2",
            "TS1",
            "Struc3",
            "TS2",
            "Struc4",
            "TS3",
            "Struc5",
            "TS4",
            "Struc6",
            "Product",
        ]
    )
    profiles = np.vstack([a, b, c])
    regress = np.ones((11), dtype=bool)
    dvs, r2s = find_1_dv(profiles, tags, coeff, regress, verb=2)
    assert dvs == [9]
    assert tags[9] == "Struc6"
    assert np.allclose(r2s, [0.6436413778677442, 0.6419178563084623], 4)


def test_imputer():
    coeff = [0, 0, 1, 0, 1, 0, 0, 1]
    dgr = -28.1
    # Ni(PMe3)
    a = np.array([0.0, 0.2, 27.4, 10.5, 23.9, -19.9, -27.5, -0.5])
    # Ni(SiMes3)
    b = np.array([0.0, 6.34, 28.6, 11.42, 22.81, -17.96, -40.31, -0.07])
    # Ni(PBu3)
    c = np.array([0.0, 4.68, 27.95, 13.85, 25.95, -14.93, -24.80, 2.07])
    d = np.vstack([a, b, c])

    # with erased values
    a2 = np.array([0.0, 0.2, 27.4, 10.5, 23.9, -19.9, -27.5, -0.5])
    b2 = np.array([0.0, 6.34, 28.6, np.nan, 22.81, -17.96, -40.31, -0.07])
    c2 = np.array([0.0, 4.68, 27.95, 13.85, 25.95, -14.93, -24.80, 2.07])
    d2 = np.vstack([a2, b2, c2])
    cb = np.ones((3))
    ms = np.ones((3))
    regress = np.ones((8), dtype=bool)
    tags = np.array(
        [
            "Reactants",
            "Struc2",
            "TS1",
            "Struc3",
            "TS2",
            "Struc4",
            "TS3",
            "Struc5",
            "TS4",
            "Struc6",
            "Product",
        ]
    )
    d2_test = np.copy(d2)
    d_simple = curate_d(d2_test, regress, cb, ms, tags, imputer_strat="simple", verb=2)[
        0
    ]

    d2_test = np.copy(d2)
    d_knn = curate_d(d2_test, regress, cb, ms, tags, imputer_strat="knn", verb=2)[0]

    d2_test = np.copy(d2)
    d_iterative = curate_d(
        d2_test, regress, cb, ms, tags, imputer_strat="iterative", verb=2
    )[0]
    assert np.isclose(np.linalg.norm(d - d_simple), 11.50428)
    assert np.isclose(np.linalg.norm(d - d_knn), 0.75500, rtol=0.1)
    assert np.isclose(np.linalg.norm(d - d_iterative), 0.80902, rtol=0.1)


if __name__ == "__main__":
    test_dv1()
    test_imputer()

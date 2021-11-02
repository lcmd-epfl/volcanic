#!/usr/bin/env python

import numpy as np
import sklearn as sk
import sklearn.linear_model
from sklearn.impute import SimpleImputer, KNNImputer
from exceptions import InputError

rng = np.random.default_rng()


def call_imputer(a, b, imputer_strat="simple"):
    if imputer_strat == "simple":
        imputer = SimpleImputer()
        newa = imputer.fit_transform(a.reshape(-1, 1)).flatten()
        return newa
    if imputer_strat == "knn":
        imputer = KNNImputer(n_neighbors=2)
        newa = imputer.fit_transform(np.vstack([b, a]).T)[:, 1]
        return newa
    if imputer_strat == "none":
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
                    f"Among data series {tagsit[outlier]} some outliers (very large values) were detected: {dit[outlier,i].flatten()} and will be skipped."
                )
            dit[outlier, i] = np.nan
        if any(absd < mintol):
            outlier = np.where(absd < mintol)
            if verb > 1:
                print(
                    f"Among data series {tagsit[outlier]} some outliers (very small values) were detected: {dit[outlier,i].flatten()} and will be skipped."
                )
            dit[outlier, i] = np.nan
        if i > 0:
            dit[:, i] = call_imputer(dit[:, i], dit[:, i - 1], imputer_strat)
        if i == 0:
            dit[:, i] = call_imputer(dit[:, i], dit[:, i + 1], imputer_strat)
        curated_d[:, i] = dit[:, i]
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
    try:
        assert np.isclose(d[:, 0].std(), 0)
    except AssertionError as m:
        raise InputError(
            "The first field of every profile should be the same (reference state). Exit."
        )
    tags = tags[1:]
    coeff = coeff[1:]
    regress = regress[1:]
    d = d[:, 1:]
    lnsteps = range(d.shape[1])
    regsteps = range(d[:, regress].shape[1])
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
            Y = d[:, regress][:, j]
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
                f"\nWith {tags[i]} as descriptor the following r2 values were obtained : {ir2s}"
            )
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
    dvs, r2s = find_1_dv(profiles, tags, coeff, regress, verb=-1)
    assert np.allclose(dvs, [5, 1], 4)
    assert np.allclose(r2s, [0.6436413778677442, 0.6419178563084623], 4)


def test_inputer():
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
    a2 = np.array([0.0, 0.2, np.nan, 10.5, 23.9, -19.9, -27.5, -0.5])
    b2 = np.array([0.0, 6.34, 28.6, np.nan, 22.81, -17.96, -40.31, -0.07])
    c2 = np.array([0.0, 4.68, 27.95, 13.85, 25.95, -14.93, np.nan, 2.07])
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
    d_simple = curate_d(d2, regress, cb, ms, tags, imputer_strat="simple", verb=2)[0]
    d_knn = curate_d(d2, regress, cb, ms, tags, imputer_strat="knn", verb=2)[0]
    assert np.isclose(np.linalg.norm(d - d_simple), 9.17805398763812)
    assert np.isclose(np.linalg.norm(d - d_knn), 9.17805398763812)


if __name__ == "__main__":
    test_dv1()
    test_inputer()

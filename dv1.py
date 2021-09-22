#!/usr/bin/env python

import numpy as np
import sklearn as sk
import sklearn.linear_model
from sklearn.impute import SimpleImputer

rng = np.random.default_rng()


def call_imputer(a, imputer_strat="simple"):
    if imputer_strat == "simple":
        imputer = SimpleImputer()
        return imputer.fit_transform(a.reshape(-1, 1)).flatten()
    if imputer_strat == "none":
        return a


def curate_d(d, cb, ms, tags, imputer_strat="simple", verb=0):
    assert isinstance(d, np.ndarray)
    try:
        assert np.isclose(d[:, 0].std(), 0)
    except AssertionError as m:
        print(
            "The first field of every profile should be the same (reference state). Exit."
        )
        exit()
    dit = d[:, 1:]
    tagsit = tags[1:]
    curated_d = d[:, 0].T
    for i in range(dit.shape[1]):
        mean = dit[:, i].mean()
        std = dit[:, i].std()
        maxtol = np.abs(mean) + 3 * std
        mintol = np.abs(mean) - 3 * std
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
        dit[:, i] = call_imputer(dit[:, i], imputer_strat)
        curated_d = np.vstack([curated_d, dit[:, i]])
    curated_d = curated_d.T
    incomplete = np.ones_like(curated_d[:, 0], dtype=bool)
    for i in range(curated_d.shape[0]):
        n_nans = np.count_nonzero(np.isnan(d[i, :]))
        if n_nans > 0:
            if verb > 1:
                print(
                    f"Some of your reaction profiles contain {n_nans} undefined values and will not be considered:\n {d[i,:]}"
                )
            incomplete[i] = False
    curated_d = curated_d[incomplete]
    curated_cb = cb[incomplete]
    curated_ms = ms[incomplete]
    return curated_d, curated_cb, curated_ms


def find_1_dv(d, tags, coeff, verb=0):
    assert isinstance(d, np.ndarray)
    assert len(tags) == len(coeff)
    try:
        assert np.isclose(d[:, 0].std(), 0)
    except AssertionError as m:
        print(
            "The first field of every profile should be the same (reference state). Exit."
        )
        exit()
    tags = tags[1:]
    coeff = coeff[1:]
    d = d[:, 1:]
    lnsteps = range(d.shape[1])
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
        for j in lnsteps:
            Y = d[:, j]
            XY = np.vstack([d[:, i], d[:, j]]).T
            XY = XY[~np.isnan(XY).any(axis=1)]
            X = XY[:, 0].reshape(-1, 1)
            Y = XY[:, 1]
            reg = sk.linear_model.LinearRegression().fit(X, Y)
            imaes.append(sk.metrics.mean_absolute_error(Y, reg.predict(X)))
            imaps.append(sk.metrics.mean_absolute_percentage_error(Y, reg.predict(X)))
            ir2s.append(reg.score(X, Y))
            if i == j:  # Cheap sanity check
                assert np.isclose(imaps.pop(-1), 0)
                assert np.isclose(imaes.pop(-1), 0)
                assert np.isclose(ir2s.pop(-1), 1)
            else:
                if verb > 1:
                    print(
                        f"With {tags[i]} as descriptor, regressed {tags[j]} with r2 : {np.round(ir2s[-1],2)} and MAE: {np.round(imaes[-1],2)}"
                    )
        if verb > 2:
            print(
                f"\nWith {tags[i]} as descriptor the following r2 values were obtained : {ir2s}"
            )
        maes[i] = np.array(imaes).mean()
        r2s[i] = np.array(ir2s).mean()
        maps[i] = np.array(imaps).std()
        if verb > 0:
            print(
                f"\nWith {tags[i]} as descriptor,\n the mean r2 is : {np.round(r2s[i],2)},\n the mean MAE is :  {np.round(maes[i],2)}\n the std MAPE is : {np.round(maps[i],2)}\n"
            )
    criteria = []
    criteria.append(np.squeeze(np.where(r2s == np.max(r2s[~np.ma.make_mask(coeff)]))))
    criteria.append(np.squeeze(np.where(maes == np.min(maes[~np.ma.make_mask(coeff)]))))
    criteria.append(np.squeeze(np.where(maps == np.min(maps[~np.ma.make_mask(coeff)]))))
    for i, criterion in enumerate(criteria):
        if isinstance(criterion, (np.ndarray)):
            if any(criterion.shape):
                criterion = [idx for idx in criterion if ~np.ma.make_mask(coeff)[idx]]
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
    profiles = np.vstack((a, b, c))
    dvs, r2s = find_1_dv(profiles, tags, coeff, verb=-1)
    assert np.allclose(dvs, [5, 1], 4)
    assert np.allclose(r2s, [0.6436413778677442, 0.6419178563084623], 4)


if __name__ == "__main__":
    test_dv1()

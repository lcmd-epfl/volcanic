#!/usr/bin/env python

import itertools
from collections import deque

import numpy as np
import sklearn as sk
import sklearn.linear_model

from navicat_volcanic.exceptions import InputError

rng = np.random.default_rng()
sup = np.testing.suppress_warnings()
sup.filter(module=np.core)


def count_iter_items(iterable):
    """
    Consume an iterable not reading it into memory; return the number of items.
    """
    counter = itertools.count()
    deque(zip(iterable, counter), maxlen=0)  # (consume at C speed)
    return next(counter)


def dv_collinear(X, verb=0):
    u, s, vh = np.linalg.svd(X, full_matrices=True)
    c_idx = np.ones_like(s)
    c_idx = np.max(s) / s
    if np.any(np.where(c_idx > 1e9)):
        if verb > 1:
            print(
                f"Descriptors are likely to be linearly dependant given SVD. Skipping."
            )
        return True
    else:
        return False


@sup
def find_2_dv(d, tags, coeff, regress, verb=0):
    assert isinstance(d, np.ndarray)
    assert len(tags) == len(coeff)
    try:
        assert np.isclose(d[:, 0].std(), 0)
    except AssertionError as m:
        raise InputError(
            "The first field of every profile should be the same (reference state). Exit."
        )
    if np.isclose(d[:, -1].std(), 0):
        regress[-1] = False
        if verb > 0:
            print(f"\nReaction energy is constant. Assuming substrates are constant.")
    tags = tags[1:]
    coeff = coeff[1:]
    regress = regress[1:]
    d = d[:, 1:]
    lnsteps = range(d.shape[1])
    regsteps = range(d[:, regress].shape[1])
    pnsteps = itertools.combinations(lnsteps, r=2)
    lpnsteps = [pair for pair in pnsteps]
    pn = len(lpnsteps)
    pcoeff = np.zeros(pn)
    ptags = np.empty(pn, dtype=object)
    for i, pair in enumerate(lpnsteps):
        if coeff[pair[0]] or coeff[pair[1]]:
            pcoeff[i] = 1
        ptags[i] = tags[pair[0]] + " and " + tags[pair[1]]
    # Regression diagnostics
    maes = np.ones(pn)
    r2s = np.ones(pn)
    maps = np.ones(pn)
    for idx, ij in enumerate(lpnsteps):
        i = ij[0]
        j = ij[1]
        if verb > 0:
            print(
                f"\nTrying combination of {tags[i]} and {tags[j]} as descriptor variable:"
            )
        imaes = []
        imaps = []
        ir2s = []
        for k in regsteps:
            Y = d[:, regress][:, k]
            XY = np.vstack([[d[:, i], d[:, j]], d[:, k]]).T
            XY = XY[~np.isnan(XY).any(axis=1)]
            X = XY[:, :2]
            if dv_collinear(X, verb):
                maes[idx] = np.nan
                r2s[idx] = 0
                maps[idx] = np.nan
                break
            Y = XY[:, 2]
            # Fitting using scikit-learn LinearModel
            reg = sk.linear_model.LinearRegression().fit(X, Y)
            imaes.append(sk.metrics.mean_absolute_error(Y, reg.predict(X)))
            imaps.append(sk.metrics.mean_absolute_percentage_error(Y, reg.predict(X)))
            ir2s.append(reg.score(X, Y))
            if verb > 1:
                print(
                    f"With a combination of {tags[i]} and {tags[j]} as descriptor, regressed {tags[k]} with r2 : {np.round(ir2s[-1],2)} and MAE: {np.round(imaes[-1],2)}"
                )

            if verb > 2:
                print(
                    f"Linear model has coefficients : {reg.coef_} \n and intercept {reg.intercept_}"
                )
        if verb > 2:
            print(
                f"\nWith a combination of {tags[i]} and {tags[j]} as descriptor the following r2 values were obtained : {np.round(ir2s,2)}"
            )
        maes[idx] = np.around(np.array(imaes).mean(), 4)
        r2s[idx] = np.around(np.array(ir2s).mean(), 4)
        maps[idx] = np.around(np.array(imaps).std(), 4)
        if verb > 0:
            print(
                f"\nWith a combination of {tags[i]} and {tags[j]} as descriptor,\n the mean r2 is : {np.round(r2s[idx],2)},\n the mean MAE is :  {np.round(maes[idx],2)}\n the std MAPE is : {np.round(maps[idx],2)}\n"
            )
    criteria = []
    criteria.append(
        np.squeeze(np.where(r2s == np.nanmax(r2s[~np.ma.make_mask(pcoeff)])))
    )
    criteria.append(
        np.squeeze(np.where(maes == np.nanmin(maes[~np.ma.make_mask(pcoeff)])))
    )
    criteria.append(
        np.squeeze(np.where(maps == np.nanmin(maps[~np.ma.make_mask(pcoeff)])))
    )
    for i, criterion in enumerate(criteria):
        if isinstance(criterion, (np.ndarray)):
            if any(criterion.shape):
                if verb > 2:
                    print(f"A random choice has been made to break a score tie.")
                criterion = [idx for idx in criterion if ~np.ma.make_mask(pcoeff)[idx]]
                criteria[i] = rng.choice(criterion, size=1)
    a = int(criteria[0])
    b = int(criteria[1])
    c = int(criteria[2])
    if a == b:
        if a == c:
            if verb >= 0:
                print(
                    f"All indicators agree: best descriptor is the combination {ptags[a]}"
                )
            dvs = [a]
        else:
            if verb >= 0:
                print(
                    f"Disagreement: best descriptors is either the combination \n{ptags[a]} or \n{ptags[c]}"
                )
            dvs = [a, c]
    elif a == c:
        if verb >= 0:
            print(
                f"Disagreement: best descriptors is either the combination \n{ptags[a]} or \n{ptags[b]}"
            )
        dvs = [a, b]
    elif b == c:
        if verb >= 0:
            print(
                f"Disagreement: best descriptors is either the combination \n{ptags[a]} or \n{ptags[b]}"
            )
        dvs = [a, b]
    else:
        if verb >= 0:
            print(
                f"Total disagreement: best descriptors is either the combination \n{ptags[a]} or \n{ptags[b]} or \n{ptags[c]}"
            )
        dvs = [a, b, c]
    r2 = [r2s[i] for i in dvs]
    dvs = [lpnsteps[i] for i in dvs]
    return dvs, r2


def test_dv2():
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
    d = np.array(
        [
            0,
            -18.24,
            2.66,
            -17.78,
            1.14,
            -15.52,
            -2.81,
            -25.98,
            -21.26,
            -50.98,
            -43.19,
        ]
    )
    dgr = -43.19
    coeff = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=bool)
    regress = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=bool)
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
    profiles = np.vstack([a, b, c, d])
    dvs, r2s = find_2_dv(profiles, tags, coeff, regress, verb=-1)
    assert np.allclose(r2s, [0.8573974153281095, 0.832373403431629], 4)


if __name__ == "__main__":
    test_dv2()

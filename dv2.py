#!/usr/bin/env python

import numpy as np
import sklearn as sk
import sklearn.linear_model
import itertools
from collections import deque

rng = np.random.default_rng()


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
    if any(np.where(c_idx > 250)):
        if verb > 1:
            print(
                "Descriptors are likley to be linearly dependant given SVD :\n {s} \n Skipping."
            )
        return True
    else:
        return False


def find_2_dv(d, tags, coeff, verb=0):
    assert isinstance(d, np.ndarray)
    assert len(tags) == len(coeff)
    try:
        assert np.isclose(d[:, 0].std(), 0)
        assert np.isclose(d[:, -1].std(), 0)
    except AssertionError as m:
        print(
            "The first and last fields of every profile should be the same (reference state and reaction free energy). Exit."
        )
        exit()
    tags = tags[1:-1]
    coeff = coeff[1:-1]
    d = d[:, 1:-1]
    lnsteps = range(d.shape[1])
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
        for k in lnsteps:
            Y = d[:, k]
            XY = np.vstack([[d[:, i], d[:, j]], d[:, k]]).T
            XY = XY[~np.isnan(XY).any(axis=1)]
            X = XY[:, :2]
            if dv_collinear(X):
                maes[idx] = np.nan
                r2s[idx] = 0
                maps[idx] = np.nan
                continue
            Y = XY[:, 2]
            # Fitting using scikit-learn LinearModel
            reg = sk.linear_model.LinearRegression().fit(X, Y)
            imaes.append(sk.metrics.mean_absolute_error(Y, reg.predict(X)))
            imaps.append(sk.metrics.mean_absolute_percentage_error(Y, reg.predict(X)))
            ir2s.append(reg.score(X, Y))
            if i == k or j == k:
                assert np.isclose(imaps.pop(-1), 0)
                assert np.isclose(imaes.pop(-1), 0)
                assert np.isclose(ir2s.pop(-1), 1)
            else:
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
                f"\nWith a combination of {tags[i]} and {tags[j]} as descriptor the following r2 values were obtained : {ir2s}"
            )
        maes[idx] = np.array(imaes).mean()
        r2s[idx] = np.array(ir2s).mean()
        maps[idx] = np.array(imaps).std()
        if verb > 0:
            print(
                f"\nWith a combination of {tags[i]} and {tags[j]} as descriptor,\n the mean r2 is : {np.round(r2s[idx],2)},\n the mean MAE is :  {np.round(maes[idx],2)}\n the std MAPE is : {np.round(maps[idx],2)}\n"
            )
    criteria = []
    criteria.append(np.squeeze(np.where(r2s == np.max(r2s[~np.ma.make_mask(pcoeff)]))))
    criteria.append(
        np.squeeze(np.where(maes == np.min(maes[~np.ma.make_mask(pcoeff)])))
    )
    criteria.append(
        np.squeeze(np.where(maps == np.min(maps[~np.ma.make_mask(pcoeff)])))
    )
    for i, criterion in enumerate(criteria):
        if isinstance(criterion, (np.ndarray)):
            if any(criterion.shape):
                criterion = [idx for idx in criterion if ~np.ma.make_mask(pcoeff)[idx]]
                criteria[i] = rng.choice(criterion, size=1)
    a = int(criteria[0])
    b = int(criteria[1])
    c = int(criteria[2])
    if a == b:
        if a == c:
            print(
                f"All indicators agree: best descriptor is the combination {ptags[a]}"
            )
            dvs = [a]
        else:
            print(
                f"Disagreement: best descriptors is either the combination \n{ptags[a]} or \n{ptags[c]}"
            )
            dvs = [a, c]
    elif a == c:
        print(
            f"Disagreement: best descriptors is either the combination \n{ptags[a]} or \n{ptags[b]}"
        )
        dvs = [a, b]
    elif b == c:
        print(
            f"Disagreement: best descriptors is either the combination \n{ptags[a]} or \n{ptags[b]}"
        )
        dvs = [a, b]
    else:
        print(
            f"Total disagreement: best descriptors is either the combination \n{ptags[a]} or \n{ptags[b]} or \n{ptags[c]}"
        )
        dvs = [a, b, c]
    r2 = [r2s[i] for i in dvs]
    dvs = [lpnsteps[i] for i in dvs]
    return dvs, r2

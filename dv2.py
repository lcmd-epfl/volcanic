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
    # Regression diagnostics
    maes = np.ones(pn)
    r2s = np.ones(pn)
    maps = np.ones(pn)
    for i, j in itertools.combinations(lnsteps, r=2):
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
            X = XY[:, :1].reshape(-1, 1)
            Y = XY[:, 2]
            reg = sk.linear_model.LinearRegression().fit(X, Y)
            imaes.append(sk.metrics.mean_absolute_error(Y, reg.predict(X)))
            imaps.append(sk.metrics.mean_absolute_percentage_error(Y, reg.predict(X)))
            ir2s.append(reg.score(X, Y))
            if i == k or j == k:
                pass
            else:
                if verb > 1:
                    print(
                        f"With a combination of {tags[i]} and {tags[j]} as descriptor, regressed {tags[k]} with r2 : {np.round(ir2s[-1],2)} and MAE: {np.round(imaes[-1],2)}"
                    )
        if verb > 2:
            print(
                f"\nWith a combination of {tags[i]} and {tags[j]} as descriptor the following r2 values were obtained : {ir2s}"
            )
        maes[i] = np.array(imaes).mean()
        r2s[i] = np.array(ir2s).mean()
        maps[i] = np.array(imaps).std()
        if verb > 0:
            print(
                f"\nWith a combination of {tags[i]} and {tags[j]} as descriptor,\n the mean r2 is : {np.round(r2s[i],2)},\n the mean MAE is :  {np.round(maes[i],2)}\n the std MAPE is : {np.round(maps[i],2)}\n"
            )
    criteria = []
    criteria.append(np.squeeze(np.where(r2s == np.max(r2s[~np.ma.make_mask(coeff)]))))
    criteria.append(np.squeeze(np.where(maes == np.min(maes[~np.ma.make_mask(coeff)]))))
    criteria.append(np.squeeze(np.where(maps == np.min(maps[~np.ma.make_mask(coeff)]))))
    for i, criterion in enumerate(criteria):
        if isinstance(criterion, (np.ndarray)):
            if any(criterion.shape):
                criteria[i] = criterion[rng.integers(low=0, high=criterion.shape[0])]
    a = criteria[0]
    b = criteria[1]
    c = criteria[2]
    dvs = []
    if a == b:
        if a == c:
            print(f"All indicators agree: best descriptor is {tags[a]}")
            dvs.append(a)
        else:
            print(
                f"Disagreement: best descriptors is either \n{tags[a]} or \n{tags[c]}"
            )
            dvs = [a, c]
    elif a == c:
        print(f"Disagreement: best descriptors is either \n{tags[a]} or \n{tags[b]}")
        dvs = [a, b]
    elif b == c:
        print(f"Disagreement: best descriptors is either \n{tags[a]} or \n{tags[b]}")
        dvs = [a, b]
    else:
        print(
            f"Total disagreement: best descriptors is either \n{tags[a]} or \n{tags[b]} or \n{tags[c]}"
        )
        dvs = [a, b, c]
    r2 = [r2s[i] for i in dvs]
    dvs = [i + 1 for i in dvs]  # Recover the removed step of the reaction
    return dvs, r2

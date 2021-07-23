#!/usr/bin/env python

import numpy as np
import scipy.stats as stats
import matplotlib
import copy

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model
from sklearn.impute import SimpleImputer
from helpers import bround
from tof import calc_tof


def call_imputer(a, imputer_strat="simple"):
    if imputer_strat == "simple":
        imputer = SimpleImputer()
        return imputer.fit_transform(a[:, i].reshape(-1, 1)).flatten()
    if imputer_strat == "none":
        return a


def curate_d(d, cb, ms, tags, imputer_strat="simple", verb=0):
    assert isinstance(d, np.ndarray)
    try:
        assert np.isclose(d[:, 0].std(), 0)
        assert np.isclose(d[:, -1].std(), 0)
    except AssertionError as m:
        print(
            "The first and last fields of every profile should be the same (reference state and reaction free energy). Exit."
        )
        exit()
    dit = d[:, 1:-1]
    tagsit = tags[1:-1]
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
                    f"Among data series {tagsit[i]} some big outliers were detected: {dit[outlier,i].flatten()} and will be skipped."
                )
            dit[outlier, i] = np.nan
        if any(absd < mintol):
            outlier = np.where(absd < mintol)
            if verb > 1:
                print(
                    f"Among data series {tagsit[i]} some tiny outliers were detected: {dit[outlier,i].flatten()} and will be skipped."
                )
            dit[outlier, i] = np.nan
        dit[:, i] = call_imputer(dit[:, i], imputer_strat)
        curated_d = np.vstack([curated_d, dit[:, i]])
    curated_d = np.vstack([curated_d, d[:, -1]]).T
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


def find_dv(d, tags, coeff, verb=0):
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
    a = np.squeeze(np.where(r2s == np.max(r2s[~np.ma.make_mask(coeff)])))
    b = np.squeeze(np.where(maes == np.min(maes[~np.ma.make_mask(coeff)])))
    c = np.squeeze(np.where(maps == np.min(maps[~np.ma.make_mask(coeff)])))
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


def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    if ax is None:
        ax = plt.gca()

    ci = (
        t
        * s_err
        * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    )
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", alpha=0.6)

    return ax


def plot_lsfer(idx, d, tags, coeff, cb="white", ms="o", verb=0):
    dnan = d[np.isnan(d)]
    d_refill = np.zeros_like(d)
    d_refill[~np.isnan(d)] = d[~np.isnan(d)]
    tags = [str(tag) for tag in tags]
    lnsteps = range(d.shape[1])
    Xf = d[:, idx].reshape(-1)
    npoints = 500
    mape = 100
    for j in lnsteps[1:-1]:
        if verb > 0:
            print(f"Plotting regression of {tags[j]}.")
        Yf = d[:, j].reshape(-1)
        XY = np.vstack([Xf, Yf]).T
        if isinstance(cb, np.ndarray):
            cbi = np.array(cb)[~np.isnan(XY).any(axis=1)]
        else:
            cbi = cb
        if isinstance(ms, np.ndarray):
            msi = np.array(ms)[~np.isnan(XY).any(axis=1)]
        else:
            msi = ms
        XYm = XY[np.isnan(XY).any(axis=1)]
        XY = XY[~np.isnan(XY).any(axis=1)]
        Xm = XYm[:, 0].reshape(-1)
        Ym = XYm[:, 1].reshape(-1)
        X = XY[:, 0].reshape(-1)
        Y = XY[:, 1].reshape(-1)
        xmax = bround(X.max() + 10)
        xmin = bround(X.min() - 10)
        if verb > 2:
            print(f"Range of descriptor set to [ {xmin} , {xmax} ]")
        xint = np.linspace(xmin, xmax, npoints)
        p, cov = np.polyfit(X, Y, 1, cov=True)
        Y_pred = np.polyval(p, X)
        currmape = sk.metrics.mean_absolute_percentage_error(Y, Y_pred)
        for k, y in enumerate(Ym):
            if not np.isnan(Xm[k]) and np.isnan(Ym[k]):
                Ym[k] = np.polyval(p, Xm[k])
                d_refill[np.isnan(d).any(axis=1)][:, j][k] = Ym[k]
            elif not np.isnan(Ym[k]):
                if currmape < mape:
                    ptemp = p
                    ptemp[-1] -= Ym[k]
                    Xm[k] = np.roots(ptemp)
                    d_refill[np.isnan(d).any(axis=1)][:, idx][k] = Xm[k]
                    mape = currmape
            else:
                print(
                    f"Both descriptor and regression target are undefined. This should have been fixed before this point. Exiting."
                )
                exit()
        n = Y.size
        m = p.size
        dof = n - m
        t = stats.t.ppf(0.95, dof)
        resid = Y - Y_pred
        chi2 = np.sum((resid / Y_pred) ** 2)
        s_err = np.sqrt(np.sum(resid ** 2) / dof)
        fig, ax = plt.subplots(
            frameon=False, figsize=[3, 3], dpi=300, constrained_layout=True
        )
        yint = np.polyval(p, xint)
        plot_ci_manual(t, s_err, n, X, xint, yint, ax=ax)
        pi = (
            t
            * s_err
            * np.sqrt(
                1 + 1 / n + (xint - np.mean(X)) ** 2 / np.sum((X - np.mean(X)) ** 2)
            )
        )
        ax.plot(xint, yint, "-", linewidth=1, color="#000a75", alpha=0.85)
        for i in range(len(X)):
            ax.scatter(
                X[i],
                Y[i],
                s=5,
                c=cbi[i],
                marker=msi[i],
                linewidths=0.15,
                edgecolors="black",
            )
        # Border
        ax.spines["top"].set_color("black")
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_color("black")
        ax.spines["right"].set_color("black")
        ax.get_xaxis().set_tick_params(direction="out")
        ax.get_yaxis().set_tick_params(direction="out")
        ax.xaxis.tick_bottom()
        ax.yaxis.tick_left()
        # Labels and key
        plt.xlabel(f"{tags[idx]} [kcal/mol]")
        plt.ylabel(f"{tags[j]} [kcal/mol]")
        plt.xlim(xmin, xmax)
        plt.savefig(f"{tags[j]}.png")
    return d_refill


def calc_es(profile, dgr, esp=True):
    es1 = 0
    for i, lower in enumerate(profile):
        view = copy.deepcopy(profile)
        view[:i] += dgr
        j = np.argmax(view)
        upper = view[j]
        es2 = upper - lower
        if es2 > es1:
            es1 = es2
            imax = j
            imin = i
    return [-es1, imax, imin]


def plot_2d(
    x,
    y,
    px,
    py,
    xmin,
    xmax,
    xlabel="X-axis",
    ylabel="Y-axis",
    filename="plot.png",
    rid=None,
    rb=None,
    cb="white",
    ms="o",
):
    fig, ax = plt.subplots(
        frameon=False, figsize=[3, 3], dpi=300, constrained_layout=True
    )

    ax.plot(x, y, "-", linewidth=1, color="#000a75", alpha=0.85)
    for i in range(len(px)):
        ax.scatter(
            px[i],
            py[i],
            s=5,
            c=cb[i],
            marker=ms[i],
            linewidths=0.15,
            edgecolors="black",
        )
    # Border
    ax.spines["top"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.get_xaxis().set_tick_params(direction="out")
    ax.get_yaxis().set_tick_params(direction="out")
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    # Labels and key
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    if rid is not None and rb is not None:
        avgs = []
        rb.append(xmax)
        for i in range(len(rb) - 1):
            avgs.append((rb[i] + rb[i + 1]) / 2)
        for i in rb:
            ax.axvline(
                i,
                linestyle="dashed",
                color="black",
                linewidth=0.5,
                alpha=0.75,
            )
        yavg = (y.max() + y.min()) / 2
        for i, j in zip(rid, avgs):
            plt.text(
                j,
                yavg,
                i,
                fontsize=6,
                horizontalalignment="center",
                verticalalignment="center",
                rotation="vertical",
            )
    plt.savefig(filename)


def plot_volcano(idx, d, tags, coeff, dgr, cb="white", ms="o", verb=0):
    tags = [str(tag) for tag in tags]
    lnsteps = range(d.shape[1])
    X = d[:, idx].reshape(-1)
    xmax = bround(X.max() + 35)
    xmin = bround(X.min() - 35)
    npoints = 250
    if verb > 1:
        print(f"Range of descriptor set to [ {xmin} , {xmax} ]")
    xint = np.linspace(xmin, xmax, npoints)
    dgs = np.zeros((npoints, len(lnsteps)))
    for i, j in enumerate(lnsteps):
        Y = d[:, j].reshape(-1)
        p, cov = np.polyfit(X, Y, 1, cov=True)  # 1 -> degree of polynomial
        Y_pred = np.polyval(p, X)
        n = Y.size
        m = p.size
        dof = n - m
        t = stats.t.ppf(0.95, dof)
        resid = Y - Y_pred
        with np.errstate(invalid="ignore"):
            chi2 = np.sum((resid / Y_pred) ** 2)
        s_err = np.sqrt(np.sum(resid ** 2) / dof)
        yint = np.polyval(p, xint)
        dgs[:, i] = yint
    ymin = np.zeros_like(yint)
    ridmax = np.zeros_like(yint, dtype=int)
    ridmin = np.zeros_like(yint, dtype=int)
    rid = []
    rb = []
    for i in range(ymin.shape[0]):
        profile = dgs[i, :]
        ymin[i], ridmax[i], ridmin[i] = calc_es(profile, dgr, esp=True)
        if ridmax[i] != ridmax[i - 1] or ridmin[i] != ridmin[i - 1]:
            rid.append(f"{tags[ridmin[i]]} ➜ {tags[ridmax[i]]}")
            rb.append(xint[i])
    if verb > 0:
        print(f"Identified {len(rid)} different determining states.")
        for i, j in zip(rid, rb):
            print(f"{i} starting at {j}")
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        profile = d[i, :]
        px[i] = d[i, idx].reshape(-1)
        py[i] = calc_es(profile, dgr, esp=True)[0]
    xlabel = f"{tags[idx]} [kcal/mol]"
    ylabel = "-ΔG(pds) [kcal/mol]"
    filename = f"volcano_{tags[idx]}.png"
    if verb > 0:
        csvname = f"volcano_{tags[idx]}.csv"
        print(f"Saving volcano data to file {csvname}")
        zdata = list(zip(xint, ymin))
        np.savetxt(
            csvname, zdata, fmt="%.4e", delimiter=",", header="Descriptor, -\DGpds"
        )
    plot_2d(
        xint, ymin, px, py, xmin, xmax, xlabel, ylabel, filename, rid, rb, cb=cb, ms=ms
    )
    return xint, ymin, px, py, xmin, xmax, rid, rb


def plot_tof_volcano(idx, d, tags, coeff, dgr, T=298.15, cb="white", ms="o", verb=0):
    tags = [str(tag) for tag in tags]
    lnsteps = range(d.shape[1])
    X = d[:, idx].reshape(-1)
    xmax = bround(X.max() + 15)
    xmin = bround(X.min() - 15)
    npoints = 250
    if verb > 1:
        print(f"Range of descriptor set to [ {xmin} , {xmax} ]")
    xint = np.linspace(xmin, xmax, npoints)
    dgs = np.zeros((npoints, len(lnsteps)))
    for i, j in enumerate(lnsteps):
        Y = d[:, j].reshape(-1)
        p, cov = np.polyfit(X, Y, 1, cov=True)
        Y_pred = np.polyval(p, X)
        n = Y.size
        m = p.size
        dof = n - m
        t = stats.t.ppf(0.95, dof)
        resid = Y - Y_pred
        with np.errstate(invalid="ignore"):
            chi2 = np.sum((resid / Y_pred) ** 2)
        s_err = np.sqrt(np.sum(resid ** 2) / dof)
        yint = np.polyval(p, xint)
        dgs[:, i] = yint
    ytof = np.zeros_like(yint)
    # We must take the initial and ending states into account here
    for i in range(ytof.shape[0]):
        profile = dgs[i, :]
        tof = np.log10(calc_tof(profile, dgr, T, coeff, exact=True)[0])
        ytof[i] = tof
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        px[i] = d[i, idx].reshape(-1)
        profile = d[i, :]
        tof = np.log10(calc_tof(profile, dgr, T, coeff, exact=True)[0])
        py[i] = tof
        if verb > 2:
            print(f"Profile {profile} corresponds with log10(TOF) of {tof}")
    xlabel = f"{tags[idx]} [kcal/mol]"
    ylabel = "log(TOF) [1/s]"
    filename = f"tof_volcano_{tags[idx]}.png"
    if verb > 0:
        csvname = f"tof_volcano_{tags[idx]}.csv"
        print(f"Saving TOF volcano data to file {csvname}")
        zdata = list(zip(xint, ytof))
        np.savetxt(
            csvname, zdata, fmt="%.4e", delimiter=",", header="Descriptor, log10(TOF)"
        )
    plot_2d(xint, ytof, px, py, xmin, xmax, xlabel, ylabel, filename, cb=cb, ms=ms)
    return xint, ytof, px, py, xmin, xmax


if __name__ == "__main__":
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
            -11.34,
            2.66,
            -14.78,
            0.14,
            -18.22,
            -3.81,
            -20.98,
            -22.26,
            -53.98,
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
            "Struc6-NM",
            "Product",
        ]
    )
    # find_dv(profiles, tags, coeff, verb=3)
    tof0 = np.log10(calc_tof(a, dgr, 298.15, coeff, exact=True, verb=3)[0])
    tof1 = np.log10(calc_tof(a, dgr, 298.15, coeff, exact=False, verb=3)[0])
    k1, k2, k3 = calc_es(a, dgr, esp=True)
    print(f"For profile {a}, ES is {k1} with indices {k2} and {k3}")
    print(f"Profile {a} corresponds with log10(TOF) of {tof0} or approximately {tof1}")
    tof0 = np.log10(calc_tof(b, dgr, 298.15, coeff, exact=True, verb=3)[0])
    tof1 = np.log10(calc_tof(b, dgr, 298.15, coeff, exact=False, verb=3)[0])
    k1, k2, k3 = calc_es(b, dgr, esp=True)
    print(f"For profile {b}, ES is {k1} with indices {k2} and {k3}")
    print(f"Profile {b} corresponds with log10(TOF) of {tof0} or approximately {tof1}")

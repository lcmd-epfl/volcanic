#!/usr/bin/env python

import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model
from helpers import bround
from tof import calc_tof


def find_dv(d, tags, coeff, verb=0):
    assert isinstance(d, np.ndarray)
    assert len(tags) == len(coeff)
    try:
        assert np.isclose(d[:, 0].std(), 0)
        assert np.isclose(d[:, -1].std(), 0)
    except:
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
        print(f"\nTrying {tags[i]} as descriptor variable:")
        imaes = []
        imaps = []
        ir2s = []
        X = d[:, i].reshape(-1, 1)
        for j in lnsteps:
            Y = d[:, j]
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
        print(
            f"\nWith {tags[i]} as descriptor,\n the mean r2 is : {np.round(r2s[i],2)},\n the mean MAE is :  {np.round(maes[i],2)}\n the std MAPE is : {np.round(maps[i],2)}\n"
        )
    a = np.squeeze(np.where(r2s == np.max(r2s[~np.ma.make_mask(coeff)])))[0]
    b = np.squeeze(np.where(maes == np.min(maes[~np.ma.make_mask(coeff)])))[0]
    c = np.squeeze(np.where(maps == np.min(maps[~np.ma.make_mask(coeff)])))[0]
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
    dvs = [i + 1 for i in dvs]  # Recover the removed step of the reaction
    return dvs


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


def plot_lsfer(idx, d, tags, coeff, cb="white", verb=0):
    tags = [str(tag) for tag in tags]
    lnsteps = range(d.shape[1])
    X = d[:, idx].reshape(-1)
    xmax = bround(X.max() + 10)
    xmin = bround(X.min() - 10)
    npoints = 500
    if verb > 1:
        print(f"Range of descriptor set to [ {xmin} , {xmax} ]")
    xint = np.linspace(xmin, xmax, npoints)
    for j in lnsteps[1:-1]:
        if verb > 0:
            print(f"Plotting regression of {tags[j]}.")
        Y = d[:, j].reshape(-1)
        p, cov = np.polyfit(X, Y, 1, cov=True)
        Y_pred = np.polyval(p, X)
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
        ax.scatter(
            X,
            Y,
            s=2.5,
            c=cb,
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


def calc_es(profile, dgr, esp=True):
    imax = np.argmax(profile)
    imin = np.argmin(profile)
    if (imax < imin) and esp:
        profile[0:imin] += dgr
        imax = np.argmax(profile)
    es = -(profile[imax] - profile[imin])
    return [es, imax, imin]


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
):
    fig, ax = plt.subplots(
        frameon=False, figsize=[3, 3], dpi=300, constrained_layout=True
    )

    ax.plot(x, y, "-", linewidth=1, color="#000a75", alpha=0.85)
    ax.scatter(
        px,
        py,
        s=2.5,
        c=cb,
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


def plot_volcano(idx, d, tags, coeff, dgr, cb="white", verb=0):
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
        ymin[i], ridmax[i], ridmin[i] = calc_es(profile, dgr, esp=False)
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
        py[i] = calc_es(profile, dgr, esp=False)[0]
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
    plot_2d(xint, ymin, px, py, xmin, xmax, xlabel, ylabel, filename, rid, rb, cb=cb)
    return xint, ymin, px, py, xmin, xmax, rid, rb


def plot_tof_volcano(idx, d, tags, coeff, dgr, T=298.15, cb="white", verb=None):
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
    plot_2d(xint, ytof, px, py, xmin, xmax, xlabel, ylabel, filename, cb=cb)
    return xint, ytof, px, py, xmin, xmax


if __name__ == "__main__":
    a = np.array(
        [
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
        ]
    )
    dgr = -43.19
    noise = np.multiply(np.ones_like(a), np.random.normal(1, 0.1, a.shape[1]))
    noise[0] = 1.0
    noise[-1] = 1.0
    b = np.multiply(a, noise)
    c = np.multiply(a, noise[::-1])
    d = np.multiply(b, noise)
    e = np.multiply(b, noise[::-1])
    f = np.multiply(d, noise)
    g = np.multiply(e, noise)
    profiles = np.concatenate([a, b, c, d, e, f, g], axis=0)
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
    assert len(profiles[0, :]) == len(tags) == len(coeff)
    find_dv(profiles, tags, coeff, verb=3)
    profile = profiles[0, :]
    tof0 = np.log10(calc_tof(profile, dgr, 298.15, coeff, exact=True, verb=3)[0])
    tof1 = np.log10(calc_tof(profile, dgr, 298.15, coeff, exact=False, verb=3)[0])
    print(
        f"Profile {profile} corresponds with log10(TOF) of {tof0} or approximately {tof1}"
    )

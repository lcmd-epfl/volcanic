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


def find_dv(d, tags, coeff, lnsteps, verb=0):
    assert isinstance(d, np.ndarray)
    # Regression diagnostics
    tags = [str(tag) for tag in tags]
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
    if verb > 2:
        assert isinstance(np.ma.make_mask(coeff), np.ndarray)
        # print(
        #    f"{tags[~np.ma.make_mask(coeff)]}\n {r2s[~np.ma.make_mask(coeff)]}\n {maes[~np.ma.make_mask(coeff)]}\n {maps[~np.ma.make_mask(coeff)]}"
        # )
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
    return dvs


def plot_ci_manual(t, s_err, n, x, x2, y2, ax=None):
    if ax is None:
        ax = plt.gca()

    ci = (
        t
        * s_err
        * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    )
    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", alpha=0.3)

    return ax


def plot_lsfer(idx, d, tags, coeff, lnsteps, verb):
    tags = [str(tag) for tag in tags]
    X = d[:, idx].reshape(-1)
    xmax = bround(X.max() + 10)
    xmin = bround(X.min() - 10)
    npoints = 500
    if verb > 0:
        print(f"Range of descriptor set to [ {xmin} , {xmax} ]")
    xint = np.linspace(xmin, xmax, npoints)
    for j in lnsteps:
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

        ax.plot(
            X,
            Y,
            "o",
            color="black",
            markersize=2.0,
            markeredgewidth=0.5,
            markerfacecolor="None",
        )
        yint = np.polyval(p, xint)
        ax.plot(xint, yint, "-", linewidth=1.2)
        plot_ci_manual(t, s_err, n, X, xint, yint, ax=ax)
        pi = (
            t
            * s_err
            * np.sqrt(
                1 + 1 / n + (xint - np.mean(X)) ** 2 / np.sum((X - np.mean(X)) ** 2)
            )
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


def calc_es(profile, dgr, esp=False):
    imax = np.argmax(profile)
    imin = np.argmin(profile)
    if (imax > imin or imax < imin) and esp:
        profile[0:imin] += dgr
        imax = np.argmax(profile)
    return profile[imax] - profile[imin]


def plot_2d(x, y, px, py, xmin, xmax, xlabel, ylabel, filename="plot.png"):
    fig, ax = plt.subplots(
        frameon=False, figsize=[3, 3], dpi=300, constrained_layout=True
    )
    ax.plot(
        px,
        py,
        "o",
        color="black",
        markersize=2.0,
        markeredgewidth=0.5,
        markerfacecolor="None",
    )
    ax.plot(x, y, "-", linewidth=1.2)
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
    plt.savefig(filename)


def plot_volcano(idx, d, tags, coeff, lnsteps, dgr, verb):
    tags = [str(tag) for tag in tags]
    X = d[:, idx].reshape(-1)
    xmax = bround(X.max() + 35)
    xmin = bround(X.min() - 35)
    npoints = 250
    if verb > 0:
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
    ymin = np.zeros_like(yint)
    for i in range(ymin.shape[0]):
        ymin[i] = -calc_es(dgs[i, :], dgr, esp=False)
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        px[i] = d[i, idx].reshape(-1)
        py[i] = -calc_es(d[i, :], dgr, esp=False)
    xlabel = f"{tags[idx]} [kcal/mol]"
    ylabel = "-ΔG(pds) [kcal/mol]"
    filename = f"volcano_{tags[idx]}.png"
    plot_2d(xint, ymin, px, py, xmin, xmax, xlabel, ylabel, filename)


def plot_tof_volcano(idx, d, tags, coeff, lnsteps, dgr, T, verb):
    tags = [str(tag) for tag in tags]
    X = d[:, idx].reshape(-1)
    xmax = bround(X.max() + 35)
    xmin = bround(X.min() - 35)
    npoints = 250
    if verb > 0:
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
    coeff = np.append(0, coeff)
    coeff = np.append(coeff, 0)
    for i in range(ytof.shape[0]):
        profile = np.append(np.append(0, dgs[i, :]), dgr)
        tof = np.log10(calc_tof(profile, dgr, T, coeff, exact=True)[0])
        ytof[i] = tof
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        px[i] = d[i, idx].reshape(-1)
        profile = np.append(np.append(0, d[i, :]), dgr)
        tof = np.log10(calc_tof(profile, dgr, T, coeff, exact=True)[0])
        py[i] = tof
        if verb > 1:
            print(f"Profile {profile} corresponds with log10(TOF) of {tof}")
    xlabel = f"{tags[idx]} [kcal/mol]"
    ylabel = "TOF [1/s]"
    filename = f"tof_volcano_{tags[idx]}.png"
    plot_2d(xint, ytof, px, py, xmin, xmax, xlabel, ylabel, filename)


if __name__ == "__main__":
    a = np.array([[-11.34, 2.66, -14.78, 0.14, -18.22, -13.81, -20.98, -22.26, -53.98]])
    dgr = -43.19
    lnsteps = range(a.shape[1])
    noise1 = np.multiply(np.ones_like(a), np.random.normal(1, 0.25, a.shape[1]))
    noise2 = np.multiply(np.ones_like(a), np.random.normal(1, 0.15, a.shape[1]))
    b = np.multiply(a, noise1)
    c = np.multiply(b, noise1[::-1])
    d = np.multiply(b, noise2)
    e = np.multiply(d, noise1)
    f = np.multiply(e, noise2[::-1])
    g = np.multiply(f, noise1)
    profiles = np.concatenate([a, b, c, d, e, f, g], axis=0)
    coeff_tof = np.array([0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0], dtype=int)
    coeff_dv = np.array(coeff_tof[1:-1])
    tags = np.array(
        [
            "Struc2",
            "TS1",
            "Struc3",
            "TS2",
            "Struc4",
            "TS3",
            "Struc5",
            "TS4",
            "Struc6-NM",
        ]
    )
    find_dv(profiles, tags, coeff_dv, lnsteps, verb=3)
    profile = np.append(np.append(0, profiles[0, :]), dgr)
    tof0 = np.log10(calc_tof(profile, dgr, 298.15, coeff_tof, exact=True, verb=3)[0])
    tof1 = np.log10(calc_tof(profile, dgr, 298.15, coeff_tof, exact=False, verb=3)[0])
    print(
        f"Profile {profile} corresponds with log10(TOF) of {tof0} or approximately {tof1}"
    )

#!/usr/bin/env python

import numpy as np
import scipy.stats as stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model
from helpers import bround
from tof import calc_tof, calc_es, calc_s_es
from exceptions import MissingDataError


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


def plot_2d_lsfer(
    idx, d, tags, coeff, cb="white", ms="o", lmargin=10, rmargin=10, npoints=250, verb=0
):
    d_refill = np.zeros_like(d)
    d_refill[~np.isnan(d)] = d[~np.isnan(d)]
    tags = [str(tag) for tag in tags]
    lnsteps = range(d.shape[1])
    Xf = d[:, idx].reshape(-1)
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
        xmax = bround(X.max() + rmargin)
        xmin = bround(X.min() - lmargin)
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
                raise MissingDataError(
                    "Both descriptor and regression target are undefined. This should have been fixed before this point. Exiting."
                )
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

    ax.plot(x, y, "-", linewidth=1.25, color="midnightblue", alpha=0.95)
    for i in range(len(px)):
        ax.scatter(
            px[i],
            py[i],
            s=7.5,
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
        yavg = (y.max() + y.min()) * 0.5
        for i, j in zip(rid, avgs):
            plt.text(
                j,
                yavg,
                i,
                fontsize=7.5,
                horizontalalignment="center",
                verticalalignment="center",
                rotation="vertical",
            )
    plt.savefig(filename)


def plot_2d_es_volcano(
    idx,
    d,
    tags,
    coeff,
    dgr,
    cb="white",
    ms="o",
    lmargin=35,
    rmargin=35,
    npoints=250,
    verb=0,
):
    tags = [str(tag) for tag in tags]
    lnsteps = range(d.shape[1])
    X = d[:, idx].reshape(-1)
    xmax = bround(X.max() + rmargin)
    xmin = bround(X.min() - lmargin)
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
    slope = 0
    prevslope = 0
    prev = 0
    for i in range(ymin.shape[0]):
        profile = dgs[i, :-1]
        dgr = dgs[i][-1]
        ymin[i], ridmax[i], ridmin[i], diff = calc_es(profile, dgr, esp=True)
        idchange = [ridmax[i] != ridmax[i - 1], ridmin[i] != ridmin[i - 1]]
        slope = ymin[i] - prev
        prev = ymin[i]
        numchange = [np.abs(diff) > 1e-2, ~np.isclose(slope, prevslope, 1)]
        if any(idchange) and any(numchange):
            rid.append(f"{tags[ridmin[i]]} ➜ {tags[ridmax[i]]}")
            rb.append(xint[i])
            prevslope = slope
        else:
            ridmax[i] = ridmax[i - 1]
            ridmin[i] = ridmin[i - 1]
    if verb > 0:
        print(f"Identified {len(rid)} different determining states.")
        for i, j in zip(rid, rb):
            print(f"{i} starting at {j}")
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        profile = d[i, :-1]
        dgr = dgs[i][-1]
        px[i] = d[i, idx].reshape(-1)
        py[i] = calc_es(profile, dgr, esp=True)[0]
        if verb > 1:
            pointsname = f"points_es_volcano_{tags[idx]}.csv"
            zdata = list(zip(px, py))
            np.savetxt(
                pointsname,
                zdata,
                fmt="%.4e",
                delimiter=",",
                header="Descriptor, -\d_Ges",
            )
        if verb > 2:
            print(f"Profile {profile} corresponds with ES of {py[i]}")
    xlabel = f"{tags[idx]} [kcal/mol]"
    ylabel = r"-δ$G_{SPAN}$ [kcal/mol]"
    filename = f"es_volcano_{tags[idx]}.png"
    if verb > 0:
        csvname = f"es_volcano_{tags[idx]}.csv"
        print(f"Saving volcano data to file {csvname}")
        zdata = list(zip(xint, ymin))
        np.savetxt(
            csvname, zdata, fmt="%.4e", delimiter=",", header="Descriptor, -\d_Ges"
        )
    plot_2d(
        xint, ymin, px, py, xmin, xmax, xlabel, ylabel, filename, rid, rb, cb=cb, ms=ms
    )
    return xint, ymin, px, py, xmin, xmax, rid, rb


def plot_2d_k_volcano(
    idx,
    d,
    tags,
    coeff,
    dgr,
    cb="white",
    ms="o",
    lmargin=35,
    rmargin=35,
    npoints=250,
    verb=0,
):
    tags = np.array([str(tag) for tag in tags])
    tag = tags[idx]
    lnsteps = range(d.shape[1])
    X = d[:, idx].reshape(-1)
    xmax = bround(X.max() + rmargin)
    xmin = bround(X.min() - lmargin)
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
    slope = 0
    prevslope = 0
    prev = 0
    for i in range(ymin.shape[0]):
        profile = dgs[i, :-1]
        dgr = dgs[i][-1]
        ymin[i], ridmax[i], ridmin[i], diff = calc_s_es(profile, dgr, esp=True)
        idchange = [ridmax[i] != ridmax[i - 1], ridmin[i] != ridmin[i - 1]]
        slope = ymin[i] - prev
        prev = ymin[i]
        numchange = [np.abs(diff) > 1e-2, ~np.isclose(slope, prevslope, 1)]
        if any(idchange) and any(numchange):
            rid.append(f"{tags[ridmin[i]]} ➜ {tags[ridmax[i]]}")
            rb.append(xint[i])
            prevslope = slope
        else:
            ridmax[i] = ridmax[i - 1]
            ridmin[i] = ridmin[i - 1]
    if verb > 0:
        print(f"Identified {len(rid)} different determining states.")
        for i, j in zip(rid, rb):
            print(f"{i} starting at {j}")
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        profile = d[i, :-1]
        dgr = dgs[i][-1]
        px[i] = X[i].reshape(-1)
        py[i] = calc_s_es(profile, dgr, esp=True)[0]
        if verb > 1:
            pointsname = f"points_k_volcano_{tags[idx]}.csv"
            zdata = list(zip(px, py))
            np.savetxt(
                pointsname,
                zdata,
                fmt="%.4e",
                delimiter=",",
                header="Descriptor, -\d_Gkds",
            )
    xlabel = f"{tag} [kcal/mol]"
    ylabel = "-ΔG(kds) [kcal/mol]"
    filename = f"k_volcano_{tag}.png"
    if verb > 0:
        csvname = f"k_volcano_{tag}.csv"
        print(f"Saving volcano data to file {csvname}")
        zdata = list(zip(xint, ymin))
        np.savetxt(
            csvname, zdata, fmt="%.4e", delimiter=",", header="Descriptor, -\D_Gkds"
        )
    plot_2d(
        xint, ymin, px, py, xmin, xmax, xlabel, ylabel, filename, rid, rb, cb=cb, ms=ms
    )
    return xint, ymin, px, py, xmin, xmax, rid, rb


def plot_2d_t_volcano(
    idx,
    d,
    tags,
    coeff,
    dgr,
    cb="white",
    ms="o",
    lmargin=35,
    rmargin=35,
    npoints=250,
    verb=0,
):
    tags = np.array([str(tag) for tag in tags])
    tag = tags[idx]
    tags = tags[~coeff]
    lnsteps = range(np.count_nonzero(coeff == 0))
    X = d[:, idx].reshape(-1)
    xmax = bround(X.max() + rmargin)
    xmin = bround(X.min() - lmargin)
    if verb > 1:
        print(f"Range of descriptor set to [ {xmin} , {xmax} ]")
    xint = np.linspace(xmin, xmax, npoints)
    dgs = np.zeros((npoints, len(lnsteps)))
    d = d[:, ~coeff]
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
    slope = 0
    prevslope = 0
    prev = 0
    for i in range(ymin.shape[0]):
        profile = dgs[i, :-1]
        dgr = dgs[i][-1]
        ymin[i], ridmax[i], ridmin[i], diff = calc_s_es(profile, dgr, esp=True)
        idchange = [ridmax[i] != ridmax[i - 1], ridmin[i] != ridmin[i - 1]]
        slope = ymin[i] - prev
        prev = ymin[i]
        numchange = [np.abs(diff) > 1e-2, ~np.isclose(slope, prevslope, 1)]
        if any(idchange) and any(numchange):
            rid.append(f"{tags[ridmin[i]]} ➜ {tags[ridmax[i]]}")
            rb.append(xint[i])
            prevslope = slope
        else:
            ridmax[i] = ridmax[i - 1]
            ridmin[i] = ridmin[i - 1]
    if verb > 0:
        print(f"Identified {len(rid)} different determining states.")
        for i, j in zip(rid, rb):
            print(f"{i} starting at {j}")
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        profile = d[i, :-1]
        dgr = dgs[i][-1]
        px[i] = X[i].reshape(-1)
        py[i] = calc_s_es(profile, dgr, esp=True)[0]
        if verb > 1:
            pointsname = f"points_t_volcano_{tags[idx]}.csv"
            zdata = list(zip(px, py))
            np.savetxt(
                pointsname,
                zdata,
                fmt="%.4e",
                delimiter=",",
                header="Descriptor, -\d_Gpds",
            )
    xlabel = f"{tag} [kcal/mol]"
    ylabel = "-ΔG(pds) [kcal/mol]"
    filename = f"t_volcano_{tag}.png"
    if verb > 0:
        csvname = f"t_volcano_{tag}.csv"
        print(f"Saving volcano data to file {csvname}")
        zdata = list(zip(xint, ymin))
        np.savetxt(
            csvname, zdata, fmt="%.4e", delimiter=",", header="Descriptor, -\D_Gpds"
        )
    plot_2d(
        xint, ymin, px, py, xmin, xmax, xlabel, ylabel, filename, rid, rb, cb=cb, ms=ms
    )
    return xint, ymin, px, py, xmin, xmax, rid, rb


def plot_2d_tof_volcano(
    idx,
    d,
    tags,
    coeff,
    dgr,
    T=298.15,
    cb="white",
    ms="o",
    lmargin=15,
    rmargin=15,
    npoints=250,
    verb=0,
):
    tags = [str(tag) for tag in tags]
    lnsteps = range(d.shape[1])
    X = d[:, idx].reshape(-1)
    xmax = bround(X.max() + rmargin)
    xmin = bround(X.min() - lmargin)
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
        dgr = dgs[i][-1]
        tof = np.log10(calc_tof(profile, dgr, T, coeff, exact=True)[0])
        ytof[i] = tof
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        px[i] = d[i, idx].reshape(-1)
        profile = d[i, :]
        dgr = dgs[i][-1]
        tof = np.log10(calc_tof(profile, dgr, T, coeff, exact=True)[0])
        py[i] = tof
        if verb > 2:
            print(f"Profile {profile} corresponds with log10(TOF) of {tof}")
        if verb > 1:
            pointsname = f"points_tof_volcano_{tags[idx]}.csv"
            zdata = list(zip(px, py))
            np.savetxt(
                pointsname,
                zdata,
                fmt="%.4e",
                delimiter=",",
                header="Descriptor, log10(TOF)",
            )
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

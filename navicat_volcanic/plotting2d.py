#!/usr/bin/env python

import matplotlib
import numpy as np
import scipy.stats as stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model

from navicat_volcanic.exceptions import MissingDataError
from navicat_volcanic.helpers import bround
from navicat_volcanic.tof import calc_es, calc_s_es, calc_tof, calc_atof


def get_reg_targets(idx, d, tags, coeff, regress, mode="k"):
    """Separate regression targets and regressor variables."""
    tag = tags[idx]
    tags = tags[regress]
    X = d[:, idx].reshape(-1)
    d1 = d[:, regress]
    d2 = d[:, ~regress]
    coeff = coeff[regress]
    if mode == "t":
        d1 = d1[:, ~coeff]
        tags = tags[~coeff]
    return X, tag, tags, d1, d2, coeff


def calc_ci(resid, n, dof, x, x2, y2):
    t = stats.t.ppf(0.95, dof)
    s_err = np.sqrt(np.sum(resid**2) / dof)

    ci = (
        t
        * s_err
        * np.sqrt(1 / n + (x2 - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    )

    return ci


def plot_ci(ci, x2, y2, ax=None):
    if ax is None:
        try:
            ax = plt.gca()
        except Exception as m:
            return

    ax.fill_between(x2, y2 + ci, y2 - ci, color="#b9cfe7", alpha=0.6)

    return ax


def plot_2d_lsfer(
    idx,
    d,
    tags,
    coeff,
    regress,
    cb="white",
    ms="o",
    lmargin=10,
    rmargin=10,
    npoints=250,
    plotmode=1,
    verb=0,
):
    xbase = 20
    ybase = 10
    valid_d = np.copy(regress)
    for i, r in enumerate(regress):
        if np.isclose(d[:, i].std(), 0):
            valid_d[i] = False
            if verb > 2:
                print(f"Energy constant with mean {d[:,i].mean()}, will not regress.")
    Xf, tag, tags, d, d2, coeff = get_reg_targets(
        idx, d, tags, coeff, valid_d, mode="k"
    )
    lnsteps = range(d.shape[1])
    d_refill = np.zeros_like(d)
    d_refill[~np.isnan(d)] = d[~np.isnan(d)]
    mape = 100
    for j in lnsteps[1:]:
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
        xmax = bround(X.max() + rmargin, xbase)
        xmin = bround(X.min() - lmargin, xbase)
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
        dof = n - p.size
        resid = Y - Y_pred
        fig, ax = plt.subplots(
            frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
        )
        yint = np.polyval(p, xint)
        ci = calc_ci(resid, n, dof, X, xint, yint)
        plot_ci(ci, xint, yint, ax=ax)
        ax.plot(xint, yint, "-", linewidth=1, color="#000a75", alpha=0.85, zorder=1)
        plotpoints(ax, X, Y, cbi, msi, plotmode=1)
        beautify_ax(ax)
        # Labels and key
        plt.xlabel(f"{tag} [kcal/mol]")
        plt.ylabel(f"{tags[j]} [kcal/mol]")
        plt.xlim(xmin, xmax)

        ymin, ymax = ax.get_ylim()
        ymax = bround(ymax, ybase, type="max")
        ymin = bround(ymin, ybase, type="min")
        plt.ylim(ymin, ymax)
        plt.yticks(np.arange(ymin, ymax + 0.1, ybase))
        plt.xticks(np.arange(xmin, xmax + 0.1, xbase))
        plt.savefig(f"{tags[j]}.png")
        if verb > 0:
            csvname = f"{tags[j]}.csv"
            print(f"Saving volcano data to file {csvname}")
            zdata = list(zip(xint, yint, ci))
            np.savetxt(
                csvname,
                zdata,
                fmt="%.4e",
                delimiter=",",
                header=f"Descriptor, {tags[j]}, 95%CI",
            )
    return np.hstack((d_refill, d2))


def beautify_ax(ax):
    # Border
    ax.spines["top"].set_color("black")
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["right"].set_color("black")
    ax.get_xaxis().set_tick_params(direction="out")
    ax.get_yaxis().set_tick_params(direction="out")
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()
    return ax


def plotpoints(ax, px, py, cb, ms, plotmode):
    if plotmode == 1:
        s = 30
        lw = 0.3
    else:
        s = 15
        lw = 0.25
    for i in range(len(px)):
        ax.scatter(
            px[i],
            py[i],
            s=s,
            c=cb[i],
            marker=ms[i],
            linewidths=lw,
            edgecolors="black",
            zorder=2,
        )


def plot_2d(
    x,
    y,
    px,
    py,
    ci=None,
    xmin=0,
    xmax=100,
    xbase=20,
    ybase=10,
    xlabel="X-axis",
    ylabel="Y-axis",
    filename="plot.png",
    rid=None,
    rb=None,
    cb="white",
    ms="o",
    plotmode=1,
):
    fig, ax = plt.subplots(
        frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
    )
    # Labels and key
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xmin, xmax)
    plt.xticks(np.arange(xmin, xmax + 0.1, xbase))
    if plotmode == 0:
        ax.plot(x, y, "-", linewidth=1.5, color="midnightblue", alpha=0.95)
        ax = beautify_ax(ax)
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
                    linewidth=0.75,
                    alpha=0.75,
                )
    elif plotmode == 1:
        ax.plot(x, y, "-", linewidth=1.5, color="midnightblue", alpha=0.95, zorder=1)
        ax = beautify_ax(ax)
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
                    linewidth=0.75,
                    alpha=0.75,
                    zorder=3,
                )
        plotpoints(ax, px, py, cb, ms, plotmode)
    elif plotmode == 2:
        ax.plot(x, y, "-", linewidth=1.5, color="midnightblue", alpha=0.95, zorder=1)
        ax = beautify_ax(ax)
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
                    zorder=3,
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
                    zorder=4,
                )
        plotpoints(ax, px, py, cb, ms, plotmode)
    elif plotmode == 3:
        ax.plot(x, y, "-", linewidth=1.5, color="midnightblue", alpha=0.95, zorder=1)
        ax = beautify_ax(ax)
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
                    linewidth=0.75,
                    alpha=0.75,
                    zorder=3,
                )
        plotpoints(ax, px, py, cb, ms, plotmode)
        if ci is not None:
            plot_ci(ci, x, y, ax=ax)
    ymin, ymax = ax.get_ylim()
    ymax = bround(ymax, ybase, type="max")
    ymin = bround(ymin, ybase, type="min")
    plt.ylim(ymin, ymax)
    plt.yticks(np.arange(ymin, ymax + 0.1, ybase))
    plt.savefig(filename)


def plot_2d_es_volcano(
    idx,
    d,
    tags,
    coeff,
    regress,
    dgr,
    cb="white",
    ms="o",
    lmargin=35,
    rmargin=35,
    npoints=250,
    plotmode=1,
    verb=0,
):
    xbase = 20
    ybase = 10
    X, tag, tags, d, d2, coeff = get_reg_targets(idx, d, tags, coeff, regress, mode="k")
    lnsteps = range(d.shape[1])
    xmax = bround(X.max() + rmargin, xbase)
    xmin = bround(X.min() - lmargin, xbase)
    if verb > 1:
        print(f"Range of descriptor set to [ {xmin} , {xmax} ]")
    xint = np.linspace(xmin, xmax, npoints)
    dgs = np.zeros((npoints, len(lnsteps)))
    sigma_dgs = np.zeros((npoints, len(lnsteps)))
    for i, j in enumerate(lnsteps):
        Y = d[:, j].reshape(-1)
        if np.isclose(Y.std(), 0):
            if verb > 4:
                print(
                    f"State energy is constant at {i},{j} with mean {Y.mean()}. Setting to constant with zero uncertainty."
                )
            dgs[:, i] = Y.mean()
            sigma_dgs[:, i] = 0
            continue
        p, cov = np.polyfit(X, Y, 1, cov=True)  # 1 -> degree of polynomial
        Y_pred = np.polyval(p, X)
        n = Y.size
        m = p.size
        dof = n - m
        resid = Y - Y_pred
        with np.errstate(invalid="ignore"):
            chi2 = np.sum((resid / Y_pred) ** 2)
        yint = np.polyval(p, xint)
        ci = calc_ci(resid, n, dof, X, xint, yint)
        dgs[:, i] = yint
        sigma_dgs[:, i] = ci
    ymin = np.zeros_like(yint)
    ci = np.zeros_like(yint)
    ridmax = np.zeros_like(yint, dtype=int)
    ridmin = np.zeros_like(yint, dtype=int)
    rid = []
    rb = []
    slope = 0
    prevslope = 0
    prev = 0
    for i in range(ymin.shape[0]):
        profile = dgs[i, :-1]
        sigmas = sigma_dgs[i]
        dgr_s = dgs[i][-1]
        ymin[i], ridmax[i], ridmin[i], diff = calc_es(profile, dgr_s, esp=True)
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
        ci[i] = sigmas[ridmin[i]] + sigmas[ridmax[i]]
    if verb > 0:
        print(f"Identified {len(rid)} different determining states.")
        for i, j in zip(rid, rb):
            print(f"{i} starting at {j}")
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        profile = d[i, :-1]
        dgr_s = dgr[i]
        px[i] = X[i].reshape(-1)
        py[i] = calc_es(profile, dgr_s, esp=True)[0]
        if verb > 1:
            pointsname = f"points_es_volcano_{tag}.csv"
            zdata = list(zip(px, py))
            np.savetxt(
                pointsname,
                zdata,
                fmt="%.4e",
                delimiter=",",
                header="Descriptor, -\d_Ges",
            )
        if verb > 2:
            print(
                f"Profile {profile} with reaction energy {dgr_s} corresponds with ES of {py[i]}"
            )
    xlabel = f"{tag} [kcal/mol]"
    ylabel = r"-δ$E$ [kcal/mol]"
    filename = f"es_volcano_{tag}.png"
    if verb > 0:
        csvname = f"es_volcano_{tag}.csv"
        print(f"Saving volcano data to file {csvname}")
        zdata = list(zip(xint, ymin, ci))
        np.savetxt(
            csvname,
            zdata,
            fmt="%.4e",
            delimiter=",",
            header="Descriptor, -\d_Ges, 95%CI",
        )
    plot_2d(
        xint,
        ymin,
        px,
        py,
        ci,
        xmin,
        xmax,
        xbase,
        ybase,
        xlabel,
        ylabel,
        filename,
        rid,
        rb,
        cb=cb,
        ms=ms,
        plotmode=plotmode,
    )
    return xint, ymin, px, py, xmin, xmax, rid, rb


def plot_2d_k_volcano(
    idx,
    d,
    tags,
    coeff,
    regress,
    dgr,
    cb="white",
    ms="o",
    lmargin=35,
    rmargin=35,
    npoints=250,
    plotmode=1,
    verb=0,
):
    xbase = 20
    ybase = 10
    X, tag, tags, d, d2, coeff = get_reg_targets(idx, d, tags, coeff, regress, mode="k")
    lnsteps = range(d.shape[1])
    xmax = bround(X.max() + rmargin, xbase)
    xmin = bround(X.min() - lmargin, xbase)
    if verb > 1:
        print(f"Range of descriptor set to [ {xmin} , {xmax} ]")
    xint = np.linspace(xmin, xmax, npoints)
    dgs = np.zeros((npoints, len(lnsteps)))
    sigma_dgs = np.zeros((npoints, len(lnsteps)))
    for i, j in enumerate(lnsteps):
        Y = d[:, j].reshape(-1)
        if np.isclose(Y.std(), 0):
            if verb > 4:
                print(
                    f"State energy is constant at {i},{j} with mean {Y.mean()}. Setting to constant with zero uncertainty."
                )
            dgs[:, i] = Y.mean()
            sigma_dgs[:, i] = 0
            continue
        p, cov = np.polyfit(X, Y, 1, cov=True)  # 1 -> degree of polynomial
        Y_pred = np.polyval(p, X)
        n = Y.size
        m = p.size
        dof = n - m
        resid = Y - Y_pred
        with np.errstate(invalid="ignore"):
            chi2 = np.sum((resid / Y_pred) ** 2)
        yint = np.polyval(p, xint)
        ci = calc_ci(resid, n, dof, X, xint, yint)
        dgs[:, i] = yint
        sigma_dgs[:, i] = ci
    ymin = np.zeros_like(yint)
    ci = np.zeros_like(yint)
    ridmax = np.zeros_like(yint, dtype=int)
    ridmin = np.zeros_like(yint, dtype=int)
    rid = []
    rb = []
    slope = 0
    prevslope = 0
    prev = 0
    for i in range(ymin.shape[0]):
        profile = dgs[i, :-1]
        sigmas = sigma_dgs[i]
        dgr_s = dgs[i][-1]
        ymin[i], ridmax[i], ridmin[i], diff = calc_s_es(profile, dgr_s, esp=True)
        ci[i] = sigmas[ridmin[i]] + sigmas[ridmax[i]]
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
        dgr_s = dgr[i]
        px[i] = X[i].reshape(-1)
        py[i] = calc_s_es(profile, dgr_s, esp=True)[0]
        if verb > 1:
            pointsname = f"points_k_volcano_{tag}.csv"
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
        zdata = list(zip(xint, ymin, ci))
        np.savetxt(
            csvname,
            zdata,
            fmt="%.4e",
            delimiter=",",
            header="Descriptor, -\d_Gkds, 95%CI",
        )
    plot_2d(
        xint,
        ymin,
        px,
        py,
        ci,
        xmin,
        xmax,
        xbase,
        ybase,
        xlabel,
        ylabel,
        filename,
        rid,
        rb,
        cb=cb,
        ms=ms,
        plotmode=plotmode,
    )
    return xint, ymin, px, py, xmin, xmax, rid, rb


def plot_2d_t_volcano(
    idx,
    d,
    tags,
    coeff,
    regress,
    dgr,
    cb="white",
    ms="o",
    lmargin=35,
    rmargin=35,
    npoints=250,
    plotmode=1,
    verb=0,
):
    xbase = 20
    ybase = 10
    X, tag, tags, d, d2, coeff = get_reg_targets(idx, d, tags, coeff, regress, mode="t")
    lnsteps = range(d.shape[1])
    xmax = bround(X.max() + rmargin, xbase)
    xmin = bround(X.min() - lmargin, xbase)
    if verb > 1:
        print(f"Range of descriptor set to [ {xmin} , {xmax} ]")
    xint = np.linspace(xmin, xmax, npoints)
    dgs = np.zeros((npoints, len(lnsteps)))
    sigma_dgs = np.zeros((npoints, len(lnsteps)))
    for i, j in enumerate(lnsteps):
        Y = d[:, j].reshape(-1)
        if np.isclose(Y.std(), 0):
            if verb > 4:
                print(
                    f"State energy is constant at {i},{j} with mean {Y.mean()}. Setting to constant with zero uncertainty."
                )
            dgs[:, i] = Y.mean()
            sigma_dgs[:, i] = 0
            continue
        p, cov = np.polyfit(X, Y, 1, cov=True)  # 1 -> degree of polynomial
        Y_pred = np.polyval(p, X)
        n = Y.size
        m = p.size
        dof = n - m
        resid = Y - Y_pred
        with np.errstate(invalid="ignore"):
            chi2 = np.sum((resid / Y_pred) ** 2)
        yint = np.polyval(p, xint)
        ci = calc_ci(resid, n, dof, X, xint, yint)
        dgs[:, i] = yint
        sigma_dgs[:, i] = ci
    ymin = np.zeros_like(yint)
    ci = np.zeros_like(yint)
    ridmax = np.zeros_like(yint, dtype=int)
    ridmin = np.zeros_like(yint, dtype=int)
    rid = []
    rb = []
    slope = 0
    prevslope = 0
    prev = 0
    for i in range(ymin.shape[0]):
        profile = dgs[i, :-1]
        sigmas = sigma_dgs[i]
        dgr_s = dgs[i][-1]
        ymin[i], ridmax[i], ridmin[i], diff = calc_s_es(profile, dgr_s, esp=True)
        ci[i] = sigmas[ridmin[i]] + sigmas[ridmax[i]]
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
        dgr_s = dgr[i]
        px[i] = X[i].reshape(-1)
        py[i] = calc_s_es(profile, dgr_s, esp=True)[0]
        if verb > 1:
            pointsname = f"points_t_volcano_{tag}.csv"
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
        zdata = list(zip(xint, ymin, ci))
        np.savetxt(
            csvname,
            zdata,
            fmt="%.4e",
            delimiter=",",
            header="Descriptor, -\d_Gpds, 95%CI",
        )
    plot_2d(
        xint,
        ymin,
        px,
        py,
        ci,
        xmin,
        xmax,
        xbase,
        ybase,
        xlabel,
        ylabel,
        filename,
        rid,
        rb,
        cb=cb,
        ms=ms,
        plotmode=plotmode,
    )
    return xint, ymin, px, py, xmin, xmax, rid, rb


def plot_2d_tof_volcano(
    idx,
    d,
    tags,
    coeff,
    regress,
    dgr,
    T=298.15,
    cb="white",
    ms="o",
    lmargin=15,
    rmargin=15,
    npoints=250,
    plotmode=1,
    verb=0,
):
    xbase = 20
    ybase = 5
    X, tag, tags, d, d2, coeff = get_reg_targets(idx, d, tags, coeff, regress, mode="k")
    lnsteps = range(d.shape[1])
    xmax = bround(X.max() + rmargin, xbase)
    xmin = bround(X.min() - lmargin, xbase)
    if verb > 1:
        print(f"Range of descriptor set to [ {xmin} , {xmax} ]")
    xint = np.linspace(xmin, xmax, npoints)
    dgs = np.zeros((npoints, len(lnsteps)))
    sigma_dgs = np.zeros((npoints, len(lnsteps)))
    for i, j in enumerate(lnsteps):
        Y = d[:, j].reshape(-1)
        if np.isclose(Y.std(), 0):
            if verb > 4:
                print(
                    f"State energy is constant at {i},{j} with mean {Y.mean()}. Setting to constant with zero uncertainty."
                )
            dgs[:, i] = Y.mean()
            sigma_dgs[:, i] = 0
            continue
        p, cov = np.polyfit(X, Y, 1, cov=True)
        Y_pred = np.polyval(p, X)
        n = Y.size
        m = p.size
        dof = n - m
        resid = Y - Y_pred
        with np.errstate(invalid="ignore"):
            chi2 = np.sum((resid / Y_pred) ** 2)
        yint = np.polyval(p, xint)
        ci = calc_ci(resid, n, dof, X, xint, yint)
        dgs[:, i] = yint
        sigma_dgs[:, i] = ci
    ytof = np.zeros_like(yint)
    ci = np.zeros_like(yint)
    ridmax = np.zeros_like(yint, dtype=int)
    ridmin = np.zeros_like(yint, dtype=int)
    slope = 0
    prevslope = 0
    prev = 0
    # We must take the initial and ending states into account here
    for i in range(ytof.shape[0]):
        profile = dgs[i, :]
        sigmas = sigma_dgs[i]
        if verb > 5:
            print(
                f"95% CI uncertainties for profile {np.round(profile,2)} are {np.round(sigmas,2)}."
            )
        dgr_s = dgs[i][-1]
        tof, xtof, e = calc_tof(profile, dgr_s, T, coeff, exact=True)
        es, ridmax[i], ridmin[i], diff = calc_es(profile, dgr_s, esp=True)
        idchange = [ridmax[i] != ridmax[i - 1], ridmin[i] != ridmin[i - 1]]
        slope = np.log10(tof) - prev
        numchange = [np.abs(diff) > 1e-2, ~np.isclose(slope, prevslope, 1)]
        if any(idchange) and any(numchange):
            prevslope = slope
        else:
            ridmax[i] = ridmax[i - 1]
            ridmin[i] = ridmin[i - 1]
        sigma_d = sigmas[ridmin[i]] + sigmas[ridmax[i]]
        sigma_tof_p = calc_atof(es + sigma_d, dgr_s, T)
        sigma_tof_m = calc_atof(es - sigma_d, dgr_s, T)
        sigma_ltof_p = np.log10(sigma_tof_p)
        sigma_ltof_m = np.log10(sigma_tof_m)
        sigma_ltof = (sigma_ltof_m - sigma_ltof_p) / 2
        ltof = np.log10(tof)
        prev = ltof
        if verb > 4:
            print(
                f"Simulated profile {np.round(profile,2)} with reaction energy {np.round(dgr_s,2)} corresponds with log10(TOF) of {np.round(ltof,2)}"
            )
        if verb > 6:
            print(f"Uncertainty is {np.round(sigma_ltof,2)}")
        ci[i] = sigma_ltof
        ytof[i] = ltof
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        px[i] = X[i].reshape(-1)
        profile = d[i, :]
        dgr_s = dgr[i]
        tof = calc_tof(profile, dgr_s, T, coeff, exact=True)[0]
        ltof = np.log10(tof)
        py[i] = ltof
        if verb > 2:
            print(
                f"Profile {profile} with reaction energy {dgr_s} corresponds with log10(TOF) of {np.round(ltof,2)}"
            )
        if verb > 1:
            pointsname = f"points_tof_volcano_{tag}.csv"
            zdata = list(zip(px, py))
            np.savetxt(
                pointsname,
                zdata,
                fmt="%.4e",
                delimiter=",",
                header="Descriptor, log10(TOF)",
            )
    xlabel = f"{tag} [kcal/mol]"
    ylabel = "log(TOF) [1/s]"
    filename = f"tof_volcano_{tag}.png"
    if verb > 0:
        csvname = f"tof_volcano_{tag}.csv"
        print(f"Saving TOF volcano data to file {csvname}")
        zdata = list(zip(xint, ytof, ci))
        np.savetxt(
            csvname,
            zdata,
            fmt="%.4e",
            delimiter=",",
            header="Descriptor, log10(TOF), 95%CI",
        )
    plot_2d(
        xint,
        ytof,
        px,
        py,
        ci,
        xmin,
        xmax,
        xbase,
        ybase,
        xlabel,
        ylabel,
        filename,
        cb=cb,
        ms=ms,
        plotmode=plotmode,
    )
    return xint, ytof, px, py, xmin, xmax

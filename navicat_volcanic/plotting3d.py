#!/usr/bin/env python

import itertools

import matplotlib
import numpy as np
import scipy.stats as stats
from matplotlib import cm
from matplotlib.ticker import FuncFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.linear_model

from navicat_volcanic.exceptions import MissingDataError
from navicat_volcanic.helpers import bround
from navicat_volcanic.tof import calc_es, calc_s_es, calc_tof


def get_reg_targets(idx1, idx2, d, tags, coeff, regress, mode="k"):
    """Separate regression targets and regressor variables."""
    tag1 = tags[idx1]
    tag2 = tags[idx2]
    tags = tags[regress]
    X1 = d[:, idx1].reshape(-1)
    X2 = d[:, idx2].reshape(-1)
    d1 = d[:, regress]
    d2 = d[:, ~regress]
    coeff = coeff[regress]
    if mode == "t":
        d1 = d1[:, ~coeff]
        tags = tags[~coeff]
    return X1, X2, tag1, tag2, tags, d1, d2, coeff


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


def get_bases(X1, X2):
    return get_base(X1), get_base(X2)


def get_base(X):
    s = np.abs(X.max() - X.min())
    om = np.round(np.log10(s))
    xbase = max([(10**om) * 2, 0.1])
    if s < 3 * xbase:
        r = np.round(xbase / s)
        xbase = xbase * 3 / r
    if s > 7 * xbase:
        r = np.round(s / xbase)
        xbase = r / 3 * xbase
    if xbase > 1:
        xbase = int(xbase)
    else:
        xbase = max([xbase, 0.1])
    return xbase


def plot_3d_lsfer(
    idx1,
    idx2,
    d,
    tags,
    coeff,
    regress,
    cb="white",
    ms="o",
    lmargin=5,
    rmargin=5,
    npoints=100,
    plotmode=1,
    verb=0,
):
    valid_d = np.copy(regress)
    if np.isclose(d[:, -1].std(), 0):
        valid_d[-1] = False
        if verb > 0:
            print(f"\nReaction energy is constant. Assuming substrates are constant.")
    X1, X2, tag1, tag2, tags, d, d2, coeff = get_reg_targets(
        idx1, idx2, d, tags, coeff, valid_d, mode="k"
    )
    x1base, x2base = get_bases(X1, X2)
    d_refill = np.zeros_like(d)
    d_refill[~np.isnan(d)] = d[~np.isnan(d)]
    lnsteps = range(d.shape[1])
    mape = 100
    for j in lnsteps[1:]:
        if verb > 0:
            print(f"Plotting regression of {tags[j]}.")
        XY = np.vstack([X1, X2, d[:, j]]).T
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
        Xm = XYm[:, :2]
        Ym = XYm[:, 2]
        X = XY[:, :2]
        Y = XY[:, 2]
        xmax = bround(Y.max() + rmargin, x1base, "max")
        xmin = bround(Y.min() - lmargin, x1base, "min")
        xint = np.sort(Y)
        reg = sk.linear_model.LinearRegression().fit(X, Y)
        if verb > 2:
            print(
                f"Linear model has coefficients : {reg.coef_} \n and intercept {reg.intercept_}"
            )
        Y_pred = reg.predict(X)
        p = reg.coef_
        currmape = sk.metrics.mean_absolute_percentage_error(Y, Y_pred)
        for k, y in enumerate(Ym):
            if not np.isnan(Xm[k, 0]) and not np.isnan(Xm[k, 1]) and np.isnan(Ym[k]):
                Ym[k] = reg.predict(Xm[k])
                d_refill[np.isnan(d).any(axis=1)][:, j][k] = Ym[k]
            elif not np.isnan(Ym[k]) and not np.isnan(Xm[k, 0]):
                if currmape < mape:
                    Xm[k, 1] = (
                        Ym[k] - reg.intercept_ - reg.coeff_[0] * X[k][0]
                    ) / reg.coeff_[1]
                    d_refill[np.isnan(d).any(axis=1)][:, idx2][k] = Xm[k, 1]
                    mape = currmape
            elif not np.isnan(Ym[k]) and not np.isnan(Xm[k, 1]):
                if currmape < mape:
                    Xm[k, 0] = (
                        Ym[k] - reg.intercept_ - reg.coeff_[1] * X[k][1]
                    ) / reg.coeff_[0]
                    d_refill[np.isnan(d).any(axis=1)][:, idx1][k] = Xm[k, 0]
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
        s_err = np.sqrt(np.sum(resid**2) / dof)
        fig, ax = plt.subplots(
            frameon=False, figsize=[3, 3], dpi=300, constrained_layout=True
        )
        yint = np.sort(Y_pred)
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
                Y_pred[i],
                Y[i],
                s=12.5,
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
        plt.xlabel(f"Function of {tag1} and {tag2}")
        plt.ylabel(f"{tags[j]} [kcal/mol]")
        plt.xlim(xmin, xmax)
        plt.savefig(f"{tags[j]}.png")
    return np.hstack((d_refill, d2))


def plot_3d_t_volcano(
    idx1,
    idx2,
    d,
    tags,
    coeff,
    regress,
    dgr,
    cb="white",
    ms="o",
    lmargin=15,
    rmargin=15,
    npoints=200,
    plotmode=1,
    verb=0,
):
    X1, X2, tag1, tag2, tags, d, d2, coeff = get_reg_targets(
        idx1, idx2, d, tags, coeff, regress, mode="t"
    )
    x1base, x2base = get_bases(X1, X2)
    lnsteps = range(d.shape[1])
    x1max = bround(X1.max() + rmargin, x1base, "max")
    x1min = bround(X1.min() - lmargin, x1base, "min")
    x2max = bround(X2.max() + rmargin, x2base, "max")
    x2min = bround(X2.min() - lmargin, x2base, "min")
    if verb > 1:
        print(
            f"Range of descriptors set to [ {x1min} , {x1max} ] and [ {x2min} , {x2max} ]"
        )
    xint = np.linspace(x1min, x1max, npoints)
    yint = np.linspace(x2min, x2max, npoints)
    grids = []
    for i, j in enumerate(lnsteps):
        gridj = np.zeros((npoints, npoints))
        XY = np.vstack([X1, X2, d[:, j]]).T
        X = XY[:, :2]
        Y = XY[:, 2]
        if np.isclose(Y.std(), 0):
            if verb > 4:
                print(
                    f"State energy is constant at {i},{j} with mean {Y.mean()}. Setting to constant with zero uncertainty."
                )
            gridj.fill(Y.mean())
            grids.append(gridj)
            continue
        reg = sk.linear_model.LinearRegression().fit(X, Y)
        Y_pred = reg.predict(X)
        for k, x1 in enumerate(xint):
            for l, x2 in enumerate(yint):
                x1x2 = np.vstack([x1, x2]).reshape(1, -1)
                gridj[k, l] = reg.predict(x1x2)
        grids.append(gridj)
    grid = np.zeros_like(gridj)
    ridmax = np.zeros_like(gridj, dtype=int)
    ridmin = np.zeros_like(gridj, dtype=int)
    rb = np.zeros_like(gridj, dtype=int)
    for k, x1 in enumerate(xint):
        for l, x2 in enumerate(yint):
            profile = [gridj[k, l] for gridj in grids][:-1]
            dgr = [gridj[k, l] for gridj in grids][-1]
            grid[k, l], ridmax[k, l], ridmin[k, l], diff = calc_s_es(
                profile, dgr, esp=True
            )
    rid = np.hstack([ridmin, ridmax])
    if verb > 0:
        pass
    ymin = grid.min()
    ymax = grid.max()
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        profile = d[i, :-1]
        dgr = d[i][-1]
        px[i] = X1[i]
        py[i] = X2[i]
    x1label = f"{tag1} [kcal/mol]"
    x2label = f"{tag2} [kcal/mol]"
    ylabel = "-ΔG(pds) [kcal/mol]"
    filename = f"t_volcano_{tag1}_{tag2}.png"
    if verb > 0:
        csvname = f"t_volcano_{tag1}_{tag2}.csv"
        print(f"Saving volcano data to file {csvname}")
        x = np.zeros_like(grid.reshape(-1))
        y = np.zeros_like(grid.reshape(-1))
        for i, xy in enumerate(itertools.product(xint, yint)):
            x[i] = xy[0]
            y[i] = xy[1]
        zdata = list(zip(x, y, grid.reshape(-1)))
        np.savetxt(
            csvname,
            zdata,
            fmt="%.4e",
            delimiter=",",
            header="Descriptor 1, Descriptor 2, -\D_pds",
        )
    if plotmode == 2:
        plot_3d_contour(
            xint,
            yint,
            grid.T,
            px,
            py,
            ymin,
            ymax,
            x1min,
            x1max,
            x2min,
            x2max,
            x1base,
            x2base,
            x1label=x1label,
            x2label=x2label,
            ylabel=ylabel,
            filename=filename,
            cb=cb,
            ms=ms,
            plotmode=plotmode,
        )
    else:
        plot_3d_scatter(
            xint,
            yint,
            grid.T,
            px,
            py,
            ymin,
            ymax,
            x1min,
            x1max,
            x2min,
            x2max,
            x1base,
            x2base,
            x1label=x1label,
            x2label=x2label,
            ylabel=ylabel,
            filename=filename,
            cb=cb,
            ms=ms,
            plotmode=plotmode,
        )
    return xint, yint, grid, px, py


def plot_3d_k_volcano(
    idx1,
    idx2,
    d,
    tags,
    coeff,
    regress,
    dgr,
    cb="white",
    ms="o",
    lmargin=15,
    rmargin=15,
    npoints=200,
    plotmode=1,
    verb=0,
):
    X1, X2, tag1, tag2, tags, d, d2, coeff = get_reg_targets(
        idx1, idx2, d, tags, coeff, regress, mode="k"
    )
    x1base, x2base = get_bases(X1, X2)
    lnsteps = range(d.shape[1])
    x1max = bround(X1.max() + rmargin, x1base, "max")
    x1min = bround(X1.min() - lmargin, x1base, "min")
    x2max = bround(X2.max() + rmargin, x2base, "max")
    x2min = bround(X2.min() - lmargin, x2base, "min")
    if verb > 1:
        print(
            f"Range of descriptors set to [ {x1min} , {x1max} ] and [ {x2min} , {x2max} ]"
        )
    xint = np.linspace(x1min, x1max, npoints)
    yint = np.linspace(x2min, x2max, npoints)
    grids = []
    for i, j in enumerate(lnsteps):
        gridj = np.zeros((npoints, npoints))
        XY = np.vstack([X1, X2, d[:, j]]).T
        X = XY[:, :2]
        Y = XY[:, 2]
        if np.isclose(Y.std(), 0):
            if verb > 4:
                print(
                    f"State energy is constant at {i},{j} with mean {Y.mean()}. Setting to constant with zero uncertainty."
                )
            gridj.fill(Y.mean())
            grids.append(gridj)
            continue
        reg = sk.linear_model.LinearRegression().fit(X, Y)
        Y_pred = reg.predict(X)
        for k, x1 in enumerate(xint):
            for l, x2 in enumerate(yint):
                x1x2 = np.vstack([x1, x2]).reshape(1, -1)
                gridj[k, l] = reg.predict(x1x2)
        grids.append(gridj)
    grid = np.zeros_like(gridj)
    ridmax = np.zeros_like(gridj, dtype=int)
    ridmin = np.zeros_like(gridj, dtype=int)
    rb = np.zeros_like(gridj, dtype=int)
    for k, x1 in enumerate(xint):
        for l, x2 in enumerate(yint):
            profile = [gridj[k, l] for gridj in grids][:-1]
            dgr = [gridj[k, l] for gridj in grids][-1]
            grid[k, l], ridmax[k, l], ridmin[k, l], diff = calc_s_es(
                profile, dgr, esp=True
            )
    rid = np.hstack([ridmin, ridmax])
    if verb > 0:
        pass
    ymin = grid.min()
    ymax = grid.max()
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        profile = d[i, :-1]
        px[i] = X1[i]
        py[i] = X2[i]
    x1label = f"{tag1} [kcal/mol]"
    x2label = f"{tag2} [kcal/mol]"
    ylabel = "-ΔG(kds) [kcal/mol]"
    filename = f"k_volcano_{tag1}_{tag2}.png"
    if verb > 0:
        csvname = f"k_volcano_{tag1}_{tag2}.csv"
        print(f"Saving volcano data to file {csvname}")
        x = np.zeros_like(grid.reshape(-1))
        y = np.zeros_like(grid.reshape(-1))
        for i, xy in enumerate(itertools.product(xint, yint)):
            x[i] = xy[0]
            y[i] = xy[1]
        zdata = list(zip(x, y, grid.reshape(-1)))
        np.savetxt(
            csvname,
            zdata,
            fmt="%.4e",
            delimiter=",",
            header="Descriptor 1, Descriptor 2, -\D_kds",
        )
    if plotmode == 2:
        plot_3d_contour(
            xint,
            yint,
            grid.T,
            px,
            py,
            ymin,
            ymax,
            x1min,
            x1max,
            x2min,
            x2max,
            x1base,
            x2base,
            x1label=x1label,
            x2label=x2label,
            ylabel=ylabel,
            filename=filename,
            cb=cb,
            ms=ms,
            plotmode=plotmode,
        )
    else:
        plot_3d_scatter(
            xint,
            yint,
            grid.T,
            px,
            py,
            ymin,
            ymax,
            x1min,
            x1max,
            x2min,
            x2max,
            x1base,
            x2base,
            x1label=x1label,
            x2label=x2label,
            ylabel=ylabel,
            filename=filename,
            cb=cb,
            ms=ms,
            plotmode=plotmode,
        )
    return xint, yint, grid, px, py


def cantor_pair(a, b):
    return (a + b) * (a + b + 1) / 2 + b


def cantor_unpair(n):
    w = np.floor((np.sqrt(8 * n + 1) - 1) / 2)
    t = (pow(w, 2) + w) / 2
    b = int(n - t)
    a = int(w - b)
    return a, b


def plot_3d_es_volcano(
    idx1,
    idx2,
    d,
    tags,
    coeff,
    regress,
    dgr,
    cb="white",
    ms="o",
    lmargin=15,
    rmargin=15,
    npoints=200,
    plotmode=1,
    verb=0,
):
    plot_regions = True
    chemical_sense = True
    X1, X2, tag1, tag2, tags, d, d2, coeff = get_reg_targets(
        idx1, idx2, d, tags, coeff, regress, mode="k"
    )
    x1base, x2base = get_bases(X1, X2)
    lnsteps = range(d.shape[1])
    x1max = bround(X1.max() + rmargin, x1base, "max")
    x1min = bround(X1.min() - lmargin, x1base, "min")
    x2max = bround(X2.max() + rmargin, x2base, "max")
    x2min = bround(X2.min() - lmargin, x2base, "min")
    if verb > 1:
        print(
            f"Range of descriptors set to [ {x1min} , {x1max} ] and [ {x2min} , {x2max} ]"
        )
    xint = np.linspace(x1min, x1max, npoints)
    yint = np.linspace(x2min, x2max, npoints)
    grids = []
    for i, j in enumerate(lnsteps):
        gridj = np.zeros((npoints, npoints))
        XY = np.vstack([X1, X2, d[:, j]]).T
        X = XY[:, :2]
        Y = XY[:, 2]
        if np.isclose(Y.std(), 0):
            if verb > 4:
                print(
                    f"State energy is constant at {i},{j} with mean {Y.mean()}. Setting to constant with zero uncertainty."
                )
            gridj.fill(Y.mean())
            grids.append(gridj)
            continue
        reg = sk.linear_model.LinearRegression().fit(X, Y)
        Y_pred = reg.predict(X)
        for k, x1 in enumerate(xint):
            for l, x2 in enumerate(yint):
                x1x2 = np.vstack([x1, x2]).reshape(1, -1)
                gridj[k, l] = reg.predict(x1x2)
        grids.append(gridj)
    grid = np.zeros_like(gridj)
    ridmax = np.zeros_like(gridj, dtype=int)
    ridmin = np.zeros_like(gridj, dtype=int)
    rb = np.zeros_like(gridj, dtype=int)
    for k, x1 in enumerate(xint):
        for l, x2 in enumerate(yint):
            profile = [gridj[k, l] for gridj in grids][:-1]
            dgr = [gridj[k, l] for gridj in grids][-1]
            grid[k, l], ridmax[k, l], ridmin[k, l], diff = calc_es(
                profile, dgr, esp=True, chemical_sense=chemical_sense
            )
    if plot_regions:
        rid = np.array(cantor_pair(ridmin, ridmax), dtype=int)
        ridgrid = np.zeros_like(gridj, dtype=int)
        (unique, count) = np.unique(rid, return_counts=True)
        id_labels = []
        for k, (span, c) in enumerate(zip(unique, count)):
            pair = np.where(rid == span)
            idmin, idmax = cantor_unpair(span)
            id_labels.append(f"{tags[idmin]}\n->n{tags[idmax]}")
            if verb > 1:
                print(
                    f"Region ID {k} corresponding to {tags[idmin]}->{tags[idmax]} in {c} points."
                )
            ridgrid[np.where(rid == span)] = k
        if verb > 0:
            print(f"Found {len(unique)} distinct regions in the energy span.")

    ymin = grid.min()
    ymax = grid.max()
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        profile = d[i, :-1]
        px[i] = X1[i]
        py[i] = X2[i]
    x1label = f"{tag1} [kcal/mol]"
    x2label = f"{tag2} [kcal/mol]"
    ylabel = r"-δ$E$ [kcal/mol]"
    filename = f"es_volcano_{tag1}_{tag2}.png"
    if plot_regions:
        filename_regions = f"regions_es_volcano_{tag1}_{tag2}.png"
    if verb > 0:
        csvname = f"es_volcano_{tag1}_{tag2}.csv"
        if plot_regions:
            csvname_regions = f"regions_es_volcano_{tag1}_{tag2}.csv"
        print(f"Saving volcano data to file {csvname}")
        x = np.zeros_like(grid.reshape(-1))
        y = np.zeros_like(grid.reshape(-1))
        for i, xy in enumerate(itertools.product(xint, yint)):
            x[i] = xy[0]
            y[i] = xy[1]
        zdata = list(zip(x, y, grid.reshape(-1)))
        if plot_regions:
            zdata_regions = list(zip(x, y, ridgrid.reshape(-1)))
        np.savetxt(
            csvname,
            zdata,
            fmt="%.4e",
            delimiter=",",
            header="Descriptor 1, Descriptor 2, -\d_Ges",
        )
        if plot_regions:
            np.savetxt(
                csvname_regions,
                zdata_regions,
                fmt="%.4e",
                delimiter=",",
                header="Descriptor 1, Descriptor 2, RegionID",
            )
    if plot_regions:
        ymin_region = 0
        ymax_region = len(unique)
        plot_3d_contour_regions(
            xint,
            yint,
            ridgrid.T,
            px,
            py,
            ymin_region,
            ymax_region,
            x1min,
            x1max,
            x2min,
            x2max,
            x1base,
            x2base,
            x1label=x1label,
            x2label=x2label,
            ylabel="Region",
            filename=filename_regions,
            cb=cb,
            ms=ms,
            nunique=len(unique),
            id_labels=id_labels,
            plotmode=3,
        )
    if plotmode == 2:
        plot_3d_contour(
            xint,
            yint,
            grid.T,
            px,
            py,
            ymin,
            ymax,
            x1min,
            x1max,
            x2min,
            x2max,
            x1base,
            x2base,
            x1label=x1label,
            x2label=x2label,
            ylabel=ylabel,
            filename=filename,
            cb=cb,
            ms=ms,
            plotmode=plotmode,
        )
    else:
        plot_3d_scatter(
            xint,
            yint,
            grid.T,
            px,
            py,
            ymin,
            ymax,
            x1min,
            x1max,
            x2min,
            x2max,
            x1base,
            x2base,
            x1label=x1label,
            x2label=x2label,
            ylabel=ylabel,
            filename=filename,
            cb=cb,
            ms=ms,
            plotmode=plotmode,
        )
    return xint, yint, grid, px, py


def plot_3d_tof_volcano(
    idx1,
    idx2,
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
    npoints=200,
    plotmode=1,
    verb=0,
):
    X1, X2, tag1, tag2, tags, d, d2, coeff = get_reg_targets(
        idx1, idx2, d, tags, coeff, regress, mode="k"
    )
    x1base, x2base = get_bases(X1, X2)
    lnsteps = range(d.shape[1])
    x1max = bround(X1.max() + rmargin, x1base, "max")
    x1min = bround(X1.min() - lmargin, x1base, "min")
    x2max = bround(X2.max() + rmargin, x2base, "max")
    x2min = bround(X2.min() - lmargin, x2base, "min")
    if verb > 1:
        print(
            f"Range of descriptors set to [ {x1min} , {x1max} ] and [ {x2min} , {x2max} ]"
        )
    xint = np.linspace(x1min, x1max, npoints)
    yint = np.linspace(x2min, x2max, npoints)
    grids = []
    for i, j in enumerate(lnsteps):
        gridj = np.zeros((npoints, npoints))
        XY = np.vstack([X1, X2, d[:, j]]).T
        X = XY[:, :2]
        Y = XY[:, 2]
        if np.isclose(Y.std(), 0):
            if verb > 4:
                print(
                    f"State energy is constant at {i},{j} with mean {Y.mean()}. Setting to constant with zero uncertainty."
                )
            gridj.fill(Y.mean())
            grids.append(gridj)
            continue
        reg = sk.linear_model.LinearRegression().fit(X, Y)
        Y_pred = reg.predict(X)
        for k, x1 in enumerate(xint):
            for l, x2 in enumerate(yint):
                x1x2 = np.vstack([x1, x2]).reshape(1, -1)
                gridj[k, l] = reg.predict(x1x2)
        grids.append(gridj)
    grid = np.zeros_like(gridj)
    rb = np.zeros_like(gridj, dtype=int)
    for k, x1 in enumerate(xint):
        for l, x2 in enumerate(yint):
            profile = [gridj[k, l] for gridj in grids]
            dgr = [gridj[k, l] for gridj in grids][-1]
            grid[k, l] = np.log10(calc_tof(profile, dgr, T, coeff, exact=True)[0])
    ymin = grid.min()
    ymax = grid.max()
    px = np.zeros_like(d[:, 0])
    py = np.zeros_like(d[:, 0])
    for i in range(d.shape[0]):
        profile = d[i, :-1]
        px[i] = X1[i]
        py[i] = X2[i]
    x1label = f"{tag1} [kcal/mol]"
    x2label = f"{tag2} [kcal/mol]"
    ylabel = "log(TOF) [1/s]"
    filename = f"tof_volcano_{tag1}_{tag2}.png"
    if verb > 0:
        csvname = f"tof_volcano_{tag1}_{tag2}.csv"
        print(f"Saving TOF volcano data to file {csvname}")
        x = np.zeros_like(grid.reshape(-1))
        y = np.zeros_like(grid.reshape(-1))
        for i, xy in enumerate(itertools.product(xint, yint)):
            x[i] = xy[0]
            y[i] = xy[1]
        zdata = list(zip(x, y, grid.reshape(-1)))
        np.savetxt(
            csvname,
            zdata,
            fmt="%.4e",
            delimiter=",",
            header="Descriptor 1, Descriptor 2, log10(TOF)",
        )
    if plotmode == 2:
        plot_3d_contour(
            xint,
            yint,
            grid.T,
            px,
            py,
            ymin,
            ymax,
            x1min,
            x1max,
            x2min,
            x2max,
            x1base,
            x2base,
            x1label=x1label,
            x2label=x2label,
            ylabel=ylabel,
            filename=filename,
            cb=cb,
            ms=ms,
            plotmode=plotmode,
        )
    else:
        plot_3d_scatter(
            xint,
            yint,
            grid.T,
            px,
            py,
            ymin,
            ymax,
            x1min,
            x1max,
            x2min,
            x2max,
            x1base,
            x2base,
            x1label=x1label,
            x2label=x2label,
            ylabel=ylabel,
            filename=filename,
            cb=cb,
            ms=ms,
            plotmode=plotmode,
        )
    return xint, yint, grid, px, py


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


def plot_3d_contour(
    xint,
    yint,
    grid,
    px,
    py,
    ymin,
    ymax,
    x1min,
    x1max,
    x2min,
    x2max,
    x1base,
    x2base,
    x1label="X1-axis",
    x2label="X2-axis",
    ylabel="Y-axis",
    filename="plot.png",
    cb="white",
    ms="o",
    plotmode=2,
):
    fig, ax = plt.subplots(
        frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
    )
    grid = np.clip(grid, ymin, ymax)
    norm = cm.colors.Normalize(vmax=ymax, vmin=ymin)
    ax = beautify_ax(ax)
    levels = np.arange(ymin - 5, ymax + 5, 2.5)
    cset = ax.contourf(
        xint,
        yint,
        grid,
        levels=levels,
        norm=norm,
        cmap=cm.get_cmap("jet", len(levels)),
    )

    # Labels and key
    plt.xlabel(x1label)
    plt.ylabel(x2label)
    plt.xlim(x1min, x1max)
    plt.ylim(x2max, x2min)
    plt.xticks(np.arange(x1min, x1max + 0.1, x1base))
    plt.yticks(np.arange(x2min, x2max + 0.1, x2base))
    ax.contour(xint, yint, grid, cset.levels, colors="black", linewidths=0.3)
    fmt = lambda x, pos: "%.0f" % x
    cbar = fig.colorbar(cset, format=FuncFormatter(fmt))
    cbar.set_label(ylabel, labelpad=15, rotation=270)
    for i in range(len(px)):
        ax.scatter(
            px[i],
            py[i],
            s=12.5,
            c=cb[i],
            marker=ms[i],
            linewidths=0.15,
            edgecolors="black",
        )
    plt.savefig(filename)


def plot_3d_contour_regions(
    xint,
    yint,
    grid,
    px,
    py,
    ymin,
    ymax,
    x1min,
    x1max,
    x2min,
    x2max,
    x1base,
    x2base,
    x1label="X1-axis",
    x2label="X2-axis",
    ylabel="Y-axis",
    filename="plot.png",
    cb="white",
    ms="o",
    nunique=2,
    id_labels=[],
    plotmode=3,
):
    fig, ax = plt.subplots(
        frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
    )
    ax = beautify_ax(ax)
    levels = np.arange(-0.1, nunique + 0.9, 1)
    cset = ax.contourf(
        xint,
        yint,
        grid,
        levels=levels,
        cmap=cm.get_cmap("Dark2", nunique + 1),
    )

    # Labels and key
    plt.xlabel(x1label)
    plt.ylabel(x2label)
    plt.xlim(x1min, x1max)
    plt.ylim(x2max, x2min)
    plt.xticks(np.arange(x1min, x1max + 0.1, x1base))
    plt.yticks(np.arange(x2min, x2max + 0.1, x2base))
    ax.contour(xint, yint, grid, cset.levels, colors="black", linewidths=0.1)
    fmt = lambda x, pos: "%.0f" % x
    cbar = fig.colorbar(cset, format=FuncFormatter(fmt))
    cbar.set_ticks([])
    cbar.set_label(ylabel, labelpad=15, rotation=270)
    for j, tlab in enumerate(id_labels):
        cbar.ax.text(
            12.5,
            0.4 + j,
            tlab,
            ha="center",
            va="center",
            weight="light",
            fontsize=3.5,
            rotation=-90,
        )
        cbar.ax.get_yaxis().labelpad = 15
    for i in range(len(px)):
        ax.scatter(
            px[i],
            py[i],
            s=12.5,
            c=cb[i],
            marker=ms[i],
            linewidths=0.15,
            edgecolors="black",
        )
    plt.savefig(filename)


def plot_3d_scatter(
    xint,
    yint,
    grid,
    px,
    py,
    ymin,
    ymax,
    x1min,
    x1max,
    x2min,
    x2max,
    x1base,
    x2base,
    x1label="X1-axis",
    x2label="X2-axis",
    ylabel="Y-axis",
    filename="plot.png",
    cb="white",
    ms="o",
    plotmode=0,
):
    fig, ax = plt.subplots(
        frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
    )
    grid = np.clip(grid, ymin, ymax)
    norm = cm.colors.Normalize(vmax=ymax, vmin=ymin)
    ax = beautify_ax(ax)
    cset = ax.imshow(
        grid,
        interpolation="antialiased",
        extent=[x1min, x1max, x2min, x2max],
        origin="lower",
        cmap=cm.jet,
        aspect="auto",
    )
    # Labels and key
    plt.xlabel(x1label)
    plt.ylabel(x2label)
    plt.xlim(x1min, x1max)
    plt.ylim(x2max, x2min)
    plt.xticks(np.arange(x1min, x1max + 0.1, x1base))
    plt.yticks(np.arange(x2min, x2max + 0.1, x2base))
    fmt = lambda x, pos: "%.0f" % x
    cbar = fig.colorbar(cset, format=FuncFormatter(fmt))
    cbar.set_label(ylabel, labelpad=15, rotation=270)
    if plotmode == 1:
        for i in range(len(px)):
            ax.scatter(
                px[i],
                py[i],
                s=12.5,
                c=cb[i],
                marker=ms[i],
                linewidths=0.15,
                edgecolors="black",
            )
    plt.savefig(filename)

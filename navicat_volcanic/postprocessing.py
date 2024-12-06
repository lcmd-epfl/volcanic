#!/usr/bin/env python

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
from navicat_volcanic.tof import calc_es, calc_s_es, calc_tof, calc_atof
from navicat_volcanic.plotting2d import (
    plotpoints,
    plot_2d,
    calc_ci,
    plot_ci,
    beautify_ax,
)


def plot_2d_fer(
    tags,
    xint,
    yints,
    cis=[],
    cb="white",
    ms="o",
    lmargin=10,
    rmargin=10,
    plotmode=1,
    verb=0,
):
    xbase = 0.5
    ybase = 10
    # xmax = bround(X.max() + rmargin, xbase)
    # xmin = bround(X.min() - lmargin, xbase)
    # xint = np.linspace(xmin, xmax, npoints)

    xmax = max(xint)
    xmin = min(xint)

    for j in range(len(yints)):
        if isinstance(ms, np.ndarray):
            msi = ms[j]
        else:
            msi = ms
        if isinstance(cb, np.ndarray):
            cbi = cb[j]
        else:
            cbi = cb

        if verb > 0:
            print(f"Plotting regression of {tags[j]}.")
        yint = yints[j]
        fig, ax = plt.subplots(
            frameon=False, figsize=[4.2, 3], dpi=300, constrained_layout=True
        )

        # Actual plotting
        if cis:
            plot_ci(cis[j], xint, yint, ax=ax)
        ax.plot(xint, yint, "-", linewidth=1, color="#000a75", alpha=0.85, zorder=1)
        beautify_ax(ax)

        # Labels and key
        plt.xlabel(f"Descriptor [kcal/mol]")
        plt.ylabel(f"{tags[j]} [kcal/mol]")
        xmin, xmax = np.min(xint), np.max(xint)
        plt.xlim(xmin, xmax)

        ymin, ymax = ax.get_ylim()
        ymax = bround(ymax, ybase, type="max")
        ymin = bround(ymin, ybase, type="min")
        plt.ylim(ymin, ymax)
        # plt.yticks(np.arange(ymin, ymax + 0.1, ybase))
        plt.xticks(np.arange(xmin, xmax + 0.1, xbase))
        plt.savefig(f"{tags[j]}.png")


def plot_2d_tof_volcano_from_fer(
    tags,
    coeff,
    xint,
    yints,
    cis=[],
    reftof=0,
    dgr=-12.60,
    T=298.15,
    cb="white",
    ms="o",
    lmargin=15,
    rmargin=15,
    plotmode=0,
    verb=0,
):
    xbase = 0.5
    ybase = 5
    # xmax = bround(X.max() + rmargin, xbase)
    # xmin = bround(X.min() - lmargin, xbase)
    # xint = np.linspace(xmin, xmax, npoints)

    xmax = max(xint)
    xmin = min(xint)

    lnsteps = list(range(len(yints)))
    dgs = np.zeros((xint.shape[0], len(lnsteps)))
    sigma_dgs = np.zeros((xint.shape[0], len(lnsteps)))
    for j in lnsteps:
        if isinstance(ms, np.ndarray):
            msi = ms[j]
        else:
            msi = ms
        if isinstance(cb, np.ndarray):
            cbi = cb[j]
        else:
            cbi = cb

        yint = yints[j]
        dgs[:, j] = yint
        if cis:
            sigma_dgs[:, j] = cis[j]

    ytof = np.zeros_like(yint)
    ci = np.zeros_like(yint)
    ridmax = np.zeros_like(yint, dtype=int)
    ridmin = np.zeros_like(yint, dtype=int)

    # We must take the initial and ending states into account here
    for i in range(ytof.shape[0]):
        profile = dgs[i, :]
        if cis:
            sigmas = sigma_dgs[i, :-1]
        dgr_s = dgs[i][-1]
        tof, xtof, e = calc_tof(profile, dgr_s, T, coeff, exact=True)
        es, ridmax[i], ridmin[i], _ = calc_es(profile, dgr_s, esp=True)
        ltof = np.log10(tof)
        ytof[i] = ltof - reftof

        if cis:
            sigma_d = sigmas[ridmin[i] - 1] + sigmas[ridmax[i] - 1]
            sigma_tof_p = calc_atof(es + sigma_d, dgr_s, T)
            sigma_tof_m = calc_atof(es - sigma_d, dgr_s, T)
            sigma_ltof_p = np.log10(sigma_tof_p)
            sigma_ltof_m = np.log10(sigma_tof_m)
            sigma_ltof = (sigma_ltof_m - sigma_ltof_p) / 2
            ci[i] = sigma_ltof
        else:
            ci[i] = 0

    # This bit would be used to plot the actual points used to fit the volcano in hte first place

    # px = np.zeros_like(d[:, 0])
    # py = np.zeros_like(d[:, 0])
    # for i in range(d.shape[0]):
    #    px[i] = X[i].reshape(-1)
    #    profile = d[i, :]
    #    dgr_s = dgr[i]
    #    tof = calc_tof(profile, dgr_s, T, coeff, exact=True)[0]
    #    ltof = np.log10(tof)
    #    py[i] = ltof
    #    if verb > 2:
    #        print(f"Profile {profile} corresponds with log10(TOF) of {ltof}")
    #    if verb > 1:
    #        pointsname = f"points_tof_volcano_{tag}.csv"
    #        zdata = list(zip(px, py))
    #        np.savetxt(
    #            pointsname,
    #            zdata,
    #            fmt="%.4e",
    #            delimiter=",",
    #            header="Descriptor, log10(TOF)",
    #        )
    px = []
    py = []

    xlabel = f"Descriptor [kcal/mol]"
    ylabel = "log(TOF) [1/s]"
    filename = f"tof_volcano.png"
    if verb > 0:
        csvname = f"tof_volcano.csv"
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


def test_postprocessing():
    xint = np.linspace(2.3, 3.6, 200)

    # Tags
    tags = ["I1", "TS1", "I2", "I3", "TS2", "I4", "P"]

    # Coeff
    coeff = np.array([0, 1, 0, 0, 1, 0, 0], dtype=bool)

    # \DeltaG_R
    dgr = -12.60

    # Taking more parameters from fit to 2D data, similar but not exactly the ones in the paper
    params = [
        [26.181, 0.8623, 2.9912, -11.3859, 0.1269, 104.952, 15.445],
        [20.3505, 0.9215, 2.8842, -6.4305, 0.2362, 91.8199, 7.5784],
        [77.7906, 0.23, 3.1494, -1.6306, 0.5691, 24.2096, -4.9785],
        [17.4216, 0.4025, 3.2615, -2.3652, 0.5756, 47.6048, 7.5767],
        [
            5.79986e01,
            5.75200e-01,
            3.22120e00,
            -1.70122e01,
            -2.61000e-02,
            1.97040e01,
            -2.33080e01,
        ],
    ]

    angles = [110.0, 100.0, 60.0, 80.0, 20.0]

    def energy_dist2(i, j, de, a, c, z1, z2, q3=0.0):
        g = 1 - c * np.cos((z2 - j) * 0.0174533) ** 2
        energy = q3 + de * (1 - np.exp(-(a / g) * (i - (z1)))) ** 2
        return energy

    def energy_dist(i, j, de, a, c, z1, z2, q3=0.0):
        return energy_dist2(i, j, de, a, c, z1, z2, q3)

    def energy_angle2(
        i, j, b, c, z1, z2, q3
    ):  # A scaled cosine for correct periodicity
        g = 1  # / (1 + np.abs(z1 - i))
        energy = b * g * np.cos(((j - z2)) * 0.0174533) ** 2
        return energy

    def energy_angle(i, j, b, c, z1, z2, q3=0.0):
        j = np.where(j > 180.0, j - 180.0, j)
        j = np.where(j < 0.0, j + 180.0, j)
        return energy_angle2(i, j, b, c, z1, z2, q3)

    def energy_func_add(i, j, de, a, q1, b, c, q2, q3):
        return (
            energy_dist(i, j, de, a, c, q1, q2) + q3 + energy_angle(i, j, b, c, q1, q2)
        )

    yints = []
    yints.append(
        np.zeros_like(xint)
    )  # The reference state, I1, is 0 for all catalysts typically
    for i, param in enumerate(params):
        yint = np.array(
            [
                energy_func_add(
                    x,
                    angles[i],
                    param[0],
                    param[1],
                    param[2],
                    param[3],
                    param[4],
                    param[5],
                    param[6],
                )
                for x in xint
            ]
        )
        yints.append(yint)
    yints.append(
        np.array([dgr for _ in xint])
    )  # The final step of the reaction (regeneration of the catalyst) has a fixed cost of dgr in this reaction (it only depends on product)
    plot_2d_fer(tags, xint, yints)
    # In the FLP paper we do not plot TOF, we plot \DeltaTOF w.r.t. some other catalyst with TOF -6.615959074058314
    plot_2d_tof_volcano_from_fer(tags, coeff, xint, yints, reftof=-6.615959074058314)


if __name__ == "__main__":
    test_postprocessing()

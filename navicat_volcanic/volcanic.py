#!/usr/bin/env python

import sys

import numpy as np

from navicat_volcanic.dv1 import curate_d, find_1_dv
from navicat_volcanic.dv2 import find_2_dv
from navicat_volcanic.exceptions import InputError
from navicat_volcanic.helpers import (
    arraydump,
    group_data_points,
    processargs,
    setflags,
    user_choose_1_dv,
    user_choose_2_dv,
)
from navicat_volcanic.plotting2d import (
    plot_2d_es_volcano,
    plot_2d_k_volcano,
    plot_2d_lsfer,
    plot_2d_t_volcano,
    plot_2d_tof_volcano,
)
from navicat_volcanic.plotting3d import (
    plot_3d_es_volcano,
    plot_3d_k_volcano,
    plot_3d_lsfer,
    plot_3d_t_volcano,
    plot_3d_tof_volcano,
)


def run_volcanic():
    (
        df,
        nd,
        verb,
        runmode,
        temp,
        imputer_strat,
        refill,
        dump,
        ic,
        fc,
        lmargin,
        rmargin,
        npoints,
        plotmode,
    ) = processargs(sys.argv[1:])

    # Fill in reaction profile names/IDs from input data.
    if verb > 0:
        print(
            f"volcanic will assume that {df.columns[0]} contains names/IDs of reaction profiles."
        )
    names = df[df.columns[0]].values

    # Atttempts to group data points based on shared characters in names.
    cb, ms = group_data_points(ic, fc, names)

    # Expecting a full reaction profile with corresponding column names. Descriptors will be identified.
    tags = np.array([str(tag) for tag in df.columns[1:]], dtype=object)
    d = np.float32(df.to_numpy()[:, 1:])

    # TS or intermediate are interpreted from column names. Coeffs is a boolean array.
    coeff = np.zeros(len(tags), dtype=bool)
    regress = np.zeros(len(tags), dtype=bool)
    for i, tag in enumerate(tags):
        if "TS" in tag.upper():
            if verb > 0:
                print(f"Assuming field {tag} corresponds to a TS.")
            coeff[i] = True
            regress[i] = True
        elif "DESCRIPTOR" in tag.upper():
            if verb > 0:
                print(
                    f"Assuming field {tag} corresponds to a non-energy descriptor variable."
                )
            start_des = tag.upper().find("DESCRIPTOR")
            tags[i] = "".join(
                [i for i in tag[:start_des]] + [i for i in tag[start_des + 10 :]]
            )
            coeff[i] = False
            regress[i] = False
        elif "PRODUCT" in tag.upper():
            if verb > 0:
                print(f"Assuming ΔG of the reaction(s) are given in field {tag}.")
            dgr = d[:, i]
            coeff[i] = False
            regress[i] = True
        else:
            if verb > 0:
                print(f"Assuming field {tag} corresponds to a non-TS stationary point.")
            coeff[i] = False
            regress[i] = True

    # Your data might contain outliers (human error, computation error) or missing points.
    # We will attempt to curate your data automatically.
    d, cb, ms = curate_d(d, regress, cb, ms, tags, imputer_strat, nstds=3, verb=verb)

    if verb > 0:
        print(f"Reaction profile is given by stationary points:\n {tags[regress]}")

    # Runmode used to set up flags for volcano generation.
    t_volcano, k_volcano, es_volcano, tof_volcano = setflags(runmode)

    if nd == 1:
        # volcanic will find best non-TS descriptor variable
        dvs, r2s = find_1_dv(d, tags, coeff, regress, verb)
        idx = user_choose_1_dv(dvs, r2s, tags)
        if verb > 0:
            print(f"Reaction profile is given by stationary points:\n {tags[regress]}")
        if idx is not None:
            print(f"Generating LSR plots using descriptor variable {tags[idx]}")
            if refill:
                d = plot_2d_lsfer(
                    idx,
                    d,
                    tags,
                    coeff,
                    regress,
                    cb,
                    ms,
                    lmargin,
                    rmargin,
                    npoints,
                    plotmode,
                    verb,
                )
            else:
                _ = plot_2d_lsfer(
                    idx,
                    d,
                    tags,
                    coeff,
                    regress,
                    cb,
                    ms,
                    lmargin,
                    rmargin,
                    npoints,
                    plotmode,
                    verb,
                )

            volcano_headers = []
            volcano_list = []
            xint = None

            if t_volcano:
                print(
                    f"Generating 2D thermodynamic volcano plot using descriptor variable {tags[idx]}"
                )
                xint, ymin, px, py, xmin, xmax, rid, rb = plot_2d_t_volcano(
                    idx,
                    d,
                    tags,
                    coeff,
                    regress,
                    dgr,
                    cb,
                    ms,
                    lmargin,
                    rmargin,
                    npoints,
                    plotmode,
                    verb,
                )
                volcano_headers.append("Thermodynamic volcano")
                volcano_list.append(ymin)

            if k_volcano:
                print(
                    f"Generating 2D kinetic volcano plot using descriptor variable {tags[idx]}"
                )
                xint, ymin, px, py, xmin, xmax, rid, rb = plot_2d_k_volcano(
                    idx,
                    d,
                    tags,
                    coeff,
                    regress,
                    dgr,
                    cb,
                    ms,
                    lmargin,
                    rmargin,
                    npoints,
                    plotmode,
                    verb,
                )
                volcano_headers.append("Kinetic volcano")
                volcano_list.append(ymin)

            if es_volcano:
                print(
                    f"Generating 2D energy span volcano plot using descriptor variable {tags[idx]}"
                )
                xint, ymin, px, py, xmin, xmax, rid, rb = plot_2d_es_volcano(
                    idx,
                    d,
                    tags,
                    coeff,
                    regress,
                    dgr,
                    cb,
                    ms,
                    lmargin,
                    rmargin,
                    npoints,
                    plotmode,
                    verb,
                )
                volcano_headers.append("ES volcano")
                volcano_list.append(ymin)

            if tof_volcano:
                print(
                    f"Generating 2D TOF volcano plot using descriptor variable {tags[idx]}"
                )
                xint, ytof, px, py, xmin, xmax = plot_2d_tof_volcano(
                    idx,
                    d,
                    tags,
                    coeff,
                    regress,
                    dgr,
                    temp,
                    cb,
                    ms,
                    lmargin,
                    rmargin,
                    npoints,
                    plotmode,
                    verb,
                )
                volcano_headers.append("TOF volcano")
                volcano_list.append(ytof)

            if dump and xint is not None:
                arraydump("2d_volcanos.hdf5", xint, volcano_list, volcano_headers)

    if nd == 2:
        # volcanic will find best non-TS combination of two descriptor variables
        dvs, r2s = find_2_dv(d, tags, coeff, regress, verb)
        idx1, idx2 = user_choose_2_dv(dvs, r2s, tags)
        if idx1 is not None and idx2 is not None:
            print(
                f"Generating multivariate LSR plots using combination of descriptor variables {tags[idx1]} and {tags[idx2]}"
            )
            if refill:
                d = plot_3d_lsfer(
                    idx1,
                    idx2,
                    d,
                    tags,
                    coeff,
                    regress,
                    cb,
                    ms,
                    lmargin,
                    rmargin,
                    npoints,
                    plotmode,
                    verb,
                )
            else:
                _ = plot_3d_lsfer(
                    idx1,
                    idx2,
                    d,
                    tags,
                    coeff,
                    regress,
                    cb,
                    ms,
                    lmargin,
                    rmargin,
                    npoints,
                    plotmode,
                    verb,
                )

            volcano_headers = []
            volcano_list = []
            x1int = None
            x2int = None

            if t_volcano:
                print(
                    f"Generating 3D thermodynamic volcano plot using combination of descriptor variables {tags[idx1]} and {tags[idx2]}"
                )
                x1int, x2int, grid, px, py = plot_3d_t_volcano(
                    idx1,
                    idx2,
                    d,
                    tags,
                    coeff,
                    regress,
                    dgr,
                    cb,
                    ms,
                    lmargin,
                    rmargin,
                    npoints,
                    plotmode,
                    verb,
                )
                volcano_headers.append("Thermodynamic volcano")
                volcano_list.append(grid)

            if k_volcano:
                print(
                    f"Generating 3D kinetic volcano plot using combination of descriptor variables {tags[idx1]} and {tags[idx2]}"
                )
                x1int, x2int, grid, px, py = plot_3d_k_volcano(
                    idx1,
                    idx2,
                    d,
                    tags,
                    coeff,
                    regress,
                    dgr,
                    cb,
                    ms,
                    lmargin,
                    rmargin,
                    npoints,
                    plotmode,
                    verb,
                )
                volcano_headers.append("Kinetic volcano")
                volcano_list.append(grid)

            if es_volcano:
                print(
                    f"Generating 3D energy span volcano plot using combination of descriptor variables {tags[idx1]} and {tags[idx2]}"
                )
                x1int, x2int, grid, px, py = plot_3d_es_volcano(
                    idx1,
                    idx2,
                    d,
                    tags,
                    coeff,
                    regress,
                    dgr,
                    cb,
                    ms,
                    lmargin,
                    rmargin,
                    npoints,
                    plotmode,
                    verb,
                )
                volcano_headers.append("ES volcano")
                volcano_list.append(grid)

            if tof_volcano:
                print(
                    f"Generating 3D TOF volcano plot using combination of descriptor variables {tags[idx1]} and {tags[idx2]}"
                )
                x1int, x2int, grid, px, py = plot_3d_tof_volcano(
                    idx1,
                    idx2,
                    d,
                    tags,
                    coeff,
                    regress,
                    dgr,
                    temp,
                    cb,
                    ms,
                    lmargin,
                    rmargin,
                    npoints,
                    plotmode,
                    verb,
                )
                volcano_headers.append("TOF volcano")
                volcano_list.append(grid)

            if dump and x1int is not None and x2int is not None:
                arraydump(
                    "3d_volcanos.hdf5",
                    np.vstack([x1int, x2int]),
                    volcano_list,
                    volcano_headers,
                )

    def volcanic_2d(
        runmode,
        idx,
        d,
        tags,
        coeff,
        regress,
        dgr,
        temp=298.15,
        cb=None,
        ms=None,
        lmargin=20,
        rmargin=20,
        npoints=200,
        plotmode=1,
        verb=0,
    ):
        if runmode == 5:
            raise InputError(
                "In function mode, options requiring explicit CLI input are disabled."
            )
        t_volcano, k_volcano, es_volcano, tof_volcano = setflags(runmode)
        d = plot_2d_lsfer(
            idx,
            d,
            tags,
            coeff,
            regress,
            cb,
            ms,
            lmargin,
            rmargin,
            npoints,
            plotmode,
            verb,
        )
        if t_volcano:
            return plot_2d_t_volcano(
                idx,
                d,
                tags,
                coeff,
                regress,
                dgr,
                cb,
                ms,
                lmargin,
                rmargin,
                npoints,
                plotmode,
                verb,
            )
        if k_volcano:
            return plot_2d_k_volcano(
                idx,
                d,
                tags,
                coeff,
                regress,
                dgr,
                cb,
                ms,
                lmargin,
                rmargin,
                npoints,
                plotmode,
                verb,
            )
        if es_volcano:
            return plot_2d_es_volcano(
                idx,
                d,
                tags,
                coeff,
                regress,
                dgr,
                cb,
                ms,
                lmargin,
                rmargin,
                npoints,
                plotmode,
                verb,
            )
        if tof_volcano:
            return plot_2d_tof_volcano(
                idx,
                d,
                tags,
                coeff,
                regress,
                dgr,
                temp,
                cb,
                ms,
                lmargin,
                rmargin,
                npoints,
                plotmode,
                verb,
            )

    def volcanic_3d(
        runmode,
        idx1,
        idx2,
        d,
        tags,
        coeff,
        regress,
        dgr,
        temp=298.15,
        cb=None,
        ms=None,
        lmargin=20,
        rmargin=20,
        npoints=200,
        plotmode=1,
        verb=0,
    ):
        if runmode == 5:
            raise InputError(
                "In function mode, options requiring explicit CLI input are disabled."
            )
        t_volcano, k_volcano, es_volcano, tof_volcano = setflags(runmode)
        if t_volcano:
            return plot_3d_t_volcano(
                idx1,
                idx2,
                d,
                tags,
                coeff,
                regress,
                dgr,
                cb,
                ms,
                lmargin,
                rmargin,
                npoints,
                plotmode,
                verb,
            )
        if k_volcano:
            return plot_3d_k_volcano(
                idx1,
                idx2,
                d,
                tags,
                coeff,
                regress,
                dgr,
                cb,
                ms,
                lmargin,
                rmargin,
                npoints,
                plotmode,
                verb,
            )
        if es_volcano:
            return plot_3d_es_volcano(
                idx1,
                idx2,
                d,
                tags,
                coeff,
                regress,
                dgr,
                cb,
                ms,
                lmargin,
                rmargin,
                npoints,
                plotmode,
                verb,
            )
        if tof_volcano:
            return plot_3d_tof_volcano(
                idx1,
                idx2,
                d,
                tags,
                coeff,
                regress,
                dgr,
                temp,
                cb,
                ms,
                lmargin,
                rmargin,
                npoints,
                plotmode,
                verb,
            )

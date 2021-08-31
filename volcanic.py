#!/usr/bin/env python

import sys
import numpy as np
from dv1 import curate_d, find_1_dv
from dv2 import find_2_dv
from plotting2d import (
    plot_2d_lsfer,
    plot_2d_t_volcano,
    plot_2d_k_volcano,
    plot_2d_tof_volcano,
)
from plotting3d import (
    plot_3d_lsfer,
    plot_3d_t_volcano,
    plot_3d_k_volcano,
    plot_3d_tof_volcano,
)
from helpers import (
    yesno,
    processargs,
    group_data_points,
    user_choose_1_dv,
    user_choose_2_dv,
    arraydump,
)


if __name__ == "__main__":
    arguments = []
    for i, arg in enumerate(sys.argv[1:]):
        arguments.append(arg)
    df, nd, verb, runmode, T, imputer_strat, refill, dump, bc, ec = processargs(
        arguments
    )


# Fill in reaction profile names/IDs from input data.
if verb > 0:
    print(
        f"VOLCANIC will assume that {df.columns[0]} contains names/IDs of reaction profiles."
    )
names = df[df.columns[0]].values

# Atttempts to group data points based on shared characters in names.
cb, ms = group_data_points(bc, ec, names)

# After the first column, we expect a full reaction profile with corresponding column names.
tags = [str(tag) for tag in df.columns[1:]]
if verb > 0:
    print(f"Reaction profile is given by stationary points:\n {tags}")
d = np.float64(df.to_numpy()[:, 1:])

# We expect the last field of any and all reaction profiles to be the reaction \DeltaG.
dgr = np.float64(d[0, -1])
if verb > 0:
    print(f"ΔG of the reaction set to {dgr}.")

# TS or intermediate are interpreted from column names. Coeffs is a boolean array.
coeffs = []
for i in tags:
    if "TS" in i.upper():
        if verb > 0:
            print(f"Assuming field {i} corresponds to a TS.")
        coeffs.append(1)
    else:
        if verb > 0:
            print(f"Assuming field {i} does not correspond to a TS.")
        coeffs.append(0)
coeff = np.array(coeffs, dtype=bool)

# Your data might contain outliers (human error, computation error) or missing points.
# We will attempt to curate your data automatically.
d, cb, ms = curate_d(d, cb, ms, tags, imputer_strat, verb)


# Runmode used to set up flags for volcano generation.
if runmode == 0:
    t_volcano = False
    k_volcano = False
    tof_volcano = False

if runmode == 1:
    t_volcano = True
    k_volcano = False
    tof_volcano = False

if runmode == 2:
    t_volcano = False
    k_volcano = True
    tof_volcano = True

if runmode == 3:
    t_volcano = yesno("Generate 2D thermodynamic volcano plot")
    k_volcano = yesno("Generate 2D kinetic volcano plot")
    tof_volcano = yesno("Generate 2D TOF volcano plot")

if nd == 1:
    # VOLCANIC will find best non-TS descriptor variable
    dvs, r2s = find_1_dv(d, tags, coeff, verb)
    idx = user_choose_1_dv(dvs, r2s, tags)
    if idx is not None:
        print(f"Generating LSFER plots using descriptor variable {tags[idx]}")
        if refill:
            d = plot_2d_lsfer(idx, d, tags, coeff, cb, ms, verb)
        else:
            plot_2d_lsfer(idx, d, tags, coeff, cb, ms, verb)

        volcano_headers = []
        volcano_list = []
        xint = None

        if t_volcano:
            print(
                f"Generating 2D thermodynamic volcano plot using descriptor variable {tags[idx]}"
            )
            # The function call also returns all the relevant information
            # xint is the x axis vector, ymin is the -\DGpds vector, px and py are the original datapoints, rid and rb are regions
            xint, ymin, px, py, xmin, xmax, rid, rb = plot_2d_t_volcano(
                idx, d, tags, coeff, dgr, cb, ms, verb
            )
            volcano_headers.append("Thermodynamic volcano")
            volcano_list.append(ymin)

        if k_volcano:
            print(
                f"Generating 2D kinetic volcano plot using descriptor variable {tags[idx]}"
            )
            # The function call also returns all the relevant information
            # xint is the x axis vector, ymin is the -\DGpds vector, px and py are the original datapoints, rid and rb are regions
            xint, ymin, px, py, xmin, xmax, rid, rb = plot_2d_k_volcano(
                idx, d, tags, coeff, dgr, cb, ms, verb
            )
            volcano_headers.append("Kinetic volcano")
            volcano_list.append(ymin)

        if tof_volcano:
            print(
                f"Generating 2D TOF volcano plot using descriptor variable {tags[idx]}"
            )
            # The function call also returns all the relevant information
            # xint is the x axis vector, ymin is the TOF vector, px and py are the original datapoints
            xint, ytof, px, py, xmin, xmax = plot_2d_tof_volcano(
                idx, d, tags, coeff, dgr, T, cb, ms, verb
            )
            volcano_headers.append("TOF volcano")
            volcano_list.append(ytof)

        if dump and xint is not None:
            arraydump("2d_volcanos.hdf5", xint, volcano_list, volcano_headers)

if nd == 2:
    # VOLCANIC will find best non-TS combination of two descriptor variables
    dvs, r2s = find_2_dv(d, tags, coeff, verb)
    idx1, idx2 = user_choose_2_dv(dvs, r2s, tags)
    if idx1 is not None and idx2 is not None:
        print(
            f"Generating multivariate LSFER plots using combination of descriptor variables {tags[idx1]} and {tags[idx2]}"
        )
        if refill:
            d = plot_3d_lsfer(idx1, idx2, d, tags, coeff, cb, ms, verb)
        else:
            plot_3d_lsfer(idx1, idx2, d, tags, coeff, cb, ms, verb)

        volcano_headers = []
        volcano_list = []
        x1int = None
        x2int = None

        if t_volcano:
            print(
                f"Generating 3D thermodynamic volcano plot using combination of descriptor variables {tags[idx1]} and {tags[idx2]}"
            )
            # The function call also returns all the relevant information
            # xint is the x axis vector, ymin is the -\DGpds vector, px and py are the original datapoints, rid and rb are regions
            x1int, x2int, grid, px, py = plot_3d_t_volcano(
                idx1, idx2, d, tags, coeff, dgr, cb, ms, verb
            )
            volcano_headers.append("Thermodynamic volcano")
            volcano_list.append(grid)

        if k_volcano:
            print(
                f"Generating 3D kinetic volcano plot using combination of descriptor variables {tags[idx1]} and {tags[idx2]}"
            )
            # The function call also returns all the relevant information
            # xint is the x axis vector, ymin is the -\DGpds vector, px and py are the original datapoints, rid and rb are regions
            x1int, x2int, grid, px, py = plot_3d_k_volcano(
                idx1, idx2, d, tags, coeff, dgr, cb, ms, verb
            )
            volcano_headers.append("Kinetic volcano")
            volcano_list.append(grid)

        if tof_volcano:
            print(
                f"Generating 3D TOF volcano plot using combination of descriptor variables {tags[idx1]} and {tags[idx2]}"
            )
            x1int, x2int, grid, px, py = plot_3d_tof_volcano(
                idx1, idx2, d, tags, coeff, dgr, T, cb, ms, verb
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

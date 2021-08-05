#!/usr/bin/env python

import sys
import numpy as np
from dv import curate_d, find_dv
from plotting import plot_lsfer, plot_2d_volcano, plot_2d_tof_volcano
from helpers import yesno, processargs, group_data_points


if __name__ == "__main__":
    arguments = []
    for i, arg in enumerate(sys.argv[1:]):
        arguments.append(arg)
    df, nd, verb, T, imputer_strat, refill, bc, ec = processargs(arguments)


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

# VOLCANIC will find best non-TS \DeltaG that correlates with all others.
dvs, r2s = find_dv(d, tags, coeff, verb)

for dv, r2 in zip(dvs, r2s):
    print(
        f"\n{tags[dv]} has been identified as a suitable descriptor variable with r2={np.round(r2,4)}."
    )
    ok = yesno("Continue using this variable?")
    if ok:
        idx = dv
        break
if not ok:
    manual = yesno("Would you want to use some other descriptor variable instead")
    if manual:
        for i, tag in enumerate(tags):
            ok = yesno(f"Use descriptor {tag}")
            if ok:
                idx = i
                break
if ok:
    print(f"Generating LSFER plots using descriptor variable {tags[idx]}")
    if refill:
        d = plot_lsfer(idx, d, tags, coeff, cb, ms, verb)
    else:
        plot_lsfer(idx, d, tags, coeff, cb, ms, verb)

volcano = yesno("Generate volcano plot")

if volcano:
    print(f"Generating volcano plot using descriptor variable {tags[idx]}")
    # The function call also returns all the relevant information
    # xint is the x axis vector, ymin is the -\DGpds vector, px and py are the original datapoints, rid and rb are regions
    xint, ymin, px, py, xmin, xmax, rid, rb = plot_2d_volcano(
        idx, d, tags, coeff, dgr, cb, ms, verb
    )


tof_volcano = yesno("Generate TOF volcano plot")

if tof_volcano:
    print(f"Generating TOF volcano plot using descriptor variable {tags[idx]}")
    # The function call also returns all the relevant information
    # xint is the x axis vector, ymin is the TOF vector, px and py are the original datapoints
    xint, ytof, px, py, xmin, xmax = plot_2d_tof_volcano(
        idx, d, tags, coeff, dgr, T, cb, ms, verb
    )

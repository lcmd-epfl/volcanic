#!/usr/bin/env python

import sys
import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk
from sklearn.manifold import TSNE
import sklearn.metrics.pairwise
import matplotlib
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import cm
from itertools import cycle
import scipy.constants as sc

from dv import curate_d, find_dv, plot_lsfer, plot_volcano, plot_tof_volcano
from helpers import yesno, processargs


if __name__ == "__main__":
    arguments = []
    for i, arg in enumerate(sys.argv[1:]):
        arguments.append(arg)
    df, verb, T, imputer_strat = processargs(arguments)

# Numpy for convenience, brief data formatting
# It is assumed that the first column of the input contains names!
names = df[df.columns[0]].values

# Example on how to generate color labels from data column. Totally optional.
# In this case, assumes that the first two characters of the name label are groups.
groups = np.array([i[0:2] for i in names], dtype=object)
type_tags = np.unique(groups)
cycol = cycle("bgrcmky")
cymar = cycle("^ospXDvH")
cdict = dict(zip(type_tags, cycol))
mdict = dict(zip(type_tags, cymar))
cb = np.array([cdict[i] for i in groups])
ms = np.array([mdict[i] for i in groups])

# This script assumes that a profile has a given format in the input data file
# and the first column corresponds to name labels, then the full energy profile
# the stationary points in the profile must have a corresponding column name
tags = [str(tag) for tag in df.columns[1:]]
d = np.float64(df.to_numpy()[:, 1:])

# We expect the last field of ANY reaction profile is the reaction \DeltaG
dgr = np.float64(d[0, -1])
print(f"ΔG of the reaction set to {dgr}.")

# TS or intermediate are interpreted from column names, see tags above
coeffs = []
for i in tags:
    if "TS" in i.upper():
        print(f"Assuming field {i} corresponds to a TS.")
        coeffs.append(1)
    else:
        print(f"Assuming field {i} does not correspond to a TS.")
        coeffs.append(0)
coeff = np.array(coeffs)

# Your data might contain outliers (human error, computation error) or missing points
# here we will try to fix that using an imputer if needed
d, cb, ms = curate_d(d, cb, ms, tags, imputer_strat, verb)

# All set, here we go!
# Will find best non-TS \DeltaG that correlates with all others
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
    print(f"Generating plots using descriptor variable {tags[idx]}")
    d = plot_lsfer(idx, d, tags, coeff, cb, ms, verb)

volcano = yesno("Generate volcano plot")

if volcano:
    print(f"Generating volcano plot using descriptor variable {tags[idx]}")
    # The function call also returns all the relevant information
    # xint is the x axis vector, ymin is the -\DGpds vector, px and py are the original datapoints, rid and rb are regions
    xint, ymin, px, py, xmin, xmax, rid, rb = plot_volcano(
        idx, d, tags, coeff, dgr, cb, ms, verb
    )


tof_volcano = yesno("Generate TOF volcano plot")

if tof_volcano:
    print(f"Generating TOF volcano plot using descriptor variable {tags[idx]}")
    # The function call also returns all the relevant information
    # xint is the x axis vector, ymin is the TOF vector, px and py are the original datapoints
    xint, ytof, px, py, xmin, xmax = plot_tof_volcano(
        idx, d, tags, coeff, dgr, T, cb, ms, verb
    )

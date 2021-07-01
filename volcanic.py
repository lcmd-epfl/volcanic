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

from dv import find_dv, plot_lsfer, plot_volcano, plot_tof_volcano
from helpers import yesno, processargs


if __name__ == "__main__":
    arguments = []
    for i, arg in enumerate(sys.argv[1:]):
        arguments.append(arg)
    df, verb, T = processargs(arguments)

# Numpy for convenience, brief data formatting
names = df[df.columns[0]].values
# Example on how to generate color labels
metals = np.array([i[0:2] for i in names], dtype=object)
type_tags = np.unique(metals)
cycol = cycle("bgrcmk")
cdict = dict(zip(metals, cycol))
cb = [cdict[i] for i in metals]

# This script assumes that a profile has a given format in the input data file
# can be edited to work with other input formats
tags = df.columns[2:-1]
mdf = df.to_numpy()
d = np.float64(mdf[:, 2:-1])
tags = [str(tag) for tag in tags]

# The last field of a profile is the reaction \DeltaG
dgr = np.float64(mdf[0, -1])
print(f"ΔG of the reaction set to {dgr}.")
lnsteps = range(d.shape[1])

# TS or intermediate are interpreted from column names
coeffs = []
for i in tags:
    if "TS" in i:
        print(f"Assuming field {i} corresponds to a TS.")
        coeffs.append(1)
    else:
        print(f"Assuming field {i} does not correspond to a TS.")
        coeffs.append(0)
coeff = np.array(coeffs)

# Will find best non-TS \DeltaG that correlates with all others
dvs = find_dv(d, tags, coeff, lnsteps, verb)

for dv in dvs:
    print(f"\n{tags[dv]} has been identified as a suitable descriptor variable.")
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
    plot_lsfer(idx, d, tags, coeff, lnsteps, cb, verb)

volcano = yesno("Generate volcano plot")

if volcano:
    print(f"Generating volcano plot using descriptor variable {tags[idx]}")
    # The function call also returns all the relevant information
    # xint is the x axis vector, ymin is the -\DGpds vector, px and py are the original datapoints, rid and rb are regions
    xint, ymin, px, py, xmin, xmax, rid, rb = plot_volcano(
        idx, d, tags, coeff, lnsteps, dgr, cb, verb
    )


tof_volcano = yesno("Generate TOF volcano plot")

if tof_volcano:
    print(f"Generating TOF volcano plot using descriptor variable {tags[idx]}")
    plot_tof_volcano(idx, d, tags, coeff, lnsteps, dgr, T, cb, verb)

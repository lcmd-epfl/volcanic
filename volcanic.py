#!/usr/bin/env python

import sys
import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk
from sklearn.manifold import TSNE
import sklearn.metrics.pairwise
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from matplotlib import cm
import scipy.constants as sc

from dv import find_dv, plot_lsfer, plot_volcano, plot_tof_volcano
from helpers import yesno

excel_terms = ["xlsx", "xls"]
verb = 0
T = 298.15

if __name__ == "__main__":
    arguments = []
    for i, arg in enumerate(sys.argv[1:]):
        arguments.append(arg)
    dfs = []
    for idx, i in enumerate(arguments):
        if i[-3:] == "csv":
            dfs.append(pd.read_csv(i))
            print(f"Input {i} was read from csv.")
        elif i.split(".")[-1] in excel_terms:
            dfs.append(pd.read_excel(i))
            print(f"Input {i} was read from excel format.")
        else:
            if i == "-t":
                T = np.float(arguments[idx + 1])
                print(f"Temperature manually set to {T}.")
            if i == "-v":
                verb = int(arguments[idx + 1])
                print(f"Verbosity manually set to {verb}.")
    if len(dfs) > 1:
        df = pd.concat(dfs)
    else:
        df = dfs[0]
    assert isinstance(df, pd.DataFrame)
    if verb > 1:
        print("Final database :")
        print(df.head())

# Numpy for convenience
names = df[df.columns[0]].values
tags = df.columns[2:-1]
tags = [str(tag) for tag in tags]
mdf = df.to_numpy()
d = np.float64(mdf[:, 2:-1])
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
    print(
        f"\n{tags[dv].format()} has been identified as a suitable descriptor variable."
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
    elif not ok:
        exit()

print(f"Generating plots using descriptor variable {tags[idx]}")
plot_lsfer(idx, d, tags, coeff, lnsteps, verb)

volcano = yesno("Generate volcano plot")

if volcano:
    plot_volcano(idx, d, tags, coeff, lnsteps, dgr, verb)

tof_volcano = yesno("Generate TOF volcano plot")

if tof_volcano:
    plot_tof_volcano(idx, d, tags, coeff, lnsteps, dgr, T, verb)

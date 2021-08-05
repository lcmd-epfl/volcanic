#!/usr/bin/env python

import sys
import pandas as pd
import numpy as np
from itertools import cycle


def yesno(question):
    """Simple Yes/No Function."""
    prompt = f"{question} ? (y/n): "
    ans = input(prompt).strip().lower()
    if ans not in ["y", "n"]:
        print(f"{ans} is invalid, please try again...")
        return yesno(question)
    if ans == "y":
        return True
    return False


def bround(x, base=5):
    return base * round(int(x) / base)


def group_data_points(bc, ec, names):
    try:
        groups = np.array([i[bc:ec].upper() for i in names], dtype=object)
    except Exception as m:
        print(f"Grouping by name characters did not work. Error message was:\n {m}")
        exit()
    type_tags = np.unique(groups)
    cycol = cycle("bgrcmky")
    cymar = cycle("^ospXDvH")
    cdict = dict(zip(type_tags, cycol))
    mdict = dict(zip(type_tags, cymar))
    cb = np.array([cdict[i] for i in groups])
    ms = np.array([mdict[i] for i in groups])
    return cb, ms


def processargs(arguments):
    nd = 1
    T = 298.15
    imputer_strat = "none"
    verb = 0
    refill = False
    bc = 0
    ec = 2
    terms = []
    filenames = []
    outname = None
    skip = False
    for idx, argument in enumerate(arguments):
        if skip:
            skip = False
            continue
        if argument == "-i" or argument == "-I":
            filename = str(arguments[idx + 1])
            filenames.append(filename)
            terms.append(filename.split(".")[-1])
            print(f"Input filename set to {filename}.")
            skip = True
        elif argument == "-nd":
            nd = int(arguments[idx + 1])
            print(f"Number of descriptor variables  manually set to {nd}.")
            skip = True
        elif argument == "-t" or argument == "-T":
            T = float(arguments[idx + 1])
            print(f"Temperature manually set to {T}.")
            skip = True
        elif argument == "-v" or argument == "-V":
            verb = int(arguments[idx + 1])
            print(f"Verbosity manually set to {verb}.")
            skip = True
        elif argument == "-is":
            imputer_strat = str(arguments[idx + 1])
            print(f"Imputer strategy manually set to {imputer_strat}.")
            skip = True
        elif argument == "-re":
            refill = bool(arguments[idx + 1])
            print(f"Refill option manually set to {refill}.")
            skip = True
        elif argument == "-bc":
            refill = int(arguments[idx + 1])
            print(f"Initial character for grouping manually set to {bc}.")
            skip = True
        elif argument == "-ec":
            refill = int(arguments[idx + 1])
            print(f"Final character for grouping manually set to {ec}.")
            skip = True
        elif argument == "-o" or argument == "-O":
            outname = str(arguments[idx + 1])
            print(
                f"Output filename set to {outname}. However, no redirection will take place because CLI input is required."
            )
            skip = True
        else:
            filename = str(arguments[idx])
            filenames.append(filename)
            terms.append(filename.split(".")[-1])
            print(f"Input filename set to {filename}.")
    dfs = check_input(terms, filenames, T, nd, imputer_strat, verb)
    if len(dfs) > 1:
        df = pd.concat(dfs)
    elif len(dfs) == 0:
        print("No input profiles detected in file. Exiting.")
        exit()
    else:
        df = dfs[0]
    assert isinstance(df, pd.DataFrame)
    if verb > 1:
        print("Final database :")
        print(df.head())
    return df, nd, verb, T, imputer_strat, refill, bc, ec


def check_input(terms, filenames, T, nd, imputer_strat, verb):
    invalid_input = False
    accepted_excel_terms = ["xls", "xlsx"]
    accepted_imputer_strats = ["simple", "none"]
    accepted_nds = [1, 2]
    dfs = []
    for term, filename in zip(terms, filenames):
        if term in accepted_excel_terms:
            dfs.append(pd.read_excel(filename))
        elif term == "csv":
            dfs.append(pd.read_csv(filename))
        else:
            print(
                f"File termination {term} was not understood. Try csv or one of {accepted_excel_terms}."
            )
            invalid_input = True
    if not isinstance(T, float):
        print("Invalid temperature input! Should be a float.")
        invalid_input = True
    if imputer_strat not in accepted_imputer_strats:
        print(
            f"Invalid imputer strat in input!\n Accepted values are:\n {accepted_imputer_strats}"
        )
        invalid_input = True
    if not isinstance(verb, int):
        print("Invalid verbosity input! Should be a positive integer or 0.")
        invalid_input = True
    if nd not in accepted_nds:
        print(f"Invalid number of descriptors in input!\n Accepted values ar:\n {nd}")
        invalid_input = True
    if invalid_input:
        exit()
    return dfs

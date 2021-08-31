#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
import itertools
import h5py
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


def user_choose_1_dv(dvs, r2s, tags):
    for dv, r2 in zip(dvs, r2s):
        print(
            f"\n{tags[dv]} has been identified as a suitable descriptor variable with r2={np.round(r2,4)}."
        )
        ok = yesno("Continue using this variable")
        if ok:
            return dv
    if not ok:
        manual = yesno("Would you want to use some other descriptor variable instead")
        if manual:
            for i, tag in enumerate(tags):
                ok = yesno(f"Use {tag} as descriptor")
                if ok:
                    return i
    return None


def user_choose_2_dv(ddvs, r2s, tags):
    tags = np.array(tags[1:-1], dtype=np.str)
    ptags = []
    for pair in itertools.combinations(tags, r=2):
        ptags.append([pair[0], pair[1]])
    for dv, r2 in zip(ddvs, r2s):
        print(
            f"\nThe combination of {tags[dv[0]]} and {tags[dv[1]]} has been identified as a suitable combined descriptor variable with r2={np.round(r2,4)}."
        )
        ok = yesno("Continue using this combined descriptor variable")
        if ok:
            return (dv[0] + 1, dv[1] + 1)
    if not ok:
        manual = yesno(
            "Would you want to use some other descriptor variable combination instead"
        )
        if manual:
            for i, ptag in enumerate(ptags):
                ok = yesno(f"Use combination of {ptag[0]} and {ptag[1]} as descriptor")
                if ok:
                    print(tags, ptag[0], ptag[1])
                    idx1 = np.where(tags == np.str(ptag[0]))[0][0] + 1
                    idx2 = np.where(tags == np.str(ptag[1]))[0][0] + 1
                    return idx1, idx2
    return None, None


def processargs(arguments):
    nd = 1
    T = 298.15
    imputer_strat = "none"
    verb = 0
    refill = False
    dump = False
    runmode = 3
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
            print(f"Number of descriptor variables manually set to {nd}.")
            skip = True
        elif argument == "-lsfer":
            runmode = 0
            print("Will only find and plot LSFERs.")
        elif argument == "-thermo":
            runmode = 1
            print("Will only build thermodynamic volcano.")
        elif argument == "-kinetic":
            runmode = 2
            print("Will only build kinetic volcanos.")
        elif argument == "-dump":
            dump = True
            print("Will dump volcano information in hdf5 file.")
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
            bc = int(arguments[idx + 1])
            print(f"Initial character for grouping manually set to {bc}.")
            skip = True
        elif argument == "-ec":
            ec = int(arguments[idx + 1])
            print(f"Final character for grouping manually set to {ec}.")
            skip = True
        elif argument == "-o" or argument == "-O":
            outname = str(arguments[idx + 1])
            print(
                f"Output filename set to {outname}. However, CLI input is currently required."
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
    return df, nd, verb, runmode, T, imputer_strat, refill, dump, bc, ec


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


def arraydump(path: str, descriptor: np.array, volcano_list, volcano_headers):
    """Dump array as an hdf5 file."""
    h5 = h5py.File(path, "w")
    assert len(volcano_list) == len(volcano_headers)
    # hdf5 file is like a dictionary, every dataset has a key and a data value (which can be an array)
    h5.create_dataset("Descriptor", data=descriptor)
    for i, j in zip(volcano_list, volcano_headers):
        h5.create_dataset(j, data=i)
    h5.close()


def arrayread(path: str):
    """Read hdf5 dataset."""
    h5 = h5py.File(path, "r")
    volcano_headers = []
    volcano_list = []
    for key in h5.keys():
        if key == "Descriptor":
            descriptor = h5[key]
        else:
            volcano_headers.append(key)
            volcano_list.append(h5[key][()])
    return descriptor, volcano_list, volcano_headers


def test_filedump():
    dv = np.linspace(0, 50, 1000)
    tv = np.linspace(-34, 13, 1000)
    kv = np.linspace(15, 25, 1000)
    volcano_list = [tv, kv]
    volcano_headers = ["Thermodynamic volcano", "Kinetic volcano"]
    arraydump("hdf5_test.hdf5", dv, volcano_list, volcano_headers)
    dv, volcano_list, volcano_headers = arrayread("hdf5_test.hdf5")
    assert np.allclose(
        tv, volcano_list[volcano_headers.index("Thermodynamic volcano")], 4
    )
    assert np.allclose(kv, volcano_list[volcano_headers.index("Kinetic volcano")], 4)
    os.remove("hdf5_test.hdf5")


if __name__ == "__main__":
    test_filedump()

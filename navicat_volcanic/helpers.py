#!/usr/bin/env python

import argparse
import itertools
import os
from itertools import cycle

import numpy as np
import pandas as pd

from navicat_volcanic.exceptions import InputError


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


def bround(x, base: float = 10, type=None) -> float:
    if type == "max":
        return base * np.ceil(x / base)
    elif type == "min":
        return base * np.floor(x / base)
    else:
        tick = base * np.round(x / base)
        return tick


def group_data_points(bc, ec, names):
    try:
        groups = np.array([str(i)[bc:ec].upper() for i in names], dtype=object)
    except Exception as m:
        raise InputError(
            f"Grouping by name characters did not work. Error message was:\n {m}"
        )
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
    tags = np.array(tags[1:], dtype=np.str_)
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
                    idx1 = np.where(tags == np.str_(ptag[0]))[0][0] + 1
                    idx2 = np.where(tags == np.str_(ptag[1]))[0][0] + 1
                    return idx1, idx2
    return None, None


def processargs(arguments):
    vbuilder = argparse.ArgumentParser(
        prog="volcanic",
        description="Build volcano plots from reaction energy profile data.",
        epilog="Remember to cite the volcanic paper: \n \nLaplaza, R., Das, S., Wodrich, M.D. et al. Constructing and interpreting volcano plots and activity maps to navigate homogeneous catalyst landscapes. Nat Protoc (2022). \nhttps://doi.org/10.1038/s41596-022-00726-2 \n \n - and enjoy!",
    )
    vbuilder.add_argument(
        "-version", "--version", action="version", version="%(prog)s 1.3.3"
    )
    runmode_arg = vbuilder.add_mutually_exclusive_group()
    vbuilder.add_argument(
        "-i",
        "--i",
        "-input",
        dest="filenames",
        nargs="?",
        action="append",
        type=str,
        required=True,
        help="Filename containing reaction energy data. See documentation for input and file formatting questions.",
    )
    vbuilder.add_argument(
        "-df",
        "--df",
        "-i2",
        "--i2",
        dest="dfilenames",
        action="append",
        type=str,
        default=[],
        help="Filename containing non-energy descriptors matching the reaction profiles. See documentation for input and file formatting questions.",
    )
    vbuilder.add_argument(
        "-nd",
        "--nd",
        dest="nd",
        type=int,
        default=1,
        help="Number of descriptor variables to use. (default: 1)",
    )
    vbuilder.add_argument(
        "-v",
        "--v",
        "--verb",
        dest="verb",
        type=int,
        default=0,
        help="Verbosity level of the code. Higher is more verbose and viceversa. Set to at least 2 to generate csv output files (default: 1)",
    )
    vbuilder.add_argument(
        "-r",
        "--r",
        dest="runmode",
        type=int,
        default=6,
        help="Defines the volcano plots to build.\n 0: LSRs\n 1: Thermodynamic\n 2: Kinetic\n 3: Energy Span\n 4: TOF\n 5: All of the above\n Other: Ask through CLI (default)",
    )
    runmode_arg.add_argument(
        "-lsfer",
        "--lsfer",
        "-lsr",
        "--lsr",
        dest="runmode",
        action="store_const",
        const=0,
        help="Set runmode to 0, building only linear scaling relationships.",
    )
    runmode_arg.add_argument(
        "-thermo",
        "--thermo",
        dest="runmode",
        action="store_const",
        const=1,
        help="Set runmode to 1, building only thermodynamic volcano.",
    )
    runmode_arg.add_argument(
        "-kinetic",
        "--kinetic",
        dest="runmode",
        action="store_const",
        const=2,
        help="Set runmode to 2, building only kinetic volcano.",
    )
    runmode_arg.add_argument(
        "-es",
        "--es",
        dest="runmode",
        action="store_const",
        const=3,
        help="Set runmode to 3, building only energy span volcano.",
    )
    runmode_arg.add_argument(
        "-tof",
        "--tof",
        dest="runmode",
        action="store_const",
        const=4,
        help="Set runmode to 4, building only TOF volcano.",
    )
    runmode_arg.add_argument(
        "-all",
        "--all",
        dest="runmode",
        action="store_const",
        const=5,
        help="Set runmode to 5, building all available volcanoes.",
    )
    vbuilder.add_argument(
        "-T",
        "-t",
        "--T",
        "--t",
        "--temp",
        "-temp",
        dest="temp",
        type=float,
        default=298.15,
        help="Temperature in K. (default: 298.15)",
    )
    vbuilder.add_argument(
        "-pm",
        "--pm",
        "-plotmode",
        "--plotmode",
        dest="plotmode",
        type=int,
        default=1,
        help="Plot mode for volcano and activity map plotting. Higher is more detailed, lower is basic. 3 includes uncertainties. (default: 1)",
    )
    vbuilder.add_argument(
        "-ic",
        "--ic",
        dest="ic",
        type=int,
        default=0,
        help="Initial character for grouping based on name for visualization. (default: 0)",
    )
    vbuilder.add_argument(
        "-fc",
        "--fc",
        dest="fc",
        type=int,
        default=2,
        help="Final character for grouping based on name for visualization. (default: 2)",
    )
    vbuilder.add_argument(
        "-rm",
        "--rm",
        dest="rmargin",
        type=int,
        default=10,
        help="Right margin to pad for visualization, in descriptor variable units. (default: 20)",
    )
    vbuilder.add_argument(
        "-lm",
        "--lm",
        dest="lmargin",
        type=int,
        default=10,
        help="Left margin to pad for visualization, in descriptor variable units. (default: 20)",
    )
    vbuilder.add_argument(
        "-np",
        "--np",
        dest="npoints",
        type=int,
        default=200,
        help="Number of grid points to use for visualization. (default: 200)",
    )
    vbuilder.add_argument(
        "-d",
        "--d",
        "-dump",
        "--dump",
        dest="dump",
        action="store_true",
        help="Flag to activate h5py dumping of data. (default: False)",
    )
    vbuilder.add_argument(
        "-is",
        "--is",
        dest="imputer_strat",
        type=str,
        default="none",
        help="Imputter to refill missing datapoints. Beta version. (default: None)",
    )
    vbuilder.add_argument(
        "-refill",
        "--refill",
        dest="refill",
        action="store_true",
        help="Refill missing values using LSRs. Beta version. (default: False)",
    )
    args = vbuilder.parse_args(arguments)

    dfs, ddfs = check_input(
        args.filenames,
        args.dfilenames,
        args.temp,
        args.nd,
        args.imputer_strat,
        args.verb,
    )
    if len(dfs) > 1:
        df = pd.concat(dfs)
    elif len(dfs) == 0:
        raise InputError("No input profiles detected in file. Exiting.")
    else:
        df = dfs[0]
    assert isinstance(df, pd.DataFrame)
    if args.verb > 1:
        print("Final reaction profile database (10 top rows):")
        print(df.head(10))

    if ddfs:
        if len(ddfs) > 1:
            ddf = pd.concat(ddfs)
        elif len(dfs) == 0:
            raise InputError("No valid descriptor files were provided. Exiting.")
        else:
            ddf = ddfs[0]
        assert isinstance(ddf, pd.DataFrame)
        if not (df.shape[0] == ddf.shape[0]):
            raise InputError(
                "Different number of entries in reaction profile input file and descriptor file. Exiting."
            )
        if args.verb > 1:
            print("Final descriptor database (top rows):")
            print(ddf.head())
        for column in ddf:
            df.insert(1, f"Descriptor {column}", ddf[column].values)
    return (
        df,
        args.nd,
        args.verb,
        args.runmode,
        args.temp,
        args.imputer_strat,
        args.refill,
        args.dump,
        args.ic,
        args.fc,
        args.lmargin,
        args.rmargin,
        args.npoints,
        args.plotmode,
    )


def check_input(filenames, dfilenames, temp, nd, imputer_strat, verb):
    accepted_excel_terms = ["xls", "xlsx"]
    accepted_imputer_strats = ["simple", "knn", "iterative", "none"]
    accepted_nds = [1, 2]
    dfs = []
    ddfs = []
    for filename in filenames:
        if filename.split(".")[-1] in accepted_excel_terms:
            dfs.append(pd.read_excel(filename))
        elif filename.split(".")[-1] == "csv":
            dfs.append(pd.read_csv(filename))
        else:
            raise InputError(
                f"File termination for filename {filename} was not understood. Try csv or one of {accepted_excel_terms}."
            )
    for dfilename in dfilenames:
        if dfilename.split(".")[-1] in accepted_excel_terms:
            dfs.append(pd.read_excel(dfilename))
        elif dfilename.split(".")[-1] == "csv":
            dfs.append(pd.read_csv(dfilename))
        else:
            raise InputError(
                f"File termination for filename {dfilename} was not understood. Try csv or one of {accepted_excel_terms}."
            )
    if not isinstance(temp, float):
        raise InputError("Invalid temperature input! Should be a float. Exiting.")
    if imputer_strat not in accepted_imputer_strats:
        raise InputError(
            f"Invalid imputer strat in input!\n Accepted values are:\n {accepted_imputer_strats}"
        )
    if not isinstance(verb, int):
        raise InputError("Invalid verbosity input! Should be a positive integer or 0.")
    if nd not in accepted_nds:
        raise InputError(
            f"Invalid number of descriptors in input!\n Accepted values ar:\n {nd}"
        )
    return dfs, ddfs


def arraydump(path: str, descriptor: np.array, volcano_list, volcano_headers):
    """Dump array as an hdf5 file."""
    import h5py

    h5 = h5py.File(path, "w")
    assert len(volcano_list) == len(volcano_headers)
    # hdf5 file is like a dictionary, every dataset has a key and a data value (which can be an array)
    h5.create_dataset("Descriptor", data=descriptor)
    for i, j in zip(volcano_list, volcano_headers):
        h5.create_dataset(j, data=i)
    h5.close()


def arrayread(path: str):
    """Read hdf5 dataset."""
    import h5py

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


def setflags(runmode):
    if runmode == 0:
        t_volcano = False
        k_volcano = False
        es_volcano = False
        tof_volcano = False
    elif runmode == 1:
        t_volcano = True
        k_volcano = False
        es_volcano = False
        tof_volcano = False
    elif runmode == 2:
        t_volcano = False
        k_volcano = True
        es_volcano = False
        tof_volcano = False
    elif runmode == 3:
        t_volcano = False
        k_volcano = False
        es_volcano = True
        tof_volcano = False
    elif runmode == 4:
        t_volcano = False
        k_volcano = False
        es_volcano = False
        tof_volcano = True
    elif runmode == 5:
        t_volcano = True
        k_volcano = True
        es_volcano = True
        tof_volcano = True
    else:
        t_volcano = yesno("Generate thermodynamic volcano plot")
        k_volcano = yesno("Generate kinetic volcano plot")
        es_volcano = yesno("Generate energy span volcano plot")
        tof_volcano = yesno("Generate TOF volcano plot")
    return t_volcano, k_volcano, es_volcano, tof_volcano


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

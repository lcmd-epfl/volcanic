#!/usr/bin/env python

import pandas as pd


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


def processargs(arguments):
    T = 298.15
    imputer_strat = "none"
    verb = 0
    refill = False
    bc = 0
    ec = 2
    terms = []
    filenames = []
    for idx, i in enumerate(arguments):
        if i == "-i":
            filename = str(arguments.pop(idx + 1))
            filenames.append(filename)
            terms.append(filename.split(".")[-1])
            print(f"Input filename set to {filename}.")
        if i == "-t":
            T = float(arguments.pop(idx + 1))
            print(f"Temperature manually set to {T}.")
        if i == "-v":
            verb = int(arguments.pop(idx + 1))
            print(f"Verbosity manually set to {verb}.")
        if i == "-is":
            imputer_strat = str(arguments.pop(idx + 1))
            print(f"Imputer strategy manually set to {imputer_strat}.")
        if i == "-re":
            refill = bool(arguments.pop(idx + 1))
            print(f"Refill option manually set to {refill}.")
        if i == "-bc":
            refill = int(arguments.pop(idx + 1))
            print(f"Initial character for grouping manually set to {bc}.")
        if i == "-ec":
            refill = int(arguments.pop(idx + 1))
            print(f"Final character for grouping manually set to {ec}.")
        else:
            filename = str(arguments.pop(idx))
            filenames.append(filename)
            terms.append(filename.split(".")[-1])
            print(f"Input filename set to {filename}.")
    dfs = check_input(terms, filenames, T, imputer_strat, verb)
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
    return df, verb, T, imputer_strat, refill, bc, ec


def check_input(terms, filenames, T, imputer_strat, verb):
    accepted_excel_terms = ["xls", "xlsx"]
    accepted_imputer_strats = ["simple", "none"]
    dfs = []
    for term, filename in zip(terms, filenames):
        if term in accepted_excel_terms:
            dfs.append(pd.read_excel(filename))
        if term == "csv":
            dfs.append(pd.read_csv(filename))
        else:
            print(
                f"File termination {term} was not understood. Try csv or one of {accepted_excel_terms}."
            )
    if not isinstance(T, float):
        print("Invalid temperature input! Should be a float.")
    if imputer_strat not in accepted_imputer_strats:
        print(
            f"Invalid imputer strat in input!\n Accepted values are:\n {accepted_imputer_strats}"
        )
    if not isinstance(verb, int):
        print("Invalid verbosity input! Should be an input.")
    return dfs

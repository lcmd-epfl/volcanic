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
    excel_terms = ["xlsx", "xls"]
    T = 298.15
    imputer_strat = "none"
    dfs = []
    verb = 0
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
            if i == "-is":
                imputer_strat = str(arguments[idx + 1])
                print(f"Imputer strategy manually set to {imputer_strat}.")
    check_input(excel_terms, T, imputer_strat, dfs, verb)
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
    return df, verb, T, imputer_strat


def check_input(excel_terms, T, imputer_strat, dfs, verb):
    accepted_excel_terms = ["csv", "xls", "lsx"] 
    accepted_imputer_strats = ["simple", "none"] 
    try :
        assert excel_terms in accepted_excel_terms
    except AssertionError :
        print(f"Invalid file termination in input!\n Accepted values are:\n {accepted_excel_terms}")
    try :
        assert isinstance(T,float)
    except AssertionError :
        print("Invalid temperature input! Should be a float.")
    try :
        assert imputer_strat in accepted_imputer_strats
    except AssertionError :
        print(f"Invalid imputer strat in input!\n Accepted values are:\n {accepted_imputer_strats}")
    try :
        assert isinstance(verb, int)
    except AssertionError :
        print("Invalid verbosity input! Should be an input.")





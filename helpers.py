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
    verb = 0
    T = 298.15
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
    elif len(dfs) == 0:
        print("No input profiles detected. Exiting.")
        exit()
    else:
        df = dfs[0]
    assert isinstance(df, pd.DataFrame)
    if verb > 1:
        print("Final database :")
        print(df.head())
    return df, verb, T

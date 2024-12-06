volcanic: Automated Generator of Volcano Plots
==============================================
[![DOI](https://zenodo.org/badge/381737392.svg)](https://zenodo.org/badge/latestdoi/381737392)

![volcanic logo](./images/volcanic_logo.png)
[![PyPI version](https://badge.fury.io/py/navicat_volcanic.svg)](https://badge.fury.io/py/navicat_volcanic)

## Contents
* [About](#about-)
* [Install](#install-)
* [Examples](#examples-)
* [Citation](#citation-)

## About [↑](#about)

The code runs on pure python with the following dependencies: 
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`
- `scikit-learn`
- Optionally, `h5py`

## Install [↑](#install)

You can install volcanic using pip:

```python
pip install navicat_volcanic
```

Afterwards, you can call volcanic as:

```python
python -m navicat_volcanic [-h] [-version] -i [FILENAMES] [-df DFILENAMES] [-nd ND] [-v VERB] [-r RUNMODE] [-lsfer | -thermo | -kinetic | -es | -tof | -all] [-T TEMP] [-pm PLOTMODE] [-ic IC] [-fc FC]
                [-rm RMARGIN] [-lm LMARGIN] [-np NPOINTS] [-d] [-is IMPUTER_STRAT] [-refill]
```
or simply
```python
navicat_volcanic [-h] [-version] -i [FILENAMES] [-df DFILENAMES] [-nd ND] [-v VERB] [-r RUNMODE] [-lsfer | -thermo | -kinetic | -es | -tof | -all] [-T TEMP] [-pm PLOTMODE] [-ic IC] [-fc FC]
                [-rm RMARGIN] [-lm LMARGIN] [-np NPOINTS] [-d] [-is IMPUTER_STRAT] [-refill]
```

Alternatively, you can download the package and execute:

```python 
python setup.py install
```

Afterwards, you can call volcanic as:

```python
python -m navicat_volcanic [-h] [-version] -i [FILENAMES] [-df DFILENAMES] [-nd ND] [-v VERB] [-r RUNMODE] [-lsfer | -thermo | -kinetic | -es | -tof | -all] [-T TEMP] [-pm PLOTMODE] [-ic IC] [-fc FC]
                [-rm RMARGIN] [-lm LMARGIN] [-np NPOINTS] [-d] [-is IMPUTER_STRAT] [-refill]
```
or
```python
navicat_volcanic [-h] [-version] -i [FILENAMES] [-df DFILENAMES] [-nd ND] [-v VERB] [-r RUNMODE] [-lsfer | -thermo | -kinetic | -es | -tof | -all] [-T TEMP] [-pm PLOTMODE] [-ic IC] [-fc FC]
                [-rm RMARGIN] [-lm LMARGIN] [-np NPOINTS] [-d] [-is IMPUTER_STRAT] [-refill]
```


Options can be consulted using the `-h` flag in either case. The help menu is quite detailed. 

Note that the volcano plot and activity map functions are directly exposed in `volcanic.py` as `volcanic_2d` and `volcanic_3d` respectively, in case you want to incorporate them in your own code.

## Examples [↑](#examples)

The examples subdirectory contains a copious amount of tests which double as examples. Any of the data files can be run as:

```python
python -m navicat_volcanic -i [FILENAME]
```

This will query the user for options and generate the volcano plots as png images. Options can be consulted with the `-h` flag.

The input of volcanic is a `pandas` compatible dataframe, which includes plain .csv and .xls files. 

Regarding format, volcanic expects headers for all columns. The first column must contain names/identifiers. Then, volcanic expects a number of columns with relative free energies for the species in the catalytic cycle (in order of appearance), whose headers must contain "TS" if the species is a transition state, and a final column whose header is "Product" containing the reaction energy. Non-energy descriptors can be input as a separate file using the `-df` flag or as extra columns whose headers contain the word "Descriptor".

High verbosity levels (`-v 1`, `-v 2`, etc.) will print the output as .csv files as well, which can be used to plot your volcano plot or activity map using external tools. An example is found in the `pretty_plotting_example` directory in this repository. Keep increasing the verbosity to get even more detailed output. This can be useful for understanding what the code is doing and what can have possibly gone wrong. To be as automated as possible, reasonable default values are set for most choices. The generated csvs also contain the 95% confidence interval value for the plot in question, which are obtained by propagating the uncertainty of the LSFERs involved in the computation. To simplify the propagation of uncertainties in the TOF computation, the uncertainty is approximated using the energy span in TOF volcano plots. 

The plotmode (`-pm 1`, `-pm 2`) option can be used to modify the default look of the generated pngs, including more detail as the plotmode level increases. 


## Citation [↑](#citation)

Please cite the accompanying manuscript, which clarifies the details of volcano plot construction. You can find it [here](https://rdcu.be/cT7uu) and in the reference:

```
Laplaza, R., Das, S., Wodrich, M.D. et al. Constructing and interpreting volcano plots and activity maps to navigate homogeneous catalyst landscapes. Nat Protoc (2022). https://doi.org/10.1038/s41596-022-00726-2
```

Kinetic volcano plots were introduced [here](https://doi.org/10.1039/C6SC01660J) and turnover frequency/energy span volcano plots were introduced [here](https://doi.org/10.1021/acscatal.9b00717). The energy span model was introduced by Kozuch and Shaik [here](https://doi.org/10.1021/ar1000956). Please include those citations where relevant. A comprehensive account can be found [here](https://doi.org/10.1021/acs.accounts.0c00857).


---



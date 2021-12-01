volcanic: Automated Generator of Volcano Plots
==============================================

![volcanic logo](./images/volcanic_logo.png)

## Contents
* [About](#about-)
* [Install](#install-)
* [Examples](#examples-)

## About [↑](#about)

The code runs on pure python with minimal dependencies: 
- `numpy`
- `scipy`
- `matplotlib`
- `pandas`

To use

## Install [↑](#install)

Download and add volcanic.py to your path. No strings attached. Run as:

```python
python volcanic.py [-h] [-version] -i [FILENAMES] [-df DFILENAMES] [-nd ND] [-v VERB] [-r RUNMODE] [-lsfer | -thermo | -kinetic | -es | -tof | -all] [-T TEMP] [-pm PLOTMODE] [-ic IC] [-fc FC]
                [-rm RMARGIN] [-lm LMARGIN] [-np NPOINTS] [-d] [-is IMPUTER_STRAT] [-refill]
```

You can also execute:

```python 
python setup.py install
```

to install volcanic as a python module. Afterwards, you can call volcanic as:

```python 
python -m volcanic [-h] [-version] -i [FILENAMES] [-df DFILENAMES] [-nd ND] [-v VERB] [-r RUNMODE] [-lsfer | -thermo | -kinetic | -es | -tof | -all] [-T TEMP] [-pm PLOTMODE] [-ic IC] [-fc FC]
                [-rm RMARGIN] [-lm LMARGIN] [-np NPOINTS] [-d] [-is IMPUTER_STRAT] [-refill]
```

Options can be consulted using the `-h` flag.

## Examples [↑](#examples)

The examples subdirectory contains a copious amount of tests which double as examples. Any of the data files can be run as:

```python
python volcanic.py -i [FILENAME]
```

This will query the user for options and generate the volcano plots as png images. Options can be consulted with the `-h` flag.

---



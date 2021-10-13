#!/usr/bin/env python

import numpy as np


class InputError(Exception):
    """Raised when there is an error in the input."""

    pass


class MissingDataError(Exception):
    """Raised when too many missing energies are found in the input. Setting an inputter might fix this."""

    pass

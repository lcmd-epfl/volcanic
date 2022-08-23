#!/usr/bin/env python

from volcanic.dv1 import test_dv1, test_imputer
from volcanic.dv2 import test_dv2
from volcanic.helpers import test_filedump
from volcanic.tof import test_aryl_ether_cleavage, test_tof

test_dv1()
test_dv2()
test_tof()
test_imputer()
test_filedump()
test_aryl_ether_cleavage()


if __name__ == "__main__":
    test_dv1()
    test_dv2()
    test_tof()
    test_imputer()
    test_filedump()
    test_aryl_ether_cleavage()

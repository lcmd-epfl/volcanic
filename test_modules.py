#!/usr/bin/env python

from dv1 import test_dv1, test_inputer
from dv2 import test_dv2
from tof import test_tof, test_aryl_ether_cleavage
from helpers import test_filedump

test_dv1()
test_dv2()
test_tof()
test_inputer()
test_filedump()
test_aryl_ether_cleavage()


if __name__ == "__main__":
    test_dv1()
    test_dv2()
    test_tof()
    test_inputer()
    test_filedump()
    test_aryl_ether_cleavage()

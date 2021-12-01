#!/usr/bin/env python

from volcanic.dv1 import test_dv1, test_inputer
from volcanic.dv2 import test_dv2
from volcanic.tof import test_tof, test_aryl_ether_cleavage
from volcanic.helpers import test_filedump

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

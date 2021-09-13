#!/usr/bin/env python

from dv1 import test_dv1
from dv2 import test_dv2
from tof import test_tof, test_aryl_ether_cleavage


test_dv1()
test_dv2()
test_tof()
test_aryl_ether_cleavage()


if __name__ == "__main__":
    test_dv1()
    test_dv2()
    test_tof()
    test_aryl_ether_cleavage()

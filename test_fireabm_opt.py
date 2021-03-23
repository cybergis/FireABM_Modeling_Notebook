#!/usr/bin/env python3
# test_fireabm_opt.py

import pytest
import mock
import FireABM_opt

# check functions in FireABM_opt (core file for simulation)
# TEST the helper functions used in the simulation

@mock.patch('FireABM_opt.seed_number', 16)
def test_set_seed(capsys):
    from FireABM_opt import check_seed
    check_seed()
    captured = capsys.readouterr()
    assert captured.out == "16\n"

# convert miles per hour to meters per second
def test_mph2ms():
    from FireABM_opt import mph2ms
    assert mph2ms(15) == 6.7056

# get squared piecewise distance between 2 points
def test_dist2():
    from FireABM_opt import dist2
    assert dist2([1, 1], [1, 1]) == 0
    assert dist2([1, 1], [1, 2]) == 1
    assert dist2([1, 1], [2, 2]) == 2
    assert dist2([1, 1], [3, 2]) == 5


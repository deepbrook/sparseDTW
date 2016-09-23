"""
Task:
Descripion of script here.
"""

# Import Built-Ins
import logging
from unittest import TestCase
# Import Third-Party
import numpy as np
# Import Homebrew

import sparse_dtw
# Init Logging Facilities
log = logging.getLogger(__name__)


class Sparse_dtwTest(TestCase):

    def setUp(self):
        self.s = [3, 4, 5, 3, 3]
        self.q = [1, 2, 2, 1, 0]

    def test_quantize(self):
        a, b = sparse_dtw.quantize(self.s, self.q)
        check_a = np.array([0.0, 0.5, 1.0, 0.0, 0.0])
        check_b = np.array([0.5, 1.0, 1.0, 0.5, 0.0])
        self.assertTrue(np.array_equal(a, check_a))
        self.assertTrue(np.array_equal(b, check_b))

    def test_euc_dist(self):
        self.assertEqual(sparse_dtw.euc_distance(5,7), 4)

    def test_populate_warp(self):
        a = sparse_dtw.populate_warp(self.s, self.q, 0.5)
        b = np.array([[4, 0, 0, 4, 9],
                      [9, 4, 4, 9, 16],
                      [16, 9, 9, 16, 0],
                      [4, 0, 0, 4, 9],
                      [4, 0, 0, 4, 9]])
        self.assertTrue(np.array_equal(a, b))

    def test_unblock_warp_matrix(self):
        SM = sparse_dtw.populate_warp(self.s, self.q, 0.5)
        a = sparse_dtw.calculat_warp_costs(self.s, self.q, SM)
        b = np.array([[4, 0, 0, 4, 13],
                      [13, 8, 12, 13, 20],
                      [29, 17, 17, 28, 38],
                      [33, 0, 0, 21, 30],
                      [37, 34, 35, 25, 30]])
        self.assertTrue(np.array_equal(a, b))

    def test_sparsesparse_dtw(self):
        sparsed = sparse_dtw.sparse_dtw(self.s, self.q, res=0.5)
        check = [(4, 4), (3, 3), (2, 2), (1, 1), (0, 0)]
        check.reverse()
        self.assertTrue(sparsed == (check, 30))



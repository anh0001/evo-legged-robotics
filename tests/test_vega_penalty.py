import unittest
import numpy as np
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.evolution.vega import VEGA

class TestCycleClosurePenalty(unittest.TestCase):
    def test_penalty_reduces_smoothness(self):
        vega = VEGA(population_size=1, chromosome_length=3, generations=1)
        idx = 0
        vega.host_lengths[idx] = 2
        vega.hosts[idx, 0] = np.zeros((2, vega.dof))
        vega.hosts[idx, 1] = np.ones((2, vega.dof)) * vega.q_range

        penalty = vega._cycle_closure_penalty(idx)
        self.assertGreater(penalty, 0)

        base = 1.0
        penalized = base * math.exp(-penalty * 2.0)
        self.assertLess(penalized, base)

if __name__ == "__main__":
    unittest.main()

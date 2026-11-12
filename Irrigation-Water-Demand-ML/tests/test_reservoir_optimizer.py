import importlib.util
from pathlib import Path
import unittest
import numpy as np

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "reservoir_optimizer.py"
spec = importlib.util.spec_from_file_location("reservoir_optimizer", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

class ReservoirOptimizerTests(unittest.TestCase):
    def test_full_supply_meets_demand(self):
        demand = np.array([100.0, 200.0, 150.0])
        release, z = module.optimize_releases(demand, 450.0, 500.0)
        self.assertTrue(np.allclose(release, demand, atol=1e-6))
        self.assertAlmostEqual(z, 0.0, places=7)

    def test_shortage_respects_budget(self):
        demand = np.array([100.0, 200.0, 100.0])
        release, z = module.optimize_releases(demand, 300.0, 500.0)
        self.assertLessEqual(release.sum(), 300.0 + 1e-6)
        self.assertTrue(np.all(release <= demand + 1e-6))
        self.assertGreater(z, 0.0)

    def test_daily_cap_respected(self):
        demand = np.array([100.0, 200.0, 100.0])
        release, _ = module.optimize_releases(demand, 400.0, 120.0)
        self.assertTrue(np.all(release <= 120.0 + 1e-6))

if __name__ == "__main__":
    unittest.main()

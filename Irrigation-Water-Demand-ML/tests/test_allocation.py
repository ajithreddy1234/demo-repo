import importlib.util
from pathlib import Path
import unittest

MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "allocation_from_iwr.py"
spec = importlib.util.spec_from_file_location("allocation_from_iwr", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
allocation_from_iwr = module.allocation_from_iwr

class AllocationTests(unittest.TestCase):
    def test_one_mm_one_hectare(self):
        out = allocation_from_iwr(1.0, 1.0)
        self.assertAlmostEqual(out["net_m3_per_ha_day"], 10.0)
        self.assertAlmostEqual(out["net_command_area_m3_day"], 10.0)

    def test_area_scaling(self):
        out = allocation_from_iwr(4.0, 100.0)
        self.assertAlmostEqual(out["net_m3_per_ha_day"], 40.0)
        self.assertAlmostEqual(out["net_command_area_m3_day"], 4000.0)

    def test_gross_release_efficiency(self):
        out = allocation_from_iwr(4.0, 100.0, 0.8, 0.75)
        self.assertAlmostEqual(out["overall_efficiency"], 0.6)
        self.assertAlmostEqual(out["gross_reservoir_release_m3_day"], 4000.0 / 0.6)

    def test_negative_iwr_rejected(self):
        with self.assertRaises(ValueError):
            allocation_from_iwr(-1.0, 1.0)

if __name__ == "__main__":
    unittest.main()

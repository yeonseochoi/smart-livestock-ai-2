import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "outputs/odor_service/v2vars"


class TestV2VarsArtifacts(unittest.TestCase):
    def test_files_and_columns(self):
        for name in ("events.csv", "terciles.csv", "correlations.csv", "ablation.csv"):
            self.assertTrue((ROOT / name).is_file(), name)
        required = {
            "events.csv": {"k", "event_id", "growth", "intensity_mean", "intensity_max", "future_reports"},
            "ablation.csv": {"k", "모델", "제외 변수", "Hit@3", "전체 대비 차이", "차이 95% 하한", "차이 95% 상한"},
        }
        for name, columns in required.items():
            self.assertTrue(columns <= set(pd.read_csv(ROOT / name, nrows=1).columns))


if __name__ == "__main__":
    unittest.main()

"""Track A 산출물 계약 검증기 자체와 fixture, 그리고 실제 산출물(있을 때)을 검사한다."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import validate_backtrack_output as v  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "grid_scores_sample.csv"
REAL = ROOT / "outputs" / "source_backtrack" / "grid_scores.csv"


class BacktrackContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.reference, cls.meta = v.load_reference()
        cls.fixture = pd.read_csv(FIXTURE, encoding="utf-8-sig")

    def test_fixture_passes(self) -> None:
        failures, summary = v.validate(self.fixture, self.reference, self.meta, min_coverage=0)
        self.assertEqual(failures, [], msg="\n".join(failures))
        self.assertEqual(summary["events"], 3)

    def test_detects_wrong_columns(self) -> None:
        broken = self.fixture.drop(columns=["history_cutoff"])
        failures, _ = v.validate(broken, self.reference, self.meta, min_coverage=0)
        self.assertTrue(any("컬럼" in f for f in failures))

    def test_detects_duplicate_key(self) -> None:
        broken = pd.concat([self.fixture, self.fixture.iloc[[0]]], ignore_index=True)
        failures, _ = v.validate(broken, self.reference, self.meta, min_coverage=0)
        self.assertTrue(any("중복" in f for f in failures))

    def test_detects_leak(self) -> None:
        broken = self.fixture.copy()
        broken.loc[0, "history_cutoff"] = str(pd.Timestamp(broken.loc[0, "event_hour"]) + pd.Timedelta(hours=1))
        failures, _ = v.validate(broken, self.reference, self.meta, min_coverage=0)
        self.assertTrue(any("누수" in f for f in failures))

    def test_detects_shifted_grid_origin(self) -> None:
        broken = self.fixture.copy()
        broken["center_latitude"] = broken["center_latitude"] + 0.001  # 약 110 m
        failures, _ = v.validate(broken, self.reference, self.meta, min_coverage=0)
        self.assertTrue(any("중심 좌표" in f for f in failures))

    def test_detects_missing_candidate_cells(self) -> None:
        broken = self.fixture.iloc[1:].copy()
        failures, _ = v.validate(broken, self.reference, self.meta, min_coverage=0)
        self.assertTrue(any("빠진 칸" in f for f in failures))

    def test_detects_unnormalized_scores(self) -> None:
        broken = self.fixture.copy()
        broken["source_fit_score"] = broken["source_fit_score"] * 0.5
        failures, _ = v.validate(broken, self.reference, self.meta, min_coverage=0)
        self.assertTrue(any("정규화" in f for f in failures))

    @unittest.skipUnless(REAL.exists(), "Track A 산출물이 아직 없음")
    def test_real_output_passes(self) -> None:
        real = pd.read_csv(REAL, encoding="utf-8-sig")
        failures, summary = v.validate(real, self.reference, self.meta)
        self.assertEqual(failures, [], msg="\n".join(failures))
        print("\nTrack A 산출물 요약:", summary)


if __name__ == "__main__":
    unittest.main()

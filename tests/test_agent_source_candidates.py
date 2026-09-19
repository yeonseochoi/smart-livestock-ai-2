"""행정 대응 문서의 발생원 후보 절: 후보가 있을 때만 나타나고, 원인 단정 표현이 없고, event_hour로 결합된다."""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from administrative_agent.models import ForecastResult, RiskArea, SourceCandidate  # noqa: E402
from administrative_agent.service import build_response_package  # noqa: E402
import generate_agent_documents as generate  # noqa: E402


def _forecast(**extra) -> ForecastResult:
    return ForecastResult(
        event_id="EVT-TEST", event_time=datetime(2024, 7, 28, 23, 0), forecast_minutes=30, grid_size_m=1000,
        areas=tuple(RiskArea(i, f"G{i}", score) for i, score in enumerate((100, 80, 60), 1)),
        model_metrics={"top_k_recall": .5, "roc_auc": .9, "pr_auc": .48}, generated_at=datetime(2024, 7, 28, 23, 1),
        **extra,
    )


class SourceCandidateDocumentTest(unittest.TestCase):
    def test_section_absent_without_candidates(self) -> None:
        package = build_response_package(_forecast())
        for text in (package.briefing, package.dispatch_order, package.response_guide):
            self.assertNotIn("발생원 후보", text)
        self.assertIn("## 4. 상황 판단", package.briefing)

    def test_section_present_with_candidates_and_no_causal_claim(self) -> None:
        candidates = (
            SourceCandidate(1, "김제시 용지면 ○○리 축산 밀집 구역(12곳)", city="김제시", species="돼지",
                            location_precision="village", distance_km=6.2, bearing_deg=45.0,
                            travel_time_min=52.0, wind_alignment=0.91, fit_score=1.0,
                            evidence_text="남서풍 2.0m/s, 도달시간 약 52분, 민원 3칸과 방위 일치"),
            SourceCandidate(2, "익산시 왕궁면 돼지 농가", city="익산시", species="돼지", location_precision="point",
                            distance_km=3.1, bearing_deg=120.0, travel_time_min=26.0, wind_alignment=0.55, fit_score=0.62),
        )
        package = build_response_package(_forecast(
            source_candidates=candidates, backtrack_uncertainty=0.41, backtrack_weather_source="asos"))
        self.assertIn("## 4. 발생원 후보 (참고)", package.briefing)
        self.assertIn("## 5. 상황 판단", package.briefing)
        self.assertIn("[리 단위 추정]", package.briefing)
        self.assertIn("중간(0.41)", package.briefing)
        self.assertIn("상풍측 발생원 후보", package.dispatch_order)
        self.assertIn("시설 특정이 아닙니다", package.response_guide)
        for text in (package.briefing, package.dispatch_order, package.response_guide):
            self.assertNotIn("원인 시설로", text)
            self.assertNotIn("원인으로 판정", text)

    def test_loader_joins_on_event_hour_not_event_id(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source_candidates.csv"
            pd.DataFrame({
                "event_hour": ["2024-07-28 23:00:00"] * 2 + ["2021-07-14 21:00:00"],
                "rank": [2, 1, 1], "source_id": ["S2", "S1", "S3"], "name": ["B", "A", "C"],
                "city": ["익산시"] * 3, "species": ["돼지"] * 3, "location_precision": ["point"] * 3,
                "distance_km": [3.0, 5.0, 1.0], "bearing_deg": [90.0, 45.0, 0.0], "travel_time_min": [30.0, 50.0, 10.0],
                "wind_alignment": [0.5, 0.9, 0.1], "fit_score": [0.6, 1.0, 1.0], "evidence_text": ["", "", ""],
            }).to_csv(path, index=False, encoding="utf-8-sig")
            loaded = generate.load_source_candidates(path, pd.Timestamp("2024-07-28 23:00"))
        self.assertEqual([c.name for c in loaded], ["A", "B"])
        self.assertIsNone(loaded[0].evidence_text)
        self.assertEqual(generate.load_source_candidates(Path(folder) / "missing.csv", pd.Timestamp("2024-07-28 23:00")), ())


if __name__ == "__main__":
    unittest.main()

"""행정 대응 문서의 사전 경보 절: 발생 위험 격자가 있을 때만 나타나고, 확률·원인 단정 표현이 없고, event_hour-1h로 결합된다."""
from __future__ import annotations

import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from administrative_agent.models import ForecastResult, OnsetAlertCell, RiskArea, SourceCandidate  # noqa: E402
from administrative_agent.service import build_response_package  # noqa: E402
import generate_agent_documents as generate  # noqa: E402


def _forecast(**extra) -> ForecastResult:
    return ForecastResult(
        event_id="EVT-TEST", event_time=datetime(2025, 7, 28, 23, 0), forecast_minutes=30, grid_size_m=1000,
        areas=tuple(RiskArea(i, f"G+{i}:+0", score) for i, score in enumerate((100, 80, 60), 1)),
        model_metrics={"top_k_recall": .5, "roc_auc": .9, "pr_auc": .48}, generated_at=datetime(2025, 7, 28, 23, 1),
        **extra,
    )


class OnsetAlertDocumentTest(unittest.TestCase):
    def test_section_absent_without_alerts(self) -> None:
        package = build_response_package(_forecast())
        self.assertNotIn("사전 경보", package.briefing)
        self.assertIn("## 4. 상황 판단", package.briefing)

    def test_section_numbering_and_overlap_mark(self) -> None:
        alerts = (
            OnsetAlertCell(1, "G+1:+0", 100, upwind_share=0.62, quiet_hour=True),
            OnsetAlertCell(2, "G+7:-3", 71, upwind_share=0.10, quiet_hour=True),
        )
        package = build_response_package(_forecast(onset_alerts=alerts, onset_reference_time=datetime(2025, 7, 28, 22, 0)))
        self.assertIn("## 4. 사전 경보 (참고)", package.briefing)
        self.assertIn("## 5. 상황 판단", package.briefing)
        self.assertIn("(확산 예측 Top 3와 일치)", package.briefing)
        self.assertIn("62%", package.briefing)
        self.assertIn("22:00", package.briefing)
        self.assertNotIn("발생 확률은", package.briefing)
        self.assertNotIn("원인 시설로", package.briefing)
        # 발생원 후보와 함께 있으면 4→5→6 순으로 번호가 이어진다.
        candidate = SourceCandidate(1, "익산시 왕궁면 돼지 농가", city="익산시", species="돼지", location_precision="point",
                                    distance_km=3.1, bearing_deg=120.0, travel_time_min=26.0, wind_alignment=0.55, fit_score=0.62)
        both = build_response_package(_forecast(onset_alerts=alerts, source_candidates=(candidate,)))
        self.assertIn("## 4. 발생원 후보 (참고)", both.briefing)
        self.assertIn("## 5. 사전 경보 (참고)", both.briefing)
        self.assertIn("## 6. 상황 판단", both.briefing)

    def test_type_lines_only_when_present(self) -> None:
        alerts = (OnsetAlertCell(1, "G+1:+0", 100, quiet_hour=True),)
        plain = build_response_package(_forecast(onset_alerts=alerts))
        self.assertNotIn("냄새 유형별 위험", plain.briefing)
        by_type = {"가축": (OnsetAlertCell(1, "G+2:+1", 100), OnsetAlertCell(2, "G+3:+1", 80)),
                   "공장": (OnsetAlertCell(1, "G+1:+0", 100),)}
        typed = build_response_package(_forecast(onset_alerts=alerts, onset_alerts_by_type=by_type))
        self.assertIn("- 가축 냄새: G+2:+1(100), G+3:+1(80)", typed.briefing)
        self.assertIn("- 공장 냄새: G+1:+0(100)", typed.briefing)
        self.assertNotIn("하수 냄새", typed.briefing)

    def test_loader_uses_hour_before_event(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "onset_alerts.csv"
            pd.DataFrame({
                "hour": ["2025-07-28 22:00:00"] * 2 + ["2025-07-28 23:00:00"],
                "rank": [1, 2, 1], "grid_x": [1, 7, 9], "grid_y": [0, -3, 9],
                "center_latitude": [35.9, 35.8, 35.7], "center_longitude": [126.9, 126.8, 126.7],
                "relative_risk": [100, 71, 100], "upwind_share_eea": [0.62, 0.1, 0.5], "city_quiet_3h": [1, 1, 0],
            }).to_csv(path, index=False, encoding="utf-8-sig")
            cells, reference = generate.load_onset_alerts(path, pd.Timestamp("2025-07-28 23:00"))
            self.assertEqual([c.grid_id for c in cells], ["G+1:+0", "G+7:-3"])
            self.assertEqual(reference, pd.Timestamp("2025-07-28 22:00"))
            self.assertTrue(cells[0].quiet_hour)
            none_cells, none_reference = generate.load_onset_alerts(path, pd.Timestamp("2025-07-29 03:00"))
            self.assertEqual(none_cells, ())
            self.assertIsNone(none_reference)
            self.assertEqual(generate.load_onset_alerts(Path(folder) / "missing.csv", pd.Timestamp("2025-07-28 23:00")), ((), None))


if __name__ == "__main__":
    unittest.main()

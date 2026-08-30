from datetime import datetime
import json
import os
import unittest
from unittest.mock import Mock, patch

from administrative_agent.llm import llm_configured, provider_name, refine_with_gemini
from administrative_agent.models import ForecastResult, RiskArea
from administrative_agent.service import build_response_package, create_completed_followup


class AdministrativeAgentTest(unittest.TestCase):
    def setUp(self):
        self.forecast = ForecastResult(
            event_id="EVT-TEST", event_time=datetime(2026, 8, 23, 14, 0),
            forecast_minutes=30, grid_size_m=1000,
            areas=tuple(RiskArea(i, f"G{i}", score) for i, score in enumerate((100, 80, 60), 1)),
            model_metrics={"top_k_recall": .494, "roc_auc": .908, "pr_auc": .472},
            generated_at=datetime(2026, 8, 23, 14, 1),
        )

    def test_creates_three_documents_with_human_review(self):
        package = build_response_package(self.forecast)
        self.assertTrue(package.review_required)
        self.assertIn("악취 민원 확산 상황 브리핑", package.briefing)
        self.assertIn("악취 민원 현장점검 지시서", package.dispatch_order)
        self.assertIn("사후 결과보고서", package.followup_report_template)
        self.assertIn("AI 현장 대응 참고 가이드", package.response_guide)
        self.assertIn("담당자 검토 필요", package.response_guide)
        self.assertIn("원인 시설을 확정하지 않", package.briefing)
        self.assertIn("## 1. 민원 발생 현황", package.briefing)
        self.assertIn("## 2. 현장 확인 항목", package.dispatch_order)
        self.assertIn("## 5. AI 예측 결과와 실제 결과 비교", package.followup_report_template)

    def test_rejects_non_operational_grid(self):
        with self.assertRaises(ValueError):
            ForecastResult(
                event_id="X", event_time=datetime.now(), forecast_minutes=30, grid_size_m=500,
                areas=self.forecast.areas, model_metrics={}, generated_at=datetime.now(),
            )

    def test_completed_followup_compares_prediction_and_field_results(self):
        package = build_response_package(self.forecast)
        report = create_completed_followup(package, {
            "author": "담당자", "inspected_at": "2026-08-23T15:00",
            "dispatch_decided_at": "14:32", "departed_at": "14:35", "arrived_at": "14:48",
            "total_distance_km": "5.2", "actual_additional_area_count": "2", "checked_area": "G1",
            "field_findings": "현장 악취 확인", "followup_required": "필요", "notes": "추가 순찰 예정",
            "areas": [
                {"rank": 1, "additional_complaint": "발생", "odor_detected": "감지", "measurement": "5배", "action": "순찰"},
                {"rank": 2, "additional_complaint": "미발생", "odor_detected": "미감지", "measurement": "미검출", "action": "관찰"},
                {"rank": 3, "additional_complaint": "발생", "odor_detected": "미확인", "measurement": "", "action": "추가 확인"},
            ],
        })
        self.assertIn("Top 3 내 실제 추가 민원 권역 포함 여부: **포함**", report)
        self.assertIn("Top 3 내 포함 권역 수: 2", report)
        self.assertIn("현장 도착시각: 14:48", report)

    def test_template_mode_disables_llm_even_with_key(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "template", "GEMINI_API_KEY": "test-key"}, clear=False):
            self.assertEqual(provider_name(), "template")
            self.assertFalse(llm_configured())

    @patch("administrative_agent.llm._post_with_retry")
    def test_gemini_structured_documents(self, post):
        generated = {
            "briefing": "# Gemini 브리핑",
            "dispatch_order": "# Gemini 점검 지시서",
            "followup_report_template": "# Gemini 사후보고서",
            "response_guide": "# Gemini 대응 가이드",
        }
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "candidates": [{"content": {"parts": [{"text": json.dumps(generated, ensure_ascii=False)}]}}]
        }
        post.return_value = response
        fallback = build_response_package(self.forecast)
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key", "GEMINI_MODEL": "gemini-3.6-flash"}, clear=False):
            package = refine_with_gemini(self.forecast, fallback)
        self.assertEqual(package.briefing, "# Gemini 브리핑")
        self.assertEqual(package.response_guide, "# Gemini 대응 가이드")
        request = post.call_args.kwargs
        self.assertIn("gemini-3.6-flash:generateContent", post.call_args.args[0])
        self.assertEqual(request["json"]["generationConfig"]["responseMimeType"], "application/json")


if __name__ == "__main__":
    unittest.main()

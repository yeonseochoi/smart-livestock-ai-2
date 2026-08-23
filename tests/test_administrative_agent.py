from datetime import datetime
import json
import os
import unittest
from unittest.mock import Mock, patch

from administrative_agent.llm import llm_configured, provider_name, refine_with_gemini
from administrative_agent.models import ForecastResult, RiskArea
from administrative_agent.service import build_response_package


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
        self.assertIn("악취 민원 상황 브리핑", package.briefing)
        self.assertIn("현장 점검 지시서", package.dispatch_order)
        self.assertIn("사후 결과보고서", package.followup_report_template)
        self.assertIn("원인 시설을 확정하지 않", package.briefing)

    def test_rejects_non_operational_grid(self):
        with self.assertRaises(ValueError):
            ForecastResult(
                event_id="X", event_time=datetime.now(), forecast_minutes=30, grid_size_m=500,
                areas=self.forecast.areas, model_metrics={}, generated_at=datetime.now(),
            )

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
        request = post.call_args.kwargs
        self.assertIn("gemini-3.6-flash:generateContent", post.call_args.args[0])
        self.assertEqual(request["json"]["generationConfig"]["responseMimeType"], "application/json")


if __name__ == "__main__":
    unittest.main()

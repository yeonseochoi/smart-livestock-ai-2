from datetime import datetime, timezone
from dataclasses import replace
import json
import os
import unittest
from unittest.mock import Mock, patch

from administrative_agent.llm import llm_configured, provider_name, refine_with_gemini, _validated_package
from administrative_agent.models import FieldCandidate, ForecastResult, RiskArea
from administrative_agent.documents import field_candidate_body
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
        self.assertIn("## 3. 현장 확인 항목", package.dispatch_order)
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
        # 후보 섹션과 원래 권역 표가 사라진 응답은 채택하지 않는다.
        self.assertEqual(package, fallback)
        request = post.call_args.kwargs
        self.assertIn("gemini-3.6-flash:generateContent", post.call_args.args[0])
        self.assertEqual(request["json"]["generationConfig"]["responseMimeType"], "application/json")

    def test_field_candidates_and_rank_actions(self):
        candidate = FieldCandidate(1, "가 농장 | 나 농장", "시험 주소", 72.3,
                                   {"풍향 일치": 35, "거리": 12, "다중 측정 일치": 15, "과거 반복": 10.3},
                                   "현장 확인 필요", True, tier="교차 확인", support_count=3, support_total=4,
                                   complaint_km=1.8, travel_km=7.4, visit_order=2)
        forecast = replace(self.forecast, field_candidates=(candidate,),
                           generated_at=datetime(2026, 9, 28, 18, 0, tzinfo=timezone.utc))
        package = build_response_package(forecast)
        for document in (package.briefing, package.dispatch_order):
            self.assertIn("가 농장 / 나 농장", document)
            self.assertIn("시험 주소", document)
            self.assertIn("방문 전 주소 확인", document)
            self.assertIn("최종 판단은 농가 경계에서 채취한 시료로 합니다", document)
            self.assertIn("교차 확인 (민원 위치 4곳 중 3곳이 풍상으로 가리킴)", document)
            self.assertIn("민원 근접 12.0/25", document)
            self.assertNotIn("방문 동선 제안", document)
        order = package.dispatch_order
        self.assertLess(order.index("## 2. 발생원으로 예측되는 농가 후보"), order.index("## 3. 현장 확인 항목"))
        self.assertLess(order.index("## 1. 점검 대상"), order.index("## 2. 발생원으로 예측되는 농가 후보"))
        self.assertIn("## 5. 발생원으로 예측되는 농가 후보", package.briefing)
        self.assertIn("2026.09.29. 03:00", package.dispatch_order)
        for line in package.dispatch_order.splitlines():
            if line.startswith(("|2순위|", "|3순위|")):
                self.assertNotIn("가장 먼저", line)
                self.assertIn("1순위 확인 후 순차 확인", line)
        self.assertNotIn("주요 우선확인 지역", package.briefing)

    def test_empty_candidates_and_llm_safety(self):
        fallback = build_response_package(self.forecast)
        self.assertIn("조건에 맞는 방문 후보 없음", fallback.briefing)
        safe = fallback.to_dict()["documents"]
        self.assertEqual(_validated_package(safe, self.forecast, fallback), fallback)
        revised = {**safe, "response_guide": safe["response_guide"] + "\n현장 상황을 기록하세요."}
        self.assertNotEqual(_validated_package(revised, self.forecast, fallback), fallback)
        for forbidden in ("발생원 확률 78%", "원인 농가 Top 5", "이 농가가 악취를 발생시켰다", "후보 안에 실제 발생원이 반드시 있다"):
            with self.subTest(forbidden=forbidden):
                bad = {**safe, "response_guide": forbidden}
                self.assertEqual(_validated_package(bad, self.forecast, fallback), fallback)
        missing = {**safe, "briefing": safe["briefing"].replace(field_candidate_body(self.forecast), "")}
        self.assertEqual(_validated_package(missing, self.forecast, fallback), fallback)


if __name__ == "__main__":
    unittest.main()

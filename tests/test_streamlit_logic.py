"""UI 실행 없이 사건 연결·문서 입력의 경계 조건을 검사한다."""
import ast
from datetime import datetime
import logging
from pathlib import Path
import unittest
from unittest.mock import Mock

import pandas as pd

from administrative_agent.models import FieldCandidate, ForecastResult, RiskArea
from administrative_agent.service import build_response_package
from administrative_agent.time_utils import now_kst
from generate_agent_documents import _relative_scores
from odor_service.region_prediction.data import assign_grid
from administrative_agent.llm import _validated_package


def app_functions(**overrides):
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    names = {"field_context_for_event", "unified_event_view", "current_forecast", "generate_documents"}
    tree = ast.Module(body=[n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    namespace = dict(datetime=datetime, logging=logging, pd=pd, FieldCandidate=FieldCandidate,
                     ForecastResult=ForecastResult, RiskArea=RiskArea, now_kst=now_kst,
                     _relative_scores=_relative_scores, assign_grid=assign_grid,
                     build_response_package=build_response_package, MAX_FIELD_CANDIDATES=3, **overrides)
    exec(compile(tree, "streamlit_app.py", "exec"), namespace)
    return namespace


class StreamlitLogicTest(unittest.TestCase):
    def warning_forecast(self, field):
        event = {"id": "EV2-WARNING", "hour": "2025-07-01 23:09", "reports": [],
                 "broad": [{"center": [35.9, 127], "score": s} for s in (.9, .8, .7)],
                 "weather": {"windSpeed": 0.3}}
        card = {"rank": 1, "name": "시험 농가", "score": 70, "address": "시험 주소"}
        context = {"event": {"field": {"cards": [card], **field}}}
        return app_functions()["current_forecast"](event, context)

    def test_weak_wind_and_unstable_candidates_are_explained_in_both_documents(self):
        warnings = ["관측 풍향을 신뢰하기 어려워 현장 풍향 확인이 우선",
                    "풍향이 ±20° 달라지면 후보 절반 이상이 바뀜"]
        forecast = self.warning_forecast({"confidence": "매우 낮음", "stability": .4,
                                         "stability_label": "낮음", "notes": warnings})
        package = build_response_package(forecast)
        for document in (package.briefing, package.dispatch_order):
            self.assertIn("풍향 신뢰도: 매우 낮음", document)
            self.assertIn("풍향 ±20° 변화 시 후보 유지율: 40%", document)
            for warning in warnings:
                self.assertIn(warning, document)
                self.assertLess(document.index(warning), document.index("### 1순위"))

    def test_llm_cannot_remove_or_soften_candidate_warnings(self):
        warning = "관측 풍향을 신뢰하기 어려워 현장 풍향 확인이 우선"
        forecast = self.warning_forecast({"confidence": "매우 낮음", "stability": .4,
                                         "stability_label": "낮음", "notes": [warning]})
        fallback = build_response_package(forecast)
        for key in ("briefing", "dispatch_order"):
            for replacement in ("", "추천 순위를 신뢰하고 이동"):
                with self.subTest(document=key, replacement=replacement):
                    documents = fallback.to_dict()["documents"]
                    documents[key] = documents[key].replace(warning, replacement)
                    # 수정되지 않은 응답은 별도 객체로 채택되므로 fallback identity로 거절 여부를 검사한다.
                    self.assertIs(_validated_package(documents, forecast, fallback), fallback)

    def test_missing_confidence_is_not_reported_as_high_or_zero_stability(self):
        forecast = self.warning_forecast({})
        package = build_response_package(forecast)
        for document in (package.briefing, package.dispatch_order):
            self.assertIn("풍향 신뢰도: 미제공", document)
            self.assertNotIn("후보 유지율: 0%", document)
            self.assertNotIn("매우 낮음", document)

    def test_stable_candidates_do_not_receive_weak_wind_warnings(self):
        forecast = self.warning_forecast({"confidence": "보통", "stability": 1.0,
                                         "stability_label": "높음", "notes": []})
        package = build_response_package(forecast)
        for document in (package.briefing, package.dispatch_order):
            self.assertIn("후보 유지율: 100%", document)
            self.assertNotIn("매우 낮음", document)
            self.assertNotIn("후보 절반 이상이 바뀜", document)
        accepted = _validated_package(package.to_dict()["documents"], forecast, package)
        self.assertIsNot(accepted, package)
        self.assertEqual(accepted, package)

    def test_id_mismatch_never_matches_nearby_time(self):
        context = {"event": {"event_id": "EV2-1"}, "event_time": datetime(2025, 1, 1)}
        funcs = app_functions(load_field_replays=Mock(return_value=[context]), replay_revision=lambda: 1)
        self.assertIsNone(funcs["field_context_for_event"]({"id": "EV2-2", "hour": "2025-01-01T00:00"}))
        self.assertEqual(funcs["field_context_for_event"]({"id": "EV2-1"})["event"], context["event"])

    def test_initial_grid_count_is_not_nearest_top3_and_exception_is_private(self):
        # 두 점은 1km 격자 경계를 넘는다. Top3 모두 같은 곳이어도 초기 격자는 둘이다.
        reports = [[35.9, 127, 2], [35.92, 127, 4]]
        event = {"id": "EV2-T", "hour": "2025-01-01 00:00", "reports": reports, "initialCount": 2,
                 "broad": [{"center": [36, 127], "score": s} for s in (.99, .96, .93)], "weather": {}}
        funcs = app_functions(llm_configured=lambda: True, provider_name=lambda: "test",
                              refine_with_llm=Mock(side_effect=RuntimeError("<script>private</script>")))
        forecast = funcs["current_forecast"](event)
        self.assertEqual(forecast.initial_grid_count, 2)
        self.assertEqual([a.relative_risk for a in forecast.areas], [100, 80, 60])
        self.assertEqual(forecast.initial_intensity_average, 3)
        with self.assertLogs(level="ERROR") as log:
            _, _, warning = funcs["generate_documents"](event)
        self.assertEqual(warning, "LLM 호출 실패로 기본 양식을 사용했습니다")
        self.assertNotIn("private", warning)
        self.assertIn("private", "\n".join(log.output))

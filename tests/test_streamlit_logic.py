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


def app_functions(**overrides):
    source = (Path(__file__).resolve().parents[1] / "streamlit_app.py").read_text(encoding="utf-8")
    names = {"field_context_for_event", "unified_event_view", "current_forecast", "generate_documents"}
    tree = ast.Module(body=[n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[])
    namespace = dict(datetime=datetime, logging=logging, pd=pd, FieldCandidate=FieldCandidate,
                     ForecastResult=ForecastResult, RiskArea=RiskArea, now_kst=now_kst,
                     _relative_scores=_relative_scores, assign_grid=assign_grid,
                     build_response_package=build_response_package, **overrides)
    exec(compile(tree, "streamlit_app.py", "exec"), namespace)
    return namespace


class StreamlitLogicTest(unittest.TestCase):
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
        self.assertEqual(warning, "LLM 호출 실패로 안전 템플릿을 사용했습니다")
        self.assertNotIn("private", warning)
        self.assertIn("private", "\n".join(log.output))

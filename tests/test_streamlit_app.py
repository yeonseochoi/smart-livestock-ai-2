from __future__ import annotations

from pathlib import Path
from datetime import datetime, timedelta
import json
import tomllib
import unittest
import re
import tempfile
import os
from unittest.mock import patch

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTest(unittest.TestCase):
    def setUp(self):
        # 로컬 API 키가 있더라도 회귀 검증은 외부 호출 없이 수행한다.
        provider = patch.dict(os.environ, {"LLM_PROVIDER": "template"})
        provider.start()
        self.addCleanup(provider.stop)

    def test_missing_weather_and_empty_candidates(self):
        source_path = next(path for path in (ROOT / "outputs/odor_service/replay").glob("*.json") if path.name != "index.json")
        replay = json.loads(source_path.read_text(encoding="utf-8"))
        replay["events"] = replay["events"][:1]
        replay["events"][0]["field"]["cards"] = []
        replay["events"][0]["weather"] = {"humidity": None, "rainfall": None}
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "fixture.json").write_text(json.dumps(replay, ensure_ascii=False), encoding="utf-8")
            source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")
            source = source.replace("ROOT = Path(__file__).resolve().parent", f"ROOT = Path({str(ROOT)!r})")
            source = source.replace('REPLAY_DIR = ROOT / "outputs" / "odor_service" / "replay"', f"REPLAY_DIR = Path({folder!r})")
            app = AppTest.from_string(source, default_timeout=20).run()
            self.assertFalse(app.exception)
            weather_html = next(item.value for item in app.markdown if item.value.startswith('<div class="weather-cards">'))
            self.assertNotIn("상대습도", weather_html)
            self.assertNotIn("강수", weather_html)
            next(b for b in app.button if b.label == "대응 문서 생성").click().run()
            self.assertFalse(app.exception)
            package = app.session_state["documents"]
            self.assertIn("조건에 맞는 방문 후보 없음", package.dispatch_order)
            self.assertIn("상대습도: 미제공", package.briefing)
            app.session_state["document_warning"] = '<script>alert("test")</script>'
            app.run()
            notices = [item.value for item in app.markdown if 'class="notice"' in item.value]
            self.assertTrue(any("&lt;script&gt;" in value for value in notices))
            self.assertTrue(all("<script>" not in value for value in notices))

    def test_deployment_uses_polling_reload(self) -> None:
        config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
        self.assertEqual(config["server"]["fileWatcherType"], "poll")
        self.assertTrue(config["server"]["runOnSave"])

    def app(self) -> AppTest:
        app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=20)
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_initial_demo_screen_and_event_navigation(self) -> None:
        app = self.app()
        self.assertTrue(any("과거 Event 재현 모드" in item.value for item in app.markdown))
        self.assertGreater(int(app.metric[0].value), 0)
        self.assertEqual(app.metric[1].value, "3")
        self.assertEqual(len(app.metric), 2)
        self.assertTrue(any(toggle.label == "검증용 실제 이후 신고 표시" for toggle in app.toggle))
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("padding-top:4.6rem", rendered)
        self.assertIn("💨", rendered)
        self.assertTrue(any("추천 대기 장소" in item.value for item in app.caption))
        self.assertIn("🌧️", rendered)
        self.assertIn("민원 집중 감지", rendered)
        self.assertIn("먼저 확인할 농가", rendered)
        self.assertIn("경계 시료 채취", rendered)
        self.assertIn("민원 확산 예측에서 현장 확인까지", rendered)
        self.assertIn("과거 반복", rendered)
        self.assertIn("왜 1순위인가요?", rendered)
        self.assertNotIn("현장에서는", rendered)
        self.assertIn("nanum-square-neo.css", rendered)
        self.assertIn("NanumSquareNeoVariable", rendered)
        self.assertNotIn("position:sticky", rendered)
        self.assertIn(".st-key-map_panel,.st-key-agent_panel{position:static;height:auto;overflow:visible", rendered)
        self.assertEqual(app.session_state["document_schema_version"], 4)
        priority_cards = [item.value for item in app.markdown if 'class="priority' in item.value]
        self.assertEqual(len(priority_cards), 3)
        self.assertTrue(all("%" not in item for item in priority_cards))
        self.assertTrue(all("가장 먼저" not in item for item in priority_cards[1:]))
        self.assertNotIn("T", app.selectbox[0].options[-1].split(" · ")[0])

        actual_toggle = next(toggle for toggle in app.toggle if toggle.label == "검증용 실제 이후 신고 표시")
        actual_toggle.set_value(True).run()
        self.assertTrue(actual_toggle.value)
        self.assertEqual(app.metric[2].label, "Top 3 중 실제 이후 신고 권역")
        self.assertRegex(app.metric[2].value, r"^[0-3]/3$")
        previous = next(button for button in app.button if button.key == "prev")
        previous.click().run()
        self.assertFalse(app.exception)
        self.assertGreater(int(app.metric[0].value), 0)
        self.assertFalse(next(toggle for toggle in app.toggle if toggle.key == "show_actual").value)
        self.assertEqual(len(app.metric), 2)

    def test_safe_template_document_generation(self) -> None:
        app = self.app()
        generate = next(button for button in app.button if button.label == "대응 문서 생성")
        generate.click().run(timeout=20)
        self.assertFalse(app.exception)
        package = app.session_state["documents"]
        self.assertEqual(package.event_id, app.session_state["document_event"])
        self.assertTrue(package.event_id.startswith("EV2-"))
        sources = [event for path in (ROOT / "outputs/odor_service/replay").glob("*.json")
                   if path.name != "index.json"
                   for event in json.loads(path.read_text(encoding="utf-8")).get("events", [])]
        source = next(event for event in sources if event["event_id"] == package.event_id)
        self.assertEqual(package.forecast.event_time, datetime.fromisoformat(source["event_hour"]))
        self.assertEqual([(area.center_latitude, area.center_longitude) for area in package.forecast.areas],
                         [(grid["lat"], grid["lon"]) for grid in source["top3"]])
        self.assertEqual(package.forecast.weather["windSpeed"], source["direction_overlap"]["wind_speed"])
        intensities = [r["intensity"] for r in source["first30_complaints"] if r.get("intensity") is not None]
        self.assertEqual(package.forecast.initial_intensity_average, round(sum(intensities) / len(intensities), 1))
        self.assertEqual(package.forecast.initial_intensity_maximum, max(intensities))
        self.assertGreater(package.forecast.initial_grid_count, 0)
        self.assertEqual(package.forecast.weather["humidity"], source["weather"]["humidity"])
        self.assertEqual(package.forecast.weather["rainfall"], source["weather"]["rainfall"])
        candidates = package.forecast.field_candidates
        expected_names = [c.get("merged_names") or c["name"] for c in source["field"]["cards"]]
        self.assertEqual([c.display_name for c in candidates], expected_names)
        for document in (package.briefing, package.dispatch_order):
            names = re.findall(r"^### \d+순위 · (.+)$", document, flags=re.M)
            self.assertEqual(names, [" ".join(name.replace("|", " / ").split()) for name in expected_names])
        self.assertEqual(str(package.forecast.generated_at.tzinfo), "Asia/Seoul")
        self.assertEqual(package.forecast.model_metrics, {})
        boundary = package.forecast.event_time
        self.assertIn(f'{boundary:%H:%M}~{boundary + timedelta(minutes=30):%H:%M}', package.briefing)
        tabs = [tab.label for tab in app.tabs]
        self.assertEqual(tabs, ["① 상황 브리핑", "② 점검 지시서", "③ 사후 결과 입력", "④ AI 대응 가이드"])
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("악취 민원 확산 상황 브리핑", rendered)
        self.assertIn("악취 민원 현장점검 지시서", rendered)
        self.assertIn("참고 기상정보", rendered)
        self.assertIn("AI 예측 결과와 실제 결과 비교", rendered)
        self.assertIn("AI 현장 대응 참고 가이드", rendered)
        inputs = [item.label for item in app.text_input]
        self.assertIn("작성자", inputs)
        self.assertIn("현장 도착시각", inputs)
        self.assertIn("실제 우선 점검 권역", inputs)
        self.assertTrue(any(button.label == "입력값으로 결과보고서 완성" for button in app.button))


if __name__ == "__main__":
    unittest.main()

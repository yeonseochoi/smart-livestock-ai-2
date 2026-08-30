from __future__ import annotations

from pathlib import Path
import tomllib
import unittest

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTest(unittest.TestCase):
    def test_deployment_disables_hot_reload_module_eviction(self) -> None:
        config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
        self.assertEqual(config["server"]["fileWatcherType"], "none")
        self.assertFalse(config["server"]["runOnSave"])

    def app(self) -> AppTest:
        app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=20)
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_initial_demo_screen_and_event_navigation(self) -> None:
        app = self.app()
        self.assertTrue(any("과거 Event 재현 모드" in item.value for item in app.markdown))
        self.assertEqual([metric.value for metric in app.metric[:3]], ["8", "3", "5"])
        self.assertTrue(any(toggle.label == "검증용 실제 이후 신고 표시" for toggle in app.toggle))
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("padding-top:4.6rem", rendered)
        self.assertIn("💨", rendered)
        self.assertIn("0.6 m/s", rendered)
        self.assertIn("🌧️", rendered)

        previous = next(button for button in app.button if button.key == "prev")
        previous.click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "11")

    def test_safe_template_document_generation(self) -> None:
        app = self.app()
        generate = next(button for button in app.button if button.label == "대응 문서 생성")
        generate.click().run(timeout=20)
        self.assertFalse(app.exception)
        tabs = [tab.label for tab in app.tabs]
        self.assertEqual(tabs, ["상황 브리핑", "현장점검 지시서", "사후 결과보고서", "현장 결과 입력"])
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("악취 민원 확산 상황 브리핑", rendered)
        self.assertIn("악취 민원 현장점검 지시서", rendered)
        self.assertIn("참고 기상정보", rendered)
        self.assertIn("AI 예측 결과와 실제 결과 비교", rendered)


if __name__ == "__main__":
    unittest.main()

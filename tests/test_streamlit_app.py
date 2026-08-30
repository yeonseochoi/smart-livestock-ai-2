from __future__ import annotations

from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]


class StreamlitAppTest(unittest.TestCase):
    def app(self) -> AppTest:
        app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=20)
        app.run()
        self.assertFalse(app.exception)
        return app

    def test_initial_demo_screen_and_event_navigation(self) -> None:
        app = self.app()
        self.assertTrue(any("과거 Event 재현 모드" in item.value for item in app.markdown))
        self.assertEqual([metric.value for metric in app.metric[:3]], ["10", "3", "0"])
        self.assertTrue(any(toggle.label == "검증용 실제 이후 신고 표시" for toggle in app.toggle))

        previous = next(button for button in app.button if button.key == "prev")
        previous.click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.metric[0].value, "27")

    def test_safe_template_document_generation(self) -> None:
        app = self.app()
        generate = next(button for button in app.button if button.label == "대응 문서 생성")
        generate.click().run(timeout=20)
        self.assertFalse(app.exception)
        tabs = [tab.label for tab in app.tabs]
        self.assertEqual(tabs, ["상황 브리핑", "점검 지시서", "사후보고서 예시", "현장 결과 입력"])
        rendered = "\n".join(item.value for item in app.markdown)
        self.assertIn("악취 민원 상황 브리핑", rendered)
        self.assertIn("현장 점검 지시서", rendered)


if __name__ == "__main__":
    unittest.main()

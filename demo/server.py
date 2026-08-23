from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from administrative_agent.config import load_env

load_env(ROOT / ".env")

from administrative_agent.llm import llm_configured, provider_name, refine_with_llm
from administrative_agent.service import build_response_package, create_completed_followup, write_response_package
from generate_agent_documents import DEFAULT_METRICS, DEFAULT_PREDICTIONS, forecast_from_csv

OUTPUT = ROOT / "outputs" / "administrative_agent"


class DemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT / "demo"), **kwargs)

    def _json(self, status: int, body: dict) -> None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        return json.loads(self.rfile.read(length) or b"{}")

    def do_GET(self) -> None:
        if self.path == "/api/status":
            self._json(200, {"ok": True, "llmConfigured": llm_configured(), "provider": provider_name(), "mode": "llm" if llm_configured() else "template"})
            return
        super().do_GET()

    def do_POST(self) -> None:
        try:
            body = self._body()
            if self.path == "/api/agent/generate":
                forecast = forecast_from_csv(
                    ROOT / DEFAULT_PREDICTIONS, ROOT / DEFAULT_METRICS, body.get("event_id"), body.get("event_time")
                )
                package, mode, warning = build_response_package(forecast), "template", None
                if body.get("use_llm", True) and llm_configured():
                    try:
                        package, mode = refine_with_llm(forecast, package), provider_name()
                    except Exception as exc:
                        warning = f"LLM 호출 실패로 안전 템플릿을 사용했습니다: {exc}"
                write_response_package(package, OUTPUT)
                self._json(200, {"ok": True, "mode": mode, "warning": warning, **package.to_dict()})
                return
            if self.path == "/api/agent/followup":
                forecast = forecast_from_csv(
                    ROOT / DEFAULT_PREDICTIONS, ROOT / DEFAULT_METRICS, body.get("event_id"), body.get("event_time")
                )
                package = build_response_package(forecast)
                report = create_completed_followup(package, body)
                OUTPUT.mkdir(parents=True, exist_ok=True)
                filename = f"followup_{forecast.event_id}_{datetime.now():%Y%m%d_%H%M%S}.md"
                (OUTPUT / filename).write_text(report, encoding="utf-8")
                self._json(200, {"ok": True, "filename": filename, "report": report})
                return
            self._json(404, {"ok": False, "error": "존재하지 않는 API입니다."})
        except Exception as exc:
            self._json(400, {"ok": False, "error": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser(description="지도 데모와 행정 대응 Agent API 실행")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DemoHandler)
    print(f"데모 서버: http://127.0.0.1:{args.port}")
    print(f"문서 생성 모드: {provider_name() if llm_configured() else '안전 템플릿'}")
    server.serve_forever()


if __name__ == "__main__":
    main()

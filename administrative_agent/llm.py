from __future__ import annotations

import json
import os
import time

import requests

from .models import ForecastResult, ResponsePackage


SCHEMA = {
    "type": "object",
    "properties": {
        "briefing": {"type": "string"},
        "dispatch_order": {"type": "string"},
        "followup_report_template": {"type": "string"},
    },
    "required": ["briefing", "dispatch_order", "followup_report_template"],
    "additionalProperties": False,
}


def _post_with_retry(url: str, **kwargs) -> requests.Response:
    """느린 생성 응답을 기다리고 일시적인 서버 오류만 한 번 재시도한다."""
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            response = requests.post(url, timeout=90, **kwargs)
            if response.status_code not in {429, 500, 502, 503, 504} or attempt == 1:
                response.raise_for_status()
                return response
            last_error = requests.HTTPError(f"Gemini temporary error: HTTP {response.status_code}")
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_error = exc
            if attempt == 1:
                raise
        time.sleep(1.5)
    raise RuntimeError(str(last_error))


def _valid(value: str | None) -> bool:
    return bool(value and not value.startswith("your_") and value != "your_supported_model_id")


def provider_name() -> str:
    explicit = os.getenv("LLM_PROVIDER", "").lower()
    if explicit in {"gemini", "openai", "template"}:
        return explicit
    if _valid(os.getenv("GEMINI_API_KEY")):
        return "gemini"
    if _valid(os.getenv("OPENAI_API_KEY")) and not _valid(os.getenv("OPENAI_MODEL")):
        return "gemini"  # 초기 예시 파일의 키 이름을 사용한 경우의 호환 처리
    return "openai"


def llm_configured() -> bool:
    if provider_name() == "template":
        return False
    if provider_name() == "gemini":
        return _valid(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY"))
    return _valid(os.getenv("OPENAI_API_KEY")) and _valid(os.getenv("OPENAI_MODEL"))


def refine_with_openai(forecast: ForecastResult, fallback: ResponsePackage) -> ResponsePackage:
    """Responses API로 문안을 다듬는다. 판단값과 순위는 변경하지 못하게 제한한다."""
    api_key, model = os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MODEL")
    if not api_key or not model:
        return fallback
    payload = {
        "model": model,
        "store": False,
        "instructions": (
            "당신은 지자체 악취 민원 대응 문서 작성 보조자다. 입력의 수치, 권역, 순위를 절대 변경하거나 "
            "새 사실을 만들지 않는다. 악취 발생원·원인 시설·실제 악취 발생을 단정하지 않는다. "
            "세 문서를 간결한 한국어 Markdown으로 작성하고, 담당자 검토 필요와 상대 우선순위라는 한계를 포함한다. "
            "각 문서는 # 제목, ## 섹션, 핵심 정보 표, 짧은 목록을 사용하고 HTML은 출력하지 않는다. "
            "사후 결과보고서는 미입력 필드가 있는 확장용 양식으로만 작성한다."
        ),
        "input": json.dumps({"forecast": forecast.to_dict(), "safe_template": fallback.to_dict()["documents"]}, ensure_ascii=False),
        "text": {"format": {"type": "json_schema", "name": "administrative_documents", "strict": True, "schema": SCHEMA}},
    }
    response = _post_with_retry(
        "https://api.openai.com/v1/responses",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
    )
    body = response.json()
    output_text = body.get("output_text")
    if not output_text:
        output_text = next(
            content["text"]
            for item in body.get("output", []) if item.get("type") == "message"
            for content in item.get("content", []) if content.get("type") == "output_text"
        )
    documents = json.loads(output_text)
    return ResponsePackage(
        event_id=forecast.event_id, review_required=True, forecast=forecast,
        briefing=documents["briefing"], dispatch_order=documents["dispatch_order"],
        followup_report_template=documents["followup_report_template"],
    )


def refine_with_gemini(forecast: ForecastResult, fallback: ResponsePackage) -> ResponsePackage:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    if not _valid(api_key):
        return fallback
    instructions = (
        "당신은 지자체 악취 민원 대응 문서 작성 보조자다. 입력의 수치, 권역, 순위를 절대 변경하거나 "
        "새 사실을 만들지 않는다. 악취 발생원·원인 시설·실제 악취 발생을 단정하지 않는다. "
        "세 문서를 간결한 한국어 Markdown으로 작성하고 담당자 검토 필요와 상대 우선순위라는 한계를 포함한다. "
        "각 문서는 # 제목, ## 섹션, 핵심 정보 표, 짧은 목록을 사용하고 HTML은 출력하지 않는다. "
        "사후 결과보고서는 미입력 필드가 있는 확장용 양식으로만 작성한다."
    )
    prompt = instructions + "\n\n입력:\n" + json.dumps(
        {"forecast": forecast.to_dict(), "safe_template": fallback.to_dict()["documents"]}, ensure_ascii=False
    )
    response = _post_with_retry(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
        json={
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": SCHEMA},
        },
    )
    body = response.json()
    output_text = body["candidates"][0]["content"]["parts"][0]["text"]
    documents = json.loads(output_text)
    return ResponsePackage(
        event_id=forecast.event_id, review_required=True, forecast=forecast,
        briefing=documents["briefing"], dispatch_order=documents["dispatch_order"],
        followup_report_template=documents["followup_report_template"],
    )


def refine_with_llm(forecast: ForecastResult, fallback: ResponsePackage) -> ResponsePackage:
    if provider_name() == "gemini":
        return refine_with_gemini(forecast, fallback)
    return refine_with_openai(forecast, fallback)

"""운영 화면과 문서는 한국 표준시를 사용한다."""
from datetime import datetime
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")


def now_kst() -> datetime:
    return datetime.now(KST)


def as_kst(value: datetime) -> datetime:
    # 원본 민원 시각은 timezone 없는 한국 현지시각이다.
    return value.replace(tzinfo=KST) if value.tzinfo is None else value.astimezone(KST)

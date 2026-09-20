"""지방행정인허가 대기배출시설에서 악취 관련 가능 업종만 선별한다."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "public_source_cache"
INPUTS = {
    "익산시": CACHE_DIR / "air_emission_iksan.csv",
    "김제시": CACHE_DIR / "air_emission_gimje.csv",
}
OUTPUT = CACHE_DIR / "odor_relevant_factories.csv"
SOURCE_URL = "https://file.localdata.go.kr/file/air_pollution_facility_installation/info"

# 사용자 지정 유지 업종을 실제 공개 열(업태·업종·주생산품·사업장명)에 적용함.
INCLUDE_PATTERNS = [
    (
        "폐기물 처리·재활용",
        re.compile(r"폐기물.*(?:처리|재활용)|지정\s*외\s*폐기물|재생용 비금속가공원료|비금속원료 재생"),
    ),
    ("사료·비료 제조", re.compile(r"사료|비료|질소화합물|상토")),
    (
        "도축·육가공",
        re.compile(r"도축|육류.*(?:가공|저장)|육지동물고기.*(?:가공|저장)|가금류.*(?:가공|저장)|육가공"),
    ),
    (
        "식품 제조",
        re.compile(
            r"식품|식료품|음·식료품|도정|제분|전분|당류|조미료|동물성 유지|식물성 유지|발효주|주정|"
            r"채소.*(?:가공|저장)|과실.*(?:가공|저장)|수산.*(?:가공|저장)"
        ),
    ),
    (
        "화학·합성수지·플라스틱 재생",
        re.compile(r"화학|합성수지|재생 플라스틱|플라스틱.*재생|플라스틱원료|타이어 재생"),
    ),
    ("레미콘·아스콘", re.compile(r"레미콘|아스콘|아스팔트 콘크리트")),
    ("피혁·염색", re.compile(r"피혁|염색")),
]

EXCLUDE_PATTERN = re.compile(
    r"자동차.*(?:수리|세차)|목욕|욕탕|세탁|병원|의료업|인쇄|일반 공공 행정|식품위생용 종이제품"
)


def clean(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def match_reason(row: pd.Series) -> str | None:
    name = clean(row.get("사업장명"))
    classified_text = " ".join(
        clean(row.get(column)) for column in ("업태구분명", "업종구분명", "주생산품명")
    ).strip()
    full_text = f"{name} {classified_text}"
    if EXCLUDE_PATTERN.search(full_text):
        return None
    # 업태·업종·주생산품이 모두 비어 있을 때만 사업장명을 보조 근거로 사용함.
    text = classified_text or name
    for reason, pattern in INCLUDE_PATTERNS:
        if pattern.search(text):
            return reason
    return None


def load_city(city: str, path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    frame = pd.read_csv(path, dtype=str, low_memory=False, encoding="cp949")
    active = frame[frame["영업상태명"].eq("영업/정상")].copy()
    active["address"] = active["도로명주소"].map(clean)
    missing_road = active["address"].eq("")
    active.loc[missing_road, "address"] = active.loc[missing_road, "지번주소"].map(clean)
    addressable = active[active["address"].ne("")].copy()
    addressable["filter_reason"] = addressable.apply(match_reason, axis=1)
    selected = addressable[addressable["filter_reason"].notna()].copy()
    selected["city"] = city
    selected["odor_relevant"] = True
    selected["source_url"] = SOURCE_URL
    selected = selected.rename(
        columns={
            "관리번호": "management_id",
            "사업장명": "name",
            "업태구분명": "business_type",
            "업종구분명": "industry_name",
            "종별명": "facility_class",
            "주생산품명": "main_product",
            "영업상태명": "status",
            "인허가일자": "permit_date",
            "최종수정시점": "last_modified",
        }
    )
    selected = selected.drop_duplicates(subset=["city", "name", "address"], keep="last")
    columns = [
        "city",
        "management_id",
        "name",
        "business_type",
        "industry_name",
        "facility_class",
        "main_product",
        "status",
        "address",
        "permit_date",
        "last_modified",
        "filter_reason",
        "odor_relevant",
        "source_url",
    ]
    stats = {
        "raw": len(frame),
        "active": len(active),
        "addressable": len(addressable),
        "selected": len(selected),
    }
    return selected[columns], stats


def main() -> None:
    parts: list[pd.DataFrame] = []
    city_stats: dict[str, dict[str, int]] = {}
    for city, path in INPUTS.items():
        selected, stats = load_city(city, path)
        parts.append(selected)
        city_stats[city] = stats

    output = pd.concat(parts, ignore_index=True)
    output = output.sort_values(["city", "filter_reason", "name"], kind="stable")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT, index=False, encoding="utf-8-sig")

    print(f"saved={OUTPUT}")
    print(f"rows={len(output)}")
    print(f"city_stats={city_stats}")
    print("by_city=" + str(output.groupby("city").size().to_dict()))
    print("by_reason=" + str(output.groupby("filter_reason").size().to_dict()))
    print("facility_class=" + str(output["facility_class"].fillna("미상").value_counts().to_dict()))


if __name__ == "__main__":
    main()

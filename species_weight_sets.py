"""축종 배출 가중치 계수 세트 3벌 + 시설 수만(대조).

계수는 결과의 존재가 아니라 정밀도에 영향을 준다는 것을 확인했으므로(test_wind_source_association.py),
하나를 고르지 않고 세 벌을 나란히 돌려 결론이 바뀌는지 표로 남긴다.

1. current      : Track A 1·2판이 쓴 초기 가정(돼지 1.0 기준). 사육두수 × 계수.
2. eea_nh3      : EMEP/EEA 대기오염물질 배출 인벤토리 가이드북 2023, 3.B 분뇨관리 표 3-2,
                  축사·저장 단계 NH3 (kg/두/년). 사육두수 × 계수. 확인일 2026-09-20.
3. jang_odor    : 장영기·정봉진 외(2010) 환경영향평가 19(1), 축사 내부 종합 악취강도, 젖소=1 상대배수.
                  시설 한 곳의 실내 농도라 두수와 곱하지 않고 **시설 단위** 가중치로 쓴다.
4. count        : 시설 1곳 = 1. 계수 없이 위치만.

문헌에 없는 축종은 ASSUMED 표시. 이 값들은 가정이며 보고서에 그대로 밝힌다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CURRENT = {
    "돼지": 1.0, "한우": 0.3, "육우": 0.3, "젖소": 0.4, "염소": 0.1, "산양": 0.1, "사슴": 0.1,
    "육계": 0.01, "종계/산란계": 0.01, "오리": 0.01, "메추리": 0.01, "부화용알생산": 0.01,
}

# EMEP/EEA 2023 표 3-2 'housing, storage and yards' 열. 한국 관행: 한우 깔짚(solid), 젖소 깔짚(solid),
# 돼지 슬러리(비육돈 90% 3.7 + 모돈 10% 12.5 ≈ 4.6), 산란계 건식(solid 0.16), 육계 깔짚(litter 0.13), 오리 0.45,
# 염소 0.4. 사슴·메추리는 표에 없음(ASSUMED: 사슴=염소, 메추리=육계×0.4).
EEA_NH3 = {
    "돼지": 4.6, "한우": 5.7, "육우": 5.7, "젖소": 16.1, "염소": 0.4, "산양": 0.4, "사슴": 0.4,
    "육계": 0.13, "종계/산란계": 0.16, "오리": 0.45, "메추리": 0.05, "부화용알생산": 0.16,
}
EEA_ASSUMED = {"사슴", "메추리"}

# 장영기 외 2010: 모돈사 35, 비육돈사 20, 분만돈사 12, 산란계사 7, 자돈사 6, 육계사 4, 젖소사 1. 한육우 미명시.
# 돼지 농가는 돈군 혼합이라 비육돈사 20을 대표값으로. ASSUMED: 한우·육우 2(젖소 1~육계 4 사이), 오리·메추리 4(가금 하한),
# 염소·산양·사슴 1, 부화용알생산 7(산란계 계열).
JANG_ODOR = {
    "돼지": 20.0, "한우": 2.0, "육우": 2.0, "젖소": 1.0, "염소": 1.0, "산양": 1.0, "사슴": 1.0,
    "육계": 4.0, "종계/산란계": 7.0, "오리": 4.0, "메추리": 4.0, "부화용알생산": 7.0,
}
JANG_ASSUMED = {"한우", "육우", "오리", "메추리", "염소", "산양", "사슴", "부화용알생산"}

SETS = {
    "current": {"factor": CURRENT, "per": "head", "assumed": set(),
                "source": "Track A 초기 가정 (validation.json constants)"},
    "eea_nh3": {"factor": EEA_NH3, "per": "head", "assumed": EEA_ASSUMED,
                "source": "EMEP/EEA Guidebook 2023 3.B Table 3-2 housing+storage NH3 kg/AAP/yr"},
    "jang_odor": {"factor": JANG_ODOR, "per": "facility", "assumed": JANG_ASSUMED,
                  "source": "장영기 외 2010 환경영향평가 19(1) 축종별 종합 악취강도(젖소=1)"},
    "count": {"factor": None, "per": "facility", "assumed": set(), "source": "시설 수만"},
}


def apply_weight_set(sources: pd.DataFrame, name: str) -> pd.DataFrame:
    """sources.csv(species, head_count, status)에 계수 세트를 적용해 emission_weight를 다시 만든다. 휴업은 0."""
    spec = SETS[name]
    out = sources.copy()
    if spec["factor"] is None:
        weight = np.ones(len(out))
    else:
        factor = out["species"].map(spec["factor"]).fillna(0.0).to_numpy(float)
        if spec["per"] == "head":
            heads = pd.to_numeric(out["head_count"], errors="coerce").fillna(0.0).to_numpy(float)
            weight = factor * heads
        else:
            weight = factor
    weight = np.where(out["status"].astype(str).str.contains("휴업"), 0.0, weight)
    out["emission_weight"] = weight
    return out

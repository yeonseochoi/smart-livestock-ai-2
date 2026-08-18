"""1km 지역 탐색 점수로 500m 조기예측 순위를 보정하는 계층형 실험."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_odor_ai_mvp as odor
import optimize_early_prediction as optimize
import sensitivity_early_prediction as sensitivity


OUTPUT_DIR = Path("outputs/hierarchical_early_prediction")


def parent_grid(value: int) -> int:
    return int(np.floor((value + 0.5) / 2.0))


def attach_coarse_score(fine: pd.DataFrame, coarse: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    coarse_scores = coarse[["event_id", "grid_x", "grid_y"]].copy()
    coarse_scores["coarse_score"] = score
    coarse_scores = coarse_scores.rename(columns={"grid_x": "parent_x", "grid_y": "parent_y"})
    result = fine.copy()
    result["parent_x"] = result["grid_x"].map(parent_grid)
    result["parent_y"] = result["grid_y"].map(parent_grid)
    result = result.merge(coarse_scores, on=["event_id", "parent_x", "parent_y"], how="left")
    result["coarse_score"] = result["coarse_score"].fillna(0.0)
    return result


def event_percentile(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("event_id")[column].rank(method="average", pct=True)


def fit_models(
    fine_fit: pd.DataFrame, fine_eval: pd.DataFrame,
    coarse_fit: pd.DataFrame, coarse_eval: pd.DataFrame, final: bool = False,
) -> pd.DataFrame:
    fine_score, _ = optimize.fit_predict(
        "extra_leaf4", fine_fit, fine_eval, sensitivity.FEATURES, final=final
    )
    coarse_score, _ = optimize.fit_predict(
        "rank_d3", coarse_fit, coarse_eval, sensitivity.FEATURES, final=final
    )
    result = attach_coarse_score(fine_eval, coarse_eval, coarse_score)
    result["fine_score"] = fine_score
    result["fine_rank"] = event_percentile(result, "fine_score")
    result["coarse_rank"] = event_percentile(result, "coarse_score")
    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    complaints, _, _, _, _ = odor.load_inputs()
    original, _ = odor.add_grid_columns(complaints)
    _, selected_hours = odor.build_bounded_events(original)
    fine, fine_train_ids, fine_test_ids = sensitivity.build_data(
        complaints, selected_hours, 500, 30, 30
    )
    coarse, coarse_train_ids, coarse_test_ids = sensitivity.build_data(
        complaints, selected_hours, 1000, 30, 30
    )
    train_ids = fine_train_ids & coarse_train_ids
    test_ids = fine_test_ids & coarse_test_ids
    ordered = (selected_hours[selected_hours["event_id"].isin(train_ids)]
               .sort_values("event_hour")["event_id"].tolist())
    cut = max(1, int(len(ordered) * 0.8))
    fit_ids, validation_ids = set(ordered[:cut]), set(ordered[cut:])

    fine_fit, fine_validation = sensitivity.prepare_prior(
        fine[fine["event_id"].isin(fit_ids)], fine[fine["event_id"].isin(validation_ids)], fit_ids
    )
    coarse_fit, coarse_validation = sensitivity.prepare_prior(
        coarse[coarse["event_id"].isin(fit_ids)], coarse[coarse["event_id"].isin(validation_ids)], fit_ids
    )
    validation = fit_models(fine_fit, fine_validation, coarse_fit, coarse_validation)
    weights = {}
    for coarse_weight in np.linspace(0, 1, 21):
        validation["hierarchical_score"] = (
            (1 - coarse_weight) * validation["fine_rank"] + coarse_weight * validation["coarse_rank"]
        )
        weights[float(coarse_weight)] = optimize.score_prediction(
            validation, validation["hierarchical_score"].to_numpy()
        )
    selected_weight = max(weights, key=lambda value: (
        weights[value]["pr_auc"], weights[value]["topk_recall"]
    ))

    fine_train, fine_test = sensitivity.prepare_prior(
        fine[fine["event_id"].isin(train_ids)], fine[fine["event_id"].isin(test_ids)], train_ids
    )
    coarse_train, coarse_test = sensitivity.prepare_prior(
        coarse[coarse["event_id"].isin(train_ids)], coarse[coarse["event_id"].isin(test_ids)], train_ids
    )
    result = fit_models(fine_train, fine_test, coarse_train, coarse_test, final=True)
    result["hierarchical_score"] = (
        (1 - selected_weight) * result["fine_rank"] + selected_weight * result["coarse_rank"]
    )
    fine_metrics = optimize.score_prediction(result, result["fine_score"].to_numpy())
    hierarchical_metrics = optimize.score_prediction(result, result["hierarchical_score"].to_numpy())
    current = json.loads(Path("outputs/odor_ai_mvp/model_metrics.json").read_text(encoding="utf-8"))["early_prediction"]
    report = {
        "selection_policy": "1km weight selected on inner temporal validation",
        "train_events": len(train_ids),
        "test_events": len(test_ids),
        "coarse_weight": float(selected_weight),
        "validation_weights": weights,
        "fine_test": fine_metrics,
        "hierarchical_test": hierarchical_metrics,
        "current_reported": {
            "pr_auc": current["selected_model_metrics"]["pr_auc"],
            "roc_auc": current["selected_model_metrics"]["roc_auc"],
            "topk_recall": current["model_topk_recall"],
        },
    }
    (OUTPUT_DIR / "hierarchical_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result.to_csv(OUTPUT_DIR / "hierarchical_test_predictions.csv", index=False, encoding="utf-8-sig")
    print("===== 500m 계층형 예측 결과 =====")
    print(f"내부 검증 선택 1km 가중치: {selected_weight:.2f}")
    print(f"500m 단독: PR {fine_metrics['pr_auc']:.3f}, ROC {fine_metrics['roc_auc']:.3f}, "
          f"Top-K {fine_metrics['topk_recall']:.3f}")
    print(f"계층형: PR {hierarchical_metrics['pr_auc']:.3f}, ROC {hierarchical_metrics['roc_auc']:.3f}, "
          f"Top-K {hierarchical_metrics['topk_recall']:.3f}")


if __name__ == "__main__":
    main()

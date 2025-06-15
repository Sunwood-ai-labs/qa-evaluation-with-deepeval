import numpy as np
import pandas as pd
from loguru import logger

def evaluate_judge_performance(ground_truth_df, judge_results_df):
    """Judgeの判定精度を人間評価と比較"""
    human_scores = ground_truth_df["human_score"]
    judge_scores = judge_results_df["Accuracy_score"]
    correlation = np.corrcoef(human_scores, judge_scores)[0, 1]
    human_pass = (human_scores >= 0.7).astype(int)
    judge_pass = (judge_scores >= 0.7).astype(int)
    agreement_rate = (human_pass == judge_pass).mean()
    return {
        "correlation": correlation,
        "agreement_rate": agreement_rate,
        "human_mean": human_scores.mean(),
        "judge_mean": judge_scores.mean()
    }

def calibrate_judge_threshold(validation_results, target_precision=0.9):
    """バリデーション結果に基づいてJudge閾値を調整"""
    scores_and_truth = [(r['score'], r['human_label']) for r in validation_results]
    scores_and_truth.sort(reverse=True)
    best_threshold = 0.5
    best_precision = 0
    for threshold in np.arange(0.1, 1.0, 0.05):
        predictions = [1 if score >= threshold else 0 for score, _ in scores_and_truth]
        true_labels = [label for _, label in scores_and_truth]
        tp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
        fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        if precision >= target_precision and precision > best_precision:
            best_precision = precision
            best_threshold = threshold
    return best_threshold, best_precision

# サンプルデータ
ground_truth_df = pd.DataFrame({
    "human_score": [0.9, 0.8, 0.6, 0.4, 0.95]
})
judge_results_df = pd.DataFrame({
    "Accuracy_score": [0.85, 0.75, 0.65, 0.5, 0.9]
})

perf = evaluate_judge_performance(ground_truth_df, judge_results_df)
logger.info(f"Judge性能評価: {perf}")

validation_results = [
    {"score": 0.9, "human_label": 1},
    {"score": 0.8, "human_label": 1},
    {"score": 0.6, "human_label": 0},
    {"score": 0.4, "human_label": 0},
    {"score": 0.95, "human_label": 1}
]
thresh, prec = calibrate_judge_threshold(validation_results, target_precision=0.8)
logger.info(f"最適閾値: {thresh}, precision: {prec}")
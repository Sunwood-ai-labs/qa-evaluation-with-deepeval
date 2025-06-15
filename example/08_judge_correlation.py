import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from loguru import logger

# 07_analysis.pyで作成したdf_resultsをimportまたは再利用してください

judges_names = ["Accuracy", "Completeness", "Clarity", "Relevance"]

# Judge間相関分析
judge_scores = df_results[[f"{name}_score" for name in judges_names]]
correlation_matrix = judge_scores.corr()

# 相関ヒートマップ
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, 
            annot=True, 
            cmap='coolwarm', 
            center=0,
            square=True)
plt.title("Judge間スコア相関")
plt.show()

# Judge間の意見が分かれた項目の特定
def find_disagreement_cases(df, threshold=0.3):
    """Judge間でスコア差が大きい項目を特定"""
    disagreement_cases = []
    for idx, row in df.iterrows():
        scores = [row[f"{name}_score"] for name in judges_names]
        score_std = np.std(scores)
        if score_std > threshold:
            disagreement_cases.append({
                "index": idx,
                "question": row["question"],
                "answer": row["answer"],
                "scores": {name: row[f"{name}_score"] for name in judges_names},
                "std": score_std
            })
    return sorted(disagreement_cases, key=lambda x: x["std"], reverse=True)

disagreements = find_disagreement_cases(df_results)
logger.info(f"Judge間意見相違項目: {len(disagreements)}件")

# 上位5件の詳細表示
for i, case in enumerate(disagreements[:5]):
    logger.info(f"\n=== 意見相違ケース {i+1} (標準偏差: {case['std']:.3f}) ===")
    logger.info(f"質問: {case['question']}")
    logger.info(f"回答: {case['answer']}")
    for judge_name in judges_names:
        score = case['scores'][judge_name]
        reason = df_results.iloc[case['index']][f"{judge_name}_reason"]
        logger.info(f"{judge_name}: {score:.3f} - {reason}")
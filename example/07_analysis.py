import pandas as pd
import matplotlib.pyplot as plt

# 06_batch_evaluation.pyで得られたresultsをimportまたは再実行して取得してください
# ここではresultsが既に存在している前提です

# 結果をDataFrameに変換
results_data = []
for result in results.test_results:
    row = {
        "question": result.input,
        "answer": result.actual_output,
        "overall_success": result.success
    }
    for metric_data in result.metrics_data:
        row[f"{metric_data.name}_score"] = metric_data.score
        row[f"{metric_data.name}_success"] = metric_data.success
        row[f"{metric_data.name}_reason"] = getattr(metric_data, 'reason', '')
    results_data.append(row)

df_results = pd.DataFrame(results_data)

# スコア分布の可視化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
judges_names = ["Accuracy", "Completeness", "Clarity", "Relevance"]

for i, judge_name in enumerate(judges_names):
    ax = axes[i//2, i%2]
    df_results[f"{judge_name}_score"].hist(bins=20, ax=ax)
    ax.set_title(f"{judge_name} Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")

plt.tight_layout()
plt.show()

# 低スコア項目の詳細確認
low_score_threshold = 0.6
for judge_name in judges_names:
    low_scores = df_results[df_results[f"{judge_name}_score"] < low_score_threshold]
    if not low_scores.empty:
        print(f"\n=== {judge_name} 低スコア項目 ===")
        for idx, row in low_scores.iterrows():
            print(f"質問: {row['question']}")
            print(f"回答: {row['answer']}")
            print(f"スコア: {row[f'{judge_name}_score']}")
            print(f"理由: {row[f'{judge_name}_reason']}")
            print("-" * 50)
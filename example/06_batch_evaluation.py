import pandas as pd
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset
from deepeval import evaluate

# CSVファイル読み込み
df = pd.read_csv("qa_dataset.csv")

# テストケース作成
test_cases = []
for _, row in df.iterrows():
    test_case = LLMTestCase(
        input=row["question"],
        actual_output=row["llm_answer"],
        expected_output=row["expected_answer"],
        retrieval_context=row.get("context", "").split("|||") if pd.notna(row.get("context")) else []
    )
    test_cases.append(test_case)

# データセット作成
dataset = EvaluationDataset(test_cases=test_cases)

# 複数Judge定義
accuracy_judge = GEval(
    name="Accuracy",
    criteria="回答が事実的に正確であるかを評価する",
    evaluation_steps=[
        "回答に含まれる事実を特定する",
        "各事実の正確性を検証する",
        "誤った情報や矛盾がないか確認する",
        "全体的な正確性を0-1で評価する"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.8
)
completeness_judge = GEval(
    name="Completeness",
    criteria="回答が質問に対して十分に完全で包括的かを評価する",
    evaluation_steps=[
        "質問が求めている情報を特定する",
        "回答がその情報を含んでいるか確認する",
        "重要な情報の欠落がないか評価する",
        "期待される回答と比較して完全性を判定する"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7
)
clarity_judge = GEval(
    name="Clarity",
    criteria="回答が分かりやすく、理解しやすい形で表現されているかを評価する",
    evaluation_steps=[
        "回答の文章構造を分析する",
        "専門用語の使用が適切か確認する",
        "論理的な流れがあるか評価する",
        "読み手にとっての理解しやすさを判定する"
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.6
)
relevance_judge = GEval(
    name="Relevance",
    criteria="回答が質問に直接関連しており、的確に答えているかを評価する",
    evaluation_steps=[
        "質問の意図を理解する",
        "回答が質問に直接対応しているか確認する",
        "無関係な情報が含まれていないか評価する",
        "質問と回答の関連度を測定する"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.8
)

judges = [accuracy_judge, completeness_judge, clarity_judge, relevance_judge]

# バッチ評価実行
results = evaluate(
    dataset=dataset,
    metrics=judges,
    max_concurrent=3,
    show_indicator=True
)

print(f"全体スコア: {results.overall_score}")
print(f"評価完了件数: {len(results.test_results)}")
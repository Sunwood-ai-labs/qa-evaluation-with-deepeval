from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

# LLM as a Judge メトリック定義
correctness_judge = GEval(
    name="Correctness",
    criteria="回答が質問に対して事実的に正確で完全かどうかを評価する",
    evaluation_steps=[
        "質問の要求内容を理解する",
        "実際の回答の内容を分析する",
        "期待される回答と比較する",
        "事実の正確性を確認する",
        "0-1のスコアで評価する"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7,
    model="gpt-4o"
)

# テストケース
test_case = LLMTestCase(
    input="日本の首都はどこですか？",
    actual_output="日本の首都は東京です。",
    expected_output="日本の首都は東京都です。人口は約1400万人で、政治・経済の中心地です。"
)

# 評価実行
correctness_judge.measure(test_case)
print(f"スコア: {correctness_judge.score}")
print(f"理由: {correctness_judge.reason}")
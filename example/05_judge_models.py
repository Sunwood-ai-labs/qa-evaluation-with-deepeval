from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval

# GPT-4による高精度評価
gpt4_judge = GEval(
    name="GPT4Correctness",
    criteria="回答の正確性を厳密に評価する",
    evaluation_steps=[
        "回答内容を詳細に分析する",
        "事実の正確性を検証する",
        "論理的一貫性を確認する"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o",
    threshold=0.8
)

# Claude による評価
claude_judge = GEval(
    name="ClaudeCorrectness", 
    criteria="回答の正確性を評価する",
    evaluation_steps=[
        "回答の事実確認を行う",
        "質問との関連性を評価する"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="claude-3-sonnet-20240229",
    threshold=0.8
)

# ローカルモデルによる評価
local_judge = GEval(
    name="LocalModelJudge",
    criteria="回答品質を評価する",
    evaluation_steps=[
        "内容の適切性を確認する",
        "質問への回答度を評価する"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="ollama/llama2",
    threshold=0.7
)

# サンプルテストケース
test_case = LLMTestCase(
    input="地球の公転周期は？",
    actual_output="地球の公転周期は約365日です。",
    expected_output="地球の公転周期は約365.25日で、これがうるう年の理由です。"
)

# 各Judgeで評価
for judge in [gpt4_judge, claude_judge, local_judge]:
    judge.measure(test_case)
    print(f"{judge.name} スコア: {judge.score}")
    print(f"{judge.name} 理由: {judge.reason}")
    print("-" * 40)
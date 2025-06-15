from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from loguru import logger

# 忠実性評価（Faithfulness）
faithfulness_judge = GEval(
    name="Faithfulness",
    criteria="回答が提供された文脈情報に忠実で、文脈から逸脱した情報を含んでいないかを評価する",
    evaluation_steps=[
        "提供された文脈情報を確認する",
        "回答の各部分が文脈に基づいているか検証する",
        "文脈にない情報で回答していないか確認する",
        "幻覚（hallucination）がないか評価する"
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT
    ],
    threshold=0.8
)

# 文脈活用度評価
context_utilization_judge = GEval(
    name="ContextUtilization",
    criteria="提供された文脈情報を適切に活用して回答しているかを評価する",
    evaluation_steps=[
        "文脈情報の関連部分を特定する",
        "回答が文脈の重要な情報を活用しているか確認する",
        "文脈から得られる最適な回答と比較する",
        "文脈活用の効果性を評価する"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT
    ],
    threshold=0.7
)

# RAGテストケース例
rag_test_case = LLMTestCase(
    input="機械学習の定義を教えてください",
    actual_output="機械学習は、コンピュータがデータから自動的にパターンを学習し、予測や決定を行う人工知能の一分野です。",
    expected_output="機械学習は人工知能の一分野で、明示的にプログラムされることなく学習する能力をコンピュータに与える研究分野です。",
    retrieval_context=[
        "機械学習（Machine Learning）は人工知能の一分野で、コンピュータシステムが明示的にプログラムされることなく、データを使用して自動的に学習し改善する能力を指します。",
        "機械学習アルゴリズムは、データの中からパターンを見つけ、そのパターンを使って予測や決定を行います。"
    ]
)

# 各Judgeで評価
for judge in [faithfulness_judge, context_utilization_judge]:
    judge.measure(rag_test_case)
    logger.info(f"{judge.name} スコア: {judge.score}")
    logger.info(f"{judge.name} 理由: {judge.reason}")
    logger.info("-" * 40)
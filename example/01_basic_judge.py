from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from loguru import logger
import sys
from dotenv import load_dotenv

# .envファイルをロード
load_dotenv()

# LLM as a Judge メトリック定義
correctness_judge = GEval(
    name="Correctness",
    criteria="""
    回答が質問に対して事実的に正確で完全かどうかを評価する
    
    評価は日本語で理由を説明してください。
    """,
    evaluation_steps=[
        "質問の要求内容を理解する",
        "実際の回答の内容を分析する",
        "期待される回答と比較する",
        "事実の正確性を確認する",
        "0-5のスコアで評価する"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7,
    model="gpt-4o-mini"
)

# テストケース
test_case = LLMTestCase(
    input="日本の首都はどこですか？",
    actual_output="日本の首都は東京です。",
    expected_output="日本の首都は東京都です。人口は約1400万人で、政治・経済の中心地です。"
)

# 評価実行
logger.info("🤖 LLM Judge評価を開始します")
try:
    correctness_judge.measure(test_case)
    logger.success(f"✅ 評価完了 - スコア: {correctness_judge.score}")
    logger.info(f"💭 評価理由: {correctness_judge.reason}")
except Exception as e:
    logger.error(f"❌ 評価中にエラーが発生しました: {e}")
    raise
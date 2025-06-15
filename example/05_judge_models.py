from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from loguru import logger
import sys

# Configure loguru for stylish output
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("model_comparison.log", rotation="1 MB")

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
    model="gpt-4o-mini",
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
    model="gpt-4o-mini",
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
    model="o3-mini",
    threshold=0.7
)

# サンプルテストケース
test_case = LLMTestCase(
    input="地球の公転周期は？",
    actual_output="地球の公転周期は約365日です。",
    expected_output="地球の公転周期は約365.25日で、これがうるう年の理由です。"
)

# 各Judgeで評価
logger.info("🔍 複数モデルでの評価を開始します")

for i, judge in enumerate([gpt4_judge, claude_judge, local_judge], 1):
    logger.info(f"🤖 {i}/3: {judge.name}で評価中...")
    try:
        judge.measure(test_case)
        logger.success(f"✅ {judge.name} - スコア: {judge.score:.3f}")
        logger.info(f"💭 {judge.name} 理由: {judge.reason}")
        logger.info("─" * 50)
    except Exception as e:
        logger.error(f"❌ {judge.name}でエラー: {e}")
        continue

logger.success("🎉 全てのモデル評価が完了しました！")
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from loguru import logger
import sys
import os
from dotenv import load_dotenv

# .envファイルをロード
load_dotenv()

# Configure loguru for stylish Japanese output
logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)
logger.add("japanese_evaluation.log", rotation="1 MB", encoding="utf-8")

# # 環境変数でOpenAIの設定を上書き（LiteLLM用）
# if os.environ.get("LITELLM_BASE_URL"):
#     os.environ["OPENAI_API_BASE"] = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
#     os.environ["OPENAI_API_KEY"] = os.environ.get("LITELLM_API_KEY", "your-api-key")
#     logger.info("🔧 LiteLLM設定を適用しました")

# 日本語専用のGEval（カスタムテンプレートを使わずに、criteriaとevaluation_stepsで日本語を強制）
correctness_judge = GEval(
    name="Correctness",
    criteria="""
    あなたは優秀な日本語評価者です。以下の基準で評価してください：
    
    回答が質問に対して事実的に正確で完全かどうかを評価してください。
    
    重要な指示：
    - 評価理由は必ず日本語で詳しく説明してください
    - 日本語以外での説明は絶対に禁止です
    - 以下の観点から評価してください：
      * 事実の正確性
      * 回答の完全性  
      * 期待される回答との一致度
    
    最終的な評価理由は日本語で書いてください。
    """,
    evaluation_steps=[
        "質問の要求内容を日本語で理解し、何が求められているかを明確にする",
        "実際の回答の内容を日本語で詳しく分析し、含まれている情報を整理する", 
        "期待される回答と実際の回答を日本語で比較し、差異を特定する",
        "事実の正確性を日本語で確認し、間違いや不足がないかチェックする",
        "0-10のスコアで評価し、その理由を必ず日本語で詳しく説明する",
        "評価の最終結果を日本語でまとめ、改善点があれば日本語で提案する"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.7,
    model="o3-mini"
)

# テストケース
test_case = LLMTestCase(
    input="日本の首都はどこですか？",
    actual_output="日本の首都は東京です。",
    expected_output="日本の首都は東京都です。人口は約1400万人で、政治・経済の中心地です。"
)

# 評価実行
logger.info("🤖 日本語LLM Judge評価を開始します")

try:
    logger.debug("📋 テストケース情報:")
    logger.debug(f"   入力: {test_case.input}")
    logger.debug(f"   実際の出力: {test_case.actual_output}")
    logger.debug(f"   期待される出力: {test_case.expected_output}")
    
    logger.info("⚡ 評価処理中...")
    correctness_judge.measure(test_case)
    
    # 評価結果をloguruで美しく表示
    logger.success("✅ 評価が正常に完了しました")
    
    # スコア情報
    score = correctness_judge.score
    threshold = correctness_judge.threshold
    passed = score >= threshold
    
    logger.info("📊 評価結果サマリー:")
    logger.info(f"   🎯 スコア: {score:.3f}")
    logger.info(f"   🎚️  しきい値: {threshold}")
    logger.info(f"   🏆 判定: {'✅ 合格' if passed else '❌ 不合格'}")
    
    # 評価理由を詳細表示
    logger.info("📝 評価理由:")
    reason_lines = correctness_judge.reason.split('\n')
    for line in reason_lines:
        if line.strip():
            logger.info(f"   {line.strip()}")
    
    # 成功ログ
    if passed:
        logger.success(f"🎉 テストケースが合格しました (スコア: {score:.3f} >= しきい値: {threshold})")
    else:
        logger.warning(f"⚠️  テストケースが不合格です (スコア: {score:.3f} < しきい値: {threshold})")
        
except Exception as e:
    logger.error(f"❌ 評価中にエラーが発生しました")
    logger.error(f"   エラー詳細: {str(e)}")
    logger.error(f"   エラータイプ: {type(e).__name__}")
    raise

logger.info("🏁 評価処理が完了しました")
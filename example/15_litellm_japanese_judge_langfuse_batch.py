import pandas as pd
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from loguru import logger
import sys
import os
from dotenv import load_dotenv
from litellm_model import LiteLLMModel
from langfuse import Langfuse

# .envファイルをロード
load_dotenv()

# loguruの日本語カスタム出力設定
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)
logger.add("litellm_japanese_evaluation_batch.log", rotation="1 MB", encoding="utf-8")

# Langfuseクライアント初期化
langfuse = Langfuse(
    secret_key=os.environ.get("LANGFUSE_SECRET_KEY", ""),
    public_key=os.environ.get("LANGFUSE_PUBLIC_KEY", ""),
    host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
)

def create_test_cases_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    test_cases = []
    for _, row in df.iterrows():
        test_case = LLMTestCase(
            input=row["question"],
            actual_output=row["llm_answer"],
            expected_output=row["expected_answer"],
            retrieval_context=row.get("context", "").split("|||") if pd.notna(row.get("context")) else []
        )
        test_cases.append(test_case)
    return test_cases

def main():
    # LiteLLMモデル情報を環境変数から取得
    model_name = os.environ.get("LITELLM_MODEL", "gpt-4o-mini")
    api_base = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
    api_key = os.environ.get("LITELLM_API_KEY", "your-api-key")

    # LiteLLMモデルインスタンス作成
    custom_model = LiteLLMModel(
        model_name=model_name,
        base_url=api_base,
        api_key=api_key
    )

    # 日本語専用GEvalメトリック
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
        threshold=0.3,
        model=custom_model
    )

    # テストケースをCSVから生成
    test_cases = create_test_cases_from_csv("example/qa_dataset.csv")

    logger.info(f"📝 テストケース数: {len(test_cases)} 件")
    success_count = 0
    fail_count = 0

    for idx, test_case in enumerate(test_cases, 1):
        logger.info(f"==== {idx}件目の評価を開始 ====")
        try:
            logger.debug(f"   入力: {test_case.input}")
            logger.debug(f"   実際の出力: {test_case.actual_output}")
            logger.debug(f"   期待される出力: {test_case.expected_output}")

            logger.info("⚡ 評価処理中...")
            correctness_judge.measure(test_case)

            score = correctness_judge.score
            threshold = correctness_judge.threshold
            passed = score >= threshold
            reason = correctness_judge.reason

            logger.info("📊 評価結果サマリー:")
            logger.info(f"   🎯 スコア: {score:.3f}")
            logger.info(f"   🎚️  しきい値: {threshold}")
            logger.info(f"   🏆 判定: {'✅ 合格' if passed else '❌ 不合格'}")

            logger.info("📝 評価理由:")
            for line in reason.split('\n'):
                if line.strip():
                    logger.info(f"   {line.strip()}")

            # Langfuseに評価結果を送信
            logger.info("🌐 Langfuseに評価結果を送信します")
            with langfuse.start_as_current_span(name="LiteLLM日本語Judge評価 Case15") as span:
                span.update_trace(
                    input={
                        "input": test_case.input,
                        "actual_output": test_case.actual_output,
                        "expected_output": test_case.expected_output
                    },
                    output={
                        "score": score,
                        "threshold": threshold,
                        "passed": passed,
                        "reason": reason
                    },
                    metadata={
                        "model": model_name,
                        "evaluation_type": "correctness",
                        "language": "japanese"
                    }
                )
                langfuse.score_current_trace(
                    name="correctness_score",
                    value=score,
                    data_type="NUMERIC",
                    comment=f"評価理由: {reason[:100]}..."
                )
            logger.success("🚀 Langfuseへの送信が完了しました")

            if passed:
                logger.success(f"🎉 テストケースが合格しました (スコア: {score:.3f} >= しきい値: {threshold})")
                success_count += 1
            else:
                logger.warning(f"⚠️  テストケースが不合格です (スコア: {score:.3f} < しきい値: {threshold})")
                fail_count += 1

        except Exception as e:
            logger.error(f"❌ 評価中にエラーが発生しました")
            logger.error(f"   エラー詳細: {str(e)}")
            logger.error(f"   エラータイプ: {type(e).__name__}")
            fail_count += 1
        finally:
            try:
                langfuse.flush()
                logger.info("🧹 Langfuseデータの送信完了を確認しました")
            except Exception as e:
                logger.warning(f"⚠️ Langfuseのflush中にエラー: {str(e)}")

    logger.info("🏁 バッチ評価処理が完了しました")
    logger.info(f"✅ 合格: {success_count}件, ❌ 不合格: {fail_count}件, 合計: {len(test_cases)}件")

if __name__ == "__main__":
    main()
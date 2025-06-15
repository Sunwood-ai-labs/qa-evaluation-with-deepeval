from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset
from deepeval import evaluate
from loguru import logger
import sys
import json

# Configure loguru for stylish output
logger.remove()
logger.add(
    sys.stderr, 
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add("pipeline.log", rotation="1 MB")

class QAEvaluationPipeline:
    """本番環境でのQA評価パイプライン"""
    def __init__(self, judge_configs):
        self.judges = [self._create_judge(config) for config in judge_configs]
    def _create_judge(self, config):
        return GEval(
            name=config["name"],
            criteria=config["criteria"],
            evaluation_steps=config["steps"],
            evaluation_params=config["params"],
            threshold=config["threshold"],
            model=config.get("model", "gpt-4o-mini")
        )
    def evaluate_qa_batch(self, qa_pairs):
        """QAペアのバッチ評価"""
        logger.info(f"🚀 パイプライン評価を開始 - {len(qa_pairs)}件のQAペアを処理")
        
        test_cases = []
        for i, qa in enumerate(qa_pairs, 1):
            logger.debug(f"📄 {i}/{len(qa_pairs)}: テストケースを作成中...")
            test_cases.append(
                LLMTestCase(
                    input=qa["question"],
                    actual_output=qa["answer"],
                    expected_output=qa.get("expected", ""),
                    retrieval_context=qa.get("context", [])
                )
            )
        
        dataset = EvaluationDataset(test_cases=test_cases)
        logger.info(f"🤖 {len(self.judges)}個のJudgeで評価実行中...")
        
        try:
            results = evaluate(test_cases=test_cases, metrics=self.judges)
            logger.success("✅ バッチ評価が完了しました！")
            return self._format_results(results)
        except Exception as e:
            logger.error(f"❌ バッチ評価中にエラー: {e}")
            raise
    def _format_results(self, results):
        """結果のフォーマット"""
        logger.info("📊 評価結果をフォーマット中...")
        
        # Calculate overall metrics
        total_tests = len(results.test_results)
        passed_tests = sum(1 for result in results.test_results if result.success)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        # Calculate average score across all metrics
        all_scores = []
        for result in results.test_results:
            for metric_data in result.metrics_data:
                if hasattr(metric_data, 'score') and metric_data.score is not None:
                    all_scores.append(metric_data.score)
        overall_score = sum(all_scores) / len(all_scores) if all_scores else 0
        
        formatted = {
            "overall_score": overall_score,
            "success_rate": success_rate,
            "detailed_results": []
        }
        
        logger.info(f"🎯 全体スコア: {overall_score:.3f}")
        logger.info(f"🎆 成功率: {success_rate:.1%}")
        
        for i, result in enumerate(results.test_results, 1):
            detailed = {
                "question": result.input,
                "answer": result.actual_output,
                "overall_success": result.success,
                "judge_scores": {}
            }
            
            logger.debug(f"📁 {i}: 詳細結果を処理中...")
            for metric_data in result.metrics_data:
                detailed["judge_scores"][metric_data.name] = {
                    "score": metric_data.score,
                    "success": metric_data.success,
                    "reason": getattr(metric_data, 'reason', '')
                }
            formatted["detailed_results"].append(detailed)
        
        return formatted

# 使用例
judge_configs = [
    {
        "name": "Accuracy",
        "criteria": "回答の事実的正確性を評価",
        "steps": ["事実確認", "正確性判定"],
        "params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        "threshold": 0.8
    },
    {
        "name": "Helpfulness", 
        "criteria": "回答の有用性を評価",
        "steps": ["有用性確認", "実用性判定"],
        "params": [LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        "threshold": 0.7
    }
]

pipeline = QAEvaluationPipeline(judge_configs)

# 評価実行
qa_data = [
    {
        "question": "Python の特徴は？",
        "answer": "Pythonは読みやすく、学習しやすいプログラミング言語です。",
        "expected": "Pythonは高水準プログラミング言語で、シンプルな文法が特徴です。"
    }
]

try:
    evaluation_results = pipeline.evaluate_qa_batch(qa_data)
    
    logger.success(f"🎉 評価完了 - 全体スコア: {evaluation_results['overall_score']:.3f}")
    logger.info(f"📊 成功率: {evaluation_results['success_rate']:.1%}")
    
    # 結果をJSON形式で表示
    logger.info("📋 詳細結果:")
    print(json.dumps(evaluation_results, ensure_ascii=False, indent=2))
    
except Exception as e:
    logger.error(f"❌ パイプライン実行中にエラー: {e}")
    raise
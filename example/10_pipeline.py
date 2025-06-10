from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.dataset import EvaluationDataset
from deepeval import evaluate

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
            model=config.get("model", "o3")
        )
    def evaluate_qa_batch(self, qa_pairs):
        """QAペアのバッチ評価"""
        test_cases = [
            LLMTestCase(
                input=qa["question"],
                actual_output=qa["answer"],
                expected_output=qa.get("expected", ""),
                retrieval_context=qa.get("context", [])
            )
            for qa in qa_pairs
        ]
        dataset = EvaluationDataset(test_cases=test_cases)
        results = evaluate(dataset=dataset, metrics=self.judges)
        return self._format_results(results)
    def _format_results(self, results):
        """結果のフォーマット"""
        formatted = {
            "overall_score": results.overall_score,
            "success_rate": results.success_rate,
            "detailed_results": []
        }
        for result in results.test_results:
            detailed = {
                "question": result.input,
                "answer": result.actual_output,
                "overall_success": result.success,
                "judge_scores": {}
            }
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

evaluation_results = pipeline.evaluate_qa_batch(qa_data)
print(f"評価完了: {evaluation_results['overall_score']}")
print(evaluation_results)
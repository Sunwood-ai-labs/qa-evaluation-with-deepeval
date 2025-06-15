from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from loguru import logger
import sys
import os
import requests
from dotenv import load_dotenv

# .envファイルをロード
load_dotenv()

# LiteLLMカスタムモデル
class LiteLLMModel(DeepEvalBaseLLM):
    def __init__(self, model_name, base_url, api_key):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
    
    def load_model(self):
        # LiteLLMは外部APIなので、ここでは設定を返すだけ
        return {
            "model": self.model_name,
            "base_url": self.base_url,
            "api_key": self.api_key
        }
    
    def generate(self, prompt: str) -> str:
        config = self.load_model()
        
        # OpenAI互換APIでLiteLLMにリクエスト
        url = f"{config['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": config["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LiteLLM API error: {e}")
            raise
    
    async def a_generate(self, prompt: str) -> str:
        # 同期版を再利用（簡単な実装）
        return self.generate(prompt)
    
    def get_model_name(self):
        return f"LiteLLM-{self.model_name}"

# 環境変数からモデル情報を取得
model_name = os.environ.get("LITELLM_MODEL", "gpt-4o-mini")
api_base = os.environ.get("LITELLM_BASE_URL", "http://localhost:4000")
api_key = os.environ.get("LITELLM_API_KEY", "your-api-key")

# カスタムLiteLLMモデルのインスタンス作成
custom_model = LiteLLMModel(
    model_name=model_name,
    base_url=api_base,
    api_key=api_key
)

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
    model=custom_model  # カスタムモデルインスタンスを使用
)

# テストケース
test_case = LLMTestCase(
    input="日本の首都はどこですか？",
    actual_output="日本の首都は東京です。",
    expected_output="日本の首都は東京都です。人口は約1400万人で、政治・経済の中心地です。"
)

# 評価実行
logger.info("🤖 LiteLLM Judge評価を開始します")
try:
    correctness_judge.measure(test_case)
    logger.success(f"✅ 評価完了 - スコア: {correctness_judge.score}")
    logger.info(f"💭 評価理由: {correctness_judge.reason}")
except Exception as e:
    logger.error(f"❌ 評価中にエラーが発生しました: {e}")
    raise
from deepeval.models.base_model import DeepEvalBaseLLM
from loguru import logger
import requests

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
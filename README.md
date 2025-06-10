# DeepEval QA LLM as a Judge サンプル集

このリポジトリは、DeepEvalを用いた「LLM as a Judge」評価の実践例をまとめたものです。  
各種評価観点・バッチ評価・分析・パイプライン化まで、example/配下にスクリプトを分割して収録しています。

## セットアップ

### 1. リポジトリをクローン
```bash
git clone (このリポジトリのURL)
cd (このリポジトリ)
```

### 2. .envファイルを作成
`.env.example` をコピーし、OpenAI APIキーを設定してください。
```bash
cp .env.example .env
# .envを編集してOPENAI_API_KEYを入力
```

### 3. Docker環境構築
```bash
docker compose build
docker compose up -d
docker compose exec app bash
# 以降、/workspace/example/ で作業
```

## exampleスクリプトの実行例

```bash
cd /workspace/example

# 基本的なJudge評価
python 01_basic_judge.py

# 複数観点での評価
python 02_multi_metrics.py

# RAG向け評価
python 03_rag_judge.py

# カスタムJudge
python 04_custom_judges.py

# Judgeモデル指定
python 05_judge_models.py

# バッチ評価
python 06_batch_evaluation.py

# 評価結果の分析
python 07_analysis.py

# Judge間の一致度分析
python 08_judge_correlation.py

# 継続的モニタリング
python 09_judge_monitoring.py

# パイプライン実装例
python 10_pipeline.py
```

## 依存ライブラリ

- deepeval
- openai
- pandas
- matplotlib
- seaborn

（全て requirements.txt で管理）

## ディレクトリ構成

```
.
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── example/
│   ├── 01_basic_judge.py
│   ├── 02_multi_metrics.py
│   ├── 03_rag_judge.py
│   ├── 04_custom_judges.py
│   ├── 05_judge_models.py
│   ├── 06_batch_evaluation.py
│   ├── 07_analysis.py
│   ├── 08_judge_correlation.py
│   ├── 09_judge_monitoring.py
│   ├── 10_pipeline.py
│   └── qa_dataset.csv
```

## 注意事項

- OpenAI APIキーが必要です（gpt-4o等を利用）。
- Claudeやローカルモデルを利用する場合は、各自API設定やローカル環境の準備が必要です。
- サンプルは教育・検証用途です。本番利用時はAPIコストやセキュリティにご注意ください。

---
DeepEval公式: https://github.com/confident-ai/deepeval
<div align="center">

![](header.png)

  <h1>🤖 DeepEval QA LLM as a Judge Sample</h1>
  
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
    <img src="https://img.shields.io/badge/DeepEval-Latest-green.svg" alt="DeepEval">
    <img src="https://img.shields.io/badge/Docker-Supported-blue.svg" alt="Docker">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
  </p>
</div>

## 🎯 概要

このリポジトリは、DeepEvalを用いた「LLM as a Judge」評価の実践例をまとめたものです。  
各種評価観点・バッチ評価・分析・パイプライン化まで、[example/](./example/)配下にスクリプトを分割して収録しています。

## 🚀 クイックスタート

詳細なセットアップ手順は [SETUP.md](./SETUP.md) をご参照ください。

### ⚡ 簡単セットアップ

```bash
# 1. リポジトリをクローン
git clone (このリポジトリのURL)
cd (このリポジトリ)

# 2. 環境変数設定
cp .env.example .env
# .envを編集してOPENAI_API_KEYを入力

# 3. Docker環境構築
docker compose up -d
docker compose exec app bash
```

## 📚 使用方法

各スクリプトの詳細な使用方法は [example/README.md](./example/README.md) をご参照ください。

### 🏃‍♂️ 実行例

```bash
cd /workspace/example

# 🎯 基本的なJudge評価
python 01_basic_judge.py

# 📊 複数観点での評価
python 02_multi_metrics.py

# 🔍 RAG向け評価
python 03_rag_judge.py

# 🛠️ カスタムJudge
python 04_custom_judges.py

# 🎮 Judgeモデル指定
python 05_judge_models.py

# 📦 バッチ評価
python 06_batch_evaluation.py

# 📈 評価結果の分析
python 07_analysis.py

# 🔗 Judge間の一致度分析
python 08_judge_correlation.py

# 📡 継続的モニタリング
python 09_judge_monitoring.py

# 🚀 パイプライン実装例
python 10_pipeline.py
```

## 🔧 技術スタック

| 技術 | 用途 | バージョン |
|------|------|-----------|
| 🐍 **DeepEval** | LLM評価フレームワーク | Latest |
| 🤖 **OpenAI** | LLMモデル | GPT-4o-mini |
| 📊 **Pandas** | データ処理 | Latest |
| 📈 **Matplotlib** | グラフ描画 | Latest |
| 🎨 **Seaborn** | 統計可視化 | Latest |

全ての依存関係は [requirements.txt](./requirements.txt) で管理されています。

## 📂 プロジェクト構造

```
📁 qa-evaluation-with-deepeval/
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml
├── 📋 requirements.txt
├── ⚙️ .env.example
├── 📖 SETUP.md
└── 📁 example/
    ├── 📖 README.md
    ├── 🎯 01_basic_judge.py
    ├── 📊 02_multi_metrics.py
    ├── 🔍 03_rag_judge.py
    ├── 🛠️ 04_custom_judges.py
    ├── 🎮 05_judge_models.py
    ├── 📦 06_batch_evaluation.py
    ├── 📈 07_analysis.py
    ├── 🔗 08_judge_correlation.py
    ├── 📡 09_judge_monitoring.py
    ├── 🚀 10_pipeline.py
    └── 📊 qa_dataset.csv
```

## ⚠️ 重要な注意事項

- 🔑 OpenAI APIキーが必要です（GPT-4o-mini等を利用）
- 🎨 Claude や ローカルモデルを利用する場合は、各自API設定やローカル環境の準備が必要です
- 📚 サンプルは教育・検証用途です。本番利用時はAPIコストやセキュリティにご注意ください

## 🔗 関連リンク

- 📖 [DeepEval 公式ドキュメント](https://github.com/confident-ai/deepeval)
- 🚀 [セットアップ詳細](./SETUP.md)
- 📝 [サンプルスクリプト詳細](./example/README.md)

---

<div align="center">
  Made with ❤️ using DeepEval
</div>
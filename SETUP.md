# 🚀 セットアップガイド

<div align="center">
  <p>
    <a href="README.md">← メインREADMEに戻る</a>
  </p>
</div>

このガイドでは、DeepEval QA LLM as a Judge サンプル集の環境構築手順を詳しく説明します。

## 📋 前提条件

### 🔧 必要なソフトウェア
- **Docker**: 20.10.0 以上
- **Docker Compose**: 1.29.0 以上
- **Git**: 2.25.0 以上

### 🔑 APIキーの準備
以下のいずれかのAPIキーが必要です：
- **OpenAI API キー** (推奨)
- **Anthropic API キー** (Claude使用時)
- **その他LLMプロバイダーのAPIキー**

## ⚡ クイックスタート

### 1. リポジトリのクローン
```bash
git clone https://github.com/your-username/qa-evaluation-with-deepeval.git
cd qa-evaluation-with-deepeval
```

### 2. 環境変数の設定
```bash
# .env.exampleをコピーして.envを作成
cp .env.example .env

# エディタで.envを編集
nano .env
```

**設定例:**
```bash
# OpenAI APIキーを設定
OPENAI_API_KEY=sk-your-openai-api-key-here

# 他のAPIキー（必要に応じて）
ANTHROPIC_API_KEY=your-anthropic-api-key-here
```

### 3. Docker環境の構築と起動
```bash
# Dockerイメージのビルド
docker compose build

# コンテナの起動
docker compose up -d

# 起動確認
docker compose ps
```

### 4. コンテナへの接続
```bash
# アプリケーションコンテナに接続
docker compose exec app bash

# 作業ディレクトリに移動
cd /workspace/example
```

### 5. 動作確認
```bash
# 基本的なサンプルを実行
python 01_basic_judge.py

# 成功すると以下のような出力が表示されます
# スコア: 0.85
# 理由: 回答は正確で、日本の首都が東京であることを適切に示している...
```

## 🔧 詳細設定

### 環境変数の詳細

| 変数名 | 説明 | 必須 | デフォルト値 |
|--------|------|------|-------------|
| `OPENAI_API_KEY` | OpenAI APIキー | ✅ | なし |
| `ANTHROPIC_API_KEY` | Anthropic APIキー | ⭕ | なし |
| `DEEPEVAL_API_KEY` | DeepEval APIキー | ⭕ | なし |

### Docker Compose設定

**主要なサービス:**
- **app**: メインアプリケーション
- **jupyter** (オプション): Jupyter Notebook環境

**ポート設定:**
- `8888`: Jupyter Notebook (有効化時)
- `8000`: アプリケーション (有効化時)

## 🐳 Docker環境の詳細

### コンテナ構成
```
qa-evaluation-with-deepeval/
├── app (メインコンテナ)
│   ├── Python 3.8+
│   ├── DeepEval
│   ├── 必要なライブラリ
│   └── /workspace (作業ディレクトリ)
```

### ボリュームマウント
- `./`: `/workspace` - プロジェクトルートを作業ディレクトリにマウント
- `./.env`: `/workspace/.env` - 環境変数ファイル

### 便利なコマンド

#### コンテナ管理
```bash
# コンテナの起動
docker compose up -d

# コンテナの停止
docker compose down

# コンテナの再起動
docker compose restart

# ログの確認
docker compose logs -f app
```

#### 開発作業
```bash
# 新しいターミナルでコンテナに接続
docker compose exec app bash

# Pythonの対話シェル
docker compose exec app python

# 依存関係の更新
docker compose exec app pip install -r requirements.txt
```

## 🏃‍♂️ 実行パターン

### 📚 学習目的での実行
```bash
# 基本から順番に実行
python 01_basic_judge.py
python 02_multi_metrics.py
python 03_rag_judge.py
```

### 🔍 RAGシステムの評価
```bash
# RAG特化の評価
python 03_rag_judge.py
python 06_batch_evaluation.py
```

### 📊 本格的な分析
```bash
# バッチ評価 → 分析の流れ
python 06_batch_evaluation.py
python 07_analysis.py
python 08_judge_correlation.py
```

## 🚨 トラブルシューティング

### よくある問題と解決方法

#### 1. APIキーエラー
**エラー:** `Authentication Error` または `Invalid API Key`

**解決方法:**
```bash
# .envファイルの確認
cat .env

# APIキーの形式確認（OpenAI）
# 正しい形式: sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

#### 2. Docker起動エラー
**エラー:** `port already in use` または `container name already exists`

**解決方法:**
```bash
# 既存コンテナの停止・削除
docker compose down
docker system prune -f

# 再起動
docker compose up -d
```

#### 3. 依存関係エラー
**エラー:** `ModuleNotFoundError` または `ImportError`

**解決方法:**
```bash
# コンテナ内で依存関係を再インストール
docker compose exec app pip install -r requirements.txt

# または、イメージの再ビルド
docker compose build --no-cache
```

#### 4. 権限エラー
**エラー:** `Permission denied` または `Access denied`

**解決方法:**
```bash
# ホスト側でのファイル権限修正
sudo chown -R $USER:$USER .

# Docker内での権限確認
docker compose exec app ls -la /workspace
```

### 🔍 デバッグモード

#### 詳細ログの有効化
```bash
# 環境変数でログレベルを設定
export DEEPEVAL_LOG_LEVEL=DEBUG

# または.envファイルに追加
echo "DEEPEVAL_LOG_LEVEL=DEBUG" >> .env
```

#### 対話的デバッグ
```bash
# Python対話シェルでの実行
docker compose exec app python
>>> from deepeval.test_case import LLMTestCase
>>> # 対話的にテスト
```

## 📊 リソース使用量の目安

### システム要件
- **CPU**: 2コア以上推奨
- **メモリ**: 4GB以上推奨
- **ディスク**: 10GB以上の空き容量

### API使用量の目安
- **基本サンプル**: ~$0.01/実行
- **バッチ評価**: ~$0.10-1.00/100ケース
- **分析処理**: 追加料金なし（ローカル処理）

## 🔧 カスタマイズ

### 独自のJudgeを追加
```python
# custom_judge.py
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams

my_custom_judge = GEval(
    name="Custom Judge",
    criteria="独自の評価基準をここに記述",
    evaluation_steps=[
        "ステップ1の説明",
        "ステップ2の説明",
        "ステップ3の説明"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT
    ],
    threshold=0.7,
    model="gpt-4o-mini"
)
```

### 新しいデータセットの追加
```bash
# 新しいCSVファイルを作成
cp qa_dataset.csv my_dataset.csv

# 06_batch_evaluation.pyでファイル名を変更
# df = pd.read_csv('my_dataset.csv')
```

## 🔗 関連リンク

- [📖 メインREADME](README.md)
- [📝 サンプルスクリプト詳細](example/README.md)
- [🐳 Docker公式ドキュメント](https://docs.docker.com/)
- [🤖 DeepEval公式ドキュメント](https://github.com/confident-ai/deepeval)
- [🔑 OpenAI API](https://platform.openai.com/docs)

## 🆘 サポート

問題が解決しない場合は、以下の情報と共にissueを作成してください：

1. **環境情報**: OS、Dockerバージョン
2. **実行コマンド**: 実行したコマンドの履歴
3. **エラーメッセージ**: 完全なエラーメッセージ
4. **設定ファイル**: .envファイルの内容（APIキー部分は除く）

---

<div align="center">
  <p>
    <strong>🎉 セットアップが完了したら、実際にサンプルを実行してみましょう！</strong>
  </p>
  <p>
    <a href="example/README.md">📝 サンプルスクリプトガイドへ →</a>
  </p>
</div>
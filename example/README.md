# 📝 DeepEval サンプルスクリプト詳細ガイド

<div align="center">
  <p>
    <a href="../README.md">← メインREADMEに戻る</a>
  </p>
</div>

このディレクトリには、DeepEval を用いた LLM as a Judge 評価の実践例が含まれています。  
基本的な評価から本格的な本番運用まで、段階的に学習できるように構成されています。

## 📋 スクリプト一覧

| スクリプト | 説明 | 主要機能 | 難易度 |
|-----------|------|---------|--------|
| [01_basic_judge.py](#01-基本的なjudge評価) | 基本的なJudge評価 | GEval基本実装 | 🟢 初級 |
| [02_multi_metrics.py](#02-複数観点での評価) | 複数観点での評価 | 多角的評価指標 | 🟡 初級 |
| [03_rag_judge.py](#03-rag向け評価) | RAG向け評価 | 検索拡張生成評価 | 🟡 中級 |
| [04_custom_judges.py](#04-カスタムjudge) | カスタムJudge | ドメイン特化評価 | 🟠 中級 |
| [05_judge_models.py](#05-judgeモデル指定) | Judgeモデル指定 | モデル比較・選択 | 🟠 中級 |
| [06_batch_evaluation.py](#06-バッチ評価) | バッチ評価 | 大規模評価処理 | 🔴 上級 |
| [07_analysis.py](#07-評価結果の分析) | 評価結果の分析 | 統計分析・可視化 | 🔴 上級 |
| [08_judge_correlation.py](#08-judge間の一致度分析) | Judge間の一致度分析 | 評価者間信頼性 | 🔴 上級 |
| [09_judge_monitoring.py](#09-継続的モニタリング) | 継続的モニタリング | 性能監視・校正 | 🔴 上級 |
| [10_pipeline.py](#10-パイプライン実装例) | パイプライン実装例 | 本番運用システム | 🔴 上級 |

## 🎯 各スクリプトの詳細

### 01. 基本的なJudge評価
**ファイル:** `01_basic_judge.py`

**概要:**  
DeepEvalの基本的な使い方を学ぶエントリーポイント。単一のJudgeを使用した評価の実装方法を示します。

**主要機能:**
- 🎯 **GEval** による基本的なJudge評価
- 📝 **評価基準**の定義と実装
- 🔍 **ステップバイステップ**の評価プロセス

**評価指標:**
- **正確性Judge**: 回答の事実的正確性を評価（閾値: 0.7）

**学習ポイント:**
```python
# 基本的なJudge定義
correctness_judge = GEval(
    name="Correctness",
    criteria="回答が質問に対して事実的に正確で完全かどうかを評価する",
    evaluation_steps=[...],
    threshold=0.7,
    model="gpt-4o-mini"
)
```

### 02. 複数観点での評価
**ファイル:** `02_multi_metrics.py`

**概要:**  
複数の評価観点を組み合わせた包括的な品質評価を実装します。

**主要機能:**
- 🎯 **4つの評価観点**による多角的評価
- 📊 **観点別スコア**の個別取得
- 🔍 **総合的品質判定**

**評価指標:**
- **正確性Judge**: 事実的正確性（閾値: 0.8）
- **完全性Judge**: 回答の網羅性（閾値: 0.7）
- **明確性Judge**: 理解しやすさ（閾値: 0.6）
- **関連性Judge**: 質問との関連性（閾値: 0.8）

**使用ケース:**
```python
# 複数Judge同時評価
judges = [accuracy_judge, completeness_judge, clarity_judge, relevance_judge]
for judge in judges:
    judge.measure(test_case)
    print(f"{judge.name}: {judge.score}")
```

### 03. RAG向け評価
**ファイル:** `03_rag_judge.py`

**概要:**  
RAG（Retrieval-Augmented Generation）システム専用の評価手法を実装します。

**主要機能:**
- 🔍 **コンテキスト活用度**の評価
- 📚 **情報源への忠実性**チェック
- 🚫 **ハルシネーション**検出

**評価指標:**
- **忠実性Judge**: 提供コンテキストへの忠実度（閾値: 0.8）
- **コンテキスト活用Judge**: 検索情報の効果的活用（閾値: 0.7）

**特徴:**
```python
# RAG特有のパラメータ使用
evaluation_params=[
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.RETRIEVAL_CONTEXT  # RAG固有
]
```

### 04. カスタムJudge
**ファイル:** `04_custom_judges.py`

**概要:**  
医療・法律など特定ドメインに特化したカスタムJudgeの実装例を示します。

**主要機能:**
- 🏥 **医療分野**専用の安全性評価
- ⚖️ **法律分野**専用のコンプライアンス評価
- 🔒 **高い信頼性**要求への対応

**評価指標:**
- **医療正確性Judge**: 医療情報の安全性（閾値: 0.9）
- **法的コンプライアンスJudge**: 法律情報の適切性（閾値: 0.85）

**安全性重視:**
```python
# 高閾値設定で安全性を重視
medical_judge = GEval(
    name="Medical Accuracy",
    criteria="医療情報の正確性と安全性を厳格に評価",
    threshold=0.9  # 高い基準
)
```

### 05. Judgeモデル指定
**ファイル:** `05_judge_models.py`

**概要:**  
異なるLLMモデルをJudgeとして使用し、その特性を比較します。

**主要機能:**
- 🤖 **複数モデル**の比較評価
- 🏢 **商用モデル** vs **ローカルモデル**
- 📈 **性能特性**の把握

**対応モデル:**
- **GPT-4 Judge**: 高精度商用モデル（O3）
- **Claude Judge**: 代替商用モデル
- **Local Model Judge**: ローカル実行モデル（Ollama/Llama2）

**モデル比較例:**
```python
# 異なるモデルでの評価
gpt4_judge = GEval(model="o3", threshold=0.8)
claude_judge = GEval(model="claude-3-haiku", threshold=0.75)
local_judge = GEval(model="ollama/llama2", threshold=0.7)
```

### 06. バッチ評価
**ファイル:** `06_batch_evaluation.py`

**概要:**  
大量のデータを効率的に処理するバッチ評価システムを実装します。

**主要機能:**
- 📊 **CSVデータ**の一括処理
- ⚡ **並列処理**による高速化
- 📈 **進捗表示**付き実行

**技術仕様:**
- **データソース**: qa_dataset.csv
- **並列度**: 最大3並列
- **処理方式**: EvaluationDataset + evaluate()

**バッチ処理例:**
```python
# 大規模データの効率的処理
dataset = EvaluationDataset(test_cases=test_cases)
evaluate(
    dataset,
    metrics=[accuracy_judge, completeness_judge, clarity_judge, relevance_judge],
    max_concurrent=3
)
```

### 07. 評価結果の分析
**ファイル:** `07_analysis.py`

**概要:**  
評価結果の統計分析と可視化により、品質の洞察を得ます。

**主要機能:**
- 📊 **統計的分析**（平均、分散、分布）
- 📈 **可視化**（ヒストグラム、散布図）
- 🔍 **問題ケース**の特定

**分析内容:**
- スコア分布の把握
- 低性能ケースの抽出
- 評価指標間の関係性分析

**可視化例:**
```python
# 結果の可視化
plt.figure(figsize=(12, 8))
plt.hist(scores, bins=20, alpha=0.7)
plt.title('Score Distribution')
plt.show()
```

### 08. Judge間の一致度分析
**ファイル:** `08_judge_correlation.py`

**概要:**  
複数のJudge間の一致度を分析し、評価の信頼性を検証します。

**主要機能:**
- 🔗 **相関分析**によるJudge間関係の把握
- 📊 **一致度マトリクス**の可視化
- ⚠️ **意見の分かれるケース**の特定

**分析手法:**
- ピアソン相関係数の計算
- ヒートマップによる可視化
- 不一致ケースの詳細分析

**信頼性評価:**
```python
# Judge間相関の分析
correlation_matrix = np.corrcoef([scores1, scores2, scores3, scores4])
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
```

### 09. 継続的モニタリング
**ファイル:** `09_judge_monitoring.py`

**概要:**  
本番運用でのJudge性能の継続的監視と校正を行います。

**主要機能:**
- 👥 **人間評価**との一致度測定
- 🎯 **閾値最適化**の自動調整
- 📊 **性能指標**の継続的追跡

**監視内容:**
- Judge-人間一致率の測定
- 精度・再現率の最適化
- 性能劣化の早期検出

**校正プロセス:**
```python
# 性能校正の実装
def calibrate_judge(judge, human_scores, target_precision=0.85):
    # 最適閾値の探索
    best_threshold = optimize_threshold(judge, human_scores, target_precision)
    judge.threshold = best_threshold
```

### 10. パイプライン実装例
**ファイル:** `10_pipeline.py`

**概要:**  
本番環境で使用できる包括的な評価パイプラインシステムを構築します。

**主要機能:**
- 🏗️ **QAEvaluationPipeline**クラスによる統合管理
- ⚙️ **設定可能**なJudge構成
- 🔄 **スケーラブル**なアーキテクチャ

**システム設計:**
- モジュラー設計による拡張性
- 設定ファイルベースの管理
- エラーハンドリング機能

**本番運用例:**
```python
# 本番用パイプライン
pipeline = QAEvaluationPipeline(
    judges=[accuracy_judge, helpfulness_judge],
    batch_size=10,
    max_concurrent=5
)
results = pipeline.evaluate_batch(test_cases)
```

## 📊 データファイル

### qa_dataset.csv
**概要**: バッチ評価用のサンプルデータセット

**構造:**
- `question`: 質問文
- `llm_answer`: LLMの回答
- `expected_answer`: 期待される回答
- `context`: RAG用コンテキスト（"|||"で区切り）

**使用例:**
```python
# CSV読み込みと処理
df = pd.read_csv('qa_dataset.csv')
test_cases = []
for _, row in df.iterrows():
    test_cases.append(LLMTestCase(
        input=row['question'],
        actual_output=row['llm_answer'],
        expected_output=row['expected_answer'],
        retrieval_context=row['context'].split('|||')
    ))
```

## 🚀 実行順序の推奨

### 🟢 初心者向け学習パス
1. `01_basic_judge.py` - 基本概念の理解
2. `02_multi_metrics.py` - 多角的評価の学習
3. `03_rag_judge.py` - RAG特化評価の実践

### 🟡 中級者向け学習パス
4. `04_custom_judges.py` - ドメイン特化の実装
5. `05_judge_models.py` - モデル選択の最適化

### 🔴 上級者向け学習パス
6. `06_batch_evaluation.py` - 大規模処理の実装
7. `07_analysis.py` - 結果分析の実践
8. `08_judge_correlation.py` - 信頼性評価の実施
9. `09_judge_monitoring.py` - 継続的改善の実装
10. `10_pipeline.py` - 本番システムの構築

## 💡 よくある使用パターン

### 🎯 基本的な品質評価
```bash
python 01_basic_judge.py
python 02_multi_metrics.py
```

### 🔍 RAGシステムの評価
```bash
python 03_rag_judge.py
python 06_batch_evaluation.py
python 07_analysis.py
```

### 🏥 ドメイン特化評価
```bash
python 04_custom_judges.py
python 09_judge_monitoring.py
```

### 🏢 本番運用システム
```bash
python 10_pipeline.py
# 継続的監視
python 09_judge_monitoring.py
```

## 🔗 関連リンク

- [📖 メインREADME](../README.md)
- [🚀 セットアップガイド](../SETUP.md)
- [📚 DeepEval公式ドキュメント](https://github.com/confident-ai/deepeval)

---

<div align="center">
  <p>
    <strong>🎓 実践的な学習を通じて、LLM as a Judge の専門家になりましょう！</strong>
  </p>
  <p>
    <a href="../README.md">← メインREADMEに戻る</a>
  </p>
</div>
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from loguru import logger

# 医療分野向けJudge
medical_accuracy_judge = GEval(
    name="MedicalAccuracy",
    criteria="""
    医療情報として適切で安全な回答かを評価する。以下の観点を考慮：
    1. 医学的事実の正確性
    2. 患者安全への配慮
    3. 適切な免責事項の有無
    4. 専門用語の正確な使用
    """,
    evaluation_steps=[
        "医学的内容の事実確認を行う",
        "患者に害を与える可能性がないか検証する",
        "適切な免責事項や医師への相談推奨があるか確認する",
        "専門用語が正確に使用されているか評価する",
        "総合的な医療情報としての適切性を判定する"
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.9
)

# 法務分野向けJudge
legal_compliance_judge = GEval(
    name="LegalCompliance",
    criteria="""
    法的情報として適切で、誤解を招かない回答かを評価する：
    1. 法的事実の正確性
    2. 法的助言と情報提供の適切な区別
    3. 管轄や時期への言及
    4. 専門家への相談推奨
    """,
    evaluation_steps=[
        "法的内容の正確性を確認する",
        "法的助言ではなく一般的情報として適切に表現されているか検証する",
        "適用される法域や時期が明確化されているか確認する",
        "専門家への相談を適切に推奨しているか評価する"
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.85
)

# サンプルテストケース
medical_case = LLMTestCase(
    input="風邪のときに市販薬を飲んでも大丈夫ですか？",
    actual_output="多くの場合、市販薬の使用は可能ですが、症状が重い場合や持病がある場合は医師に相談してください。",
    expected_output="市販薬は一般的に軽い風邪症状に使用できますが、症状が長引く場合や持病がある場合は必ず医師に相談してください。"
)

legal_case = LLMTestCase(
    input="日本で18歳は成人ですか？",
    actual_output="2022年4月から日本の成人年齢は18歳になりました。ただし、飲酒や喫煙は20歳からです。",
    expected_output=""
)

# 評価実行
medical_accuracy_judge.measure(medical_case)
logger.info(f"MedicalAccuracy スコア: {medical_accuracy_judge.score}")
logger.info(f"理由: {medical_accuracy_judge.reason}")
logger.info("-" * 40)

legal_compliance_judge.measure(legal_case)
logger.info(f"LegalCompliance スコア: {legal_compliance_judge.score}")
logger.info(f"理由: {legal_compliance_judge.reason}")
logger.info("-" * 40)
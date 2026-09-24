import os
import pandas as pd

RESULT_DIR = ""

PRED_PATH = os.path.join(
    RESULT_DIR,
    "jev_v2_test_predictions.csv"
)

df = pd.read_csv(PRED_PATH)

THRESHOLD = 0.5

df["target"] = (
    df["jev_probability"] >= THRESHOLD
).astype(int)

submission = df[["id", "target"]].copy()

SAVE_PATH = os.path.join(
    RESULT_DIR,
    "jev_threshold_050_submission.csv"
)

submission.to_csv(
    SAVE_PATH,
    index=False
)

print("Threshold:", THRESHOLD)
print()
print("예측값 분포")
print(submission["target"].value_counts())
print()
print("1 비율:", submission["target"].mean())
print()
print("저장 완료:")
print(SAVE_PATH)
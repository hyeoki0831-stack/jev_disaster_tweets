import os
import time
import requests
import numpy as np
import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

API_KEY = ""

BASE_DIR = r""

TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")
SAMPLE_PATH = os.path.join(BASE_DIR, "sample_submission.csv")

RESULT_DIR = os.path.join(BASE_DIR, "jev_result")
os.makedirs(RESULT_DIR, exist_ok=True)

URL = "https://ai-gateway.vercel.sh/v1/evaluate"

VALIDATION_N = 1000
RANDOM_STATE = 42

MAX_WORKERS = 2
MAX_RETRIES = 6
CHECKPOINT_EVERY = 50

RUN_VALIDATION = True
RUN_TEST = True
AUTO_TUNE_THRESHOLD = True

DEFAULT_THRESHOLD = 0.50

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
sample_submission = pd.read_csv(SAMPLE_PATH)

print("train shape:", train.shape)
print("test shape :", test.shape)
print()
print("train columns:", train.columns.tolist())
print("test columns :", test.columns.tolist())

def clean_value(x):
    if pd.isna(x):
        return ""
    return str(x)

def make_state(row):
    return {
        "tweet": clean_value(row["text"]),
        "keyword": clean_value(row.get("keyword", "")),
        "location": clean_value(row.get("location", ""))
    }

def call_jev(row):
    state = make_state(row)

    payload = {
        "model": "typesafe-ai/jev",
        "state": state,
        "questions": {
            "real_disaster": {
                "type": "boolean",
                "instructions": (
                    "Determine whether this tweet is genuinely about a real-world "
                    "disaster or emergency event. True means the tweet reports, "
                    "describes, warns about, reacts to, or clearly refers to an "
                    "actual real-world disaster or emergency such as a fire, flood, "
                    "earthquake, storm, explosion, serious accident, attack, casualty, "
                    "evacuation, destruction, or comparable emergency. False means "
                    "disaster-related language is metaphorical, figurative, humorous, "
                    "fictional, lyrical, promotional, hypothetical, or unrelated to "
                    "an actual real-world disaster. Focus on meaning and context, "
                    "not merely the presence of disaster-related keywords."
                )
            }
        }
    }

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                URL,
                headers=HEADERS,
                json=payload,
                timeout=45
            )

            if response.status_code == 200:
                data = response.json()

                probability = float(
                    data["answers"]["real_disaster"]["probability"]
                )

                probability = min(max(probability, 0.0), 1.0)

                return probability

            if response.status_code == 401:
                raise RuntimeError(
                    "401 인증 오류입니다. API Key를 확인하세요."
                )

            if response.status_code == 403:
                raise RuntimeError(
                    "403 접근 거부입니다.\n"
                    + response.text
                )

            if response.status_code in [429, 500, 502, 503, 504, 529]:
                wait = min(2 ** attempt, 30)

                print(
                    f"일시적 API 오류 {response.status_code} "
                    f"- {wait}초 후 재시도"
                )

                time.sleep(wait)
                continue

            last_error = (
                f"HTTP {response.status_code}: {response.text}"
            )

        except requests.RequestException as e:
            last_error = str(e)

        wait = min(2 ** attempt, 30)
        time.sleep(wait)

    raise RuntimeError(
        f"Jev 요청 최종 실패: {last_error}"
    )

def predict_dataframe(df, checkpoint_path):
    result = df.copy()
    result["jev_probability"] = np.nan

    if os.path.exists(checkpoint_path):
        old = pd.read_csv(checkpoint_path)

        if (
            len(old) == len(result)
            and "jev_probability" in old.columns
        ):
            result["jev_probability"] = old["jev_probability"]
            completed = result["jev_probability"].notna().sum()

            print(
                f"체크포인트 발견: "
                f"{completed}/{len(result)}개 완료된 상태에서 재개"
            )

    remaining = result.index[
        result["jev_probability"].isna()
    ].tolist()

    total = len(result)

    for start in range(0, len(remaining), CHECKPOINT_EVERY):
        indices = remaining[
            start:start + CHECKPOINT_EVERY
        ]

        rows = [
            result.loc[idx]
            for idx in indices
        ]

        with ThreadPoolExecutor(
            max_workers=MAX_WORKERS
        ) as executor:
            probabilities = list(
                executor.map(call_jev, rows)
            )

        for idx, probability in zip(
            indices,
            probabilities
        ):
            result.loc[
                idx,
                "jev_probability"
            ] = probability

        result.to_csv(
            checkpoint_path,
            index=False,
            encoding="utf-8-sig"
        )

        completed = (
            result["jev_probability"]
            .notna()
            .sum()
        )

        print(
            f"{completed}/{total} 완료 "
            f"({completed / total * 100:.1f}%)"
        )

    return result

print()
print("API 연결 테스트 중...")

test_probability = call_jev(train.iloc[0])

print(
    "API 정상 연결"
)
print(
    "첫 번째 train 데이터 Jev probability:",
    test_probability
)

threshold = DEFAULT_THRESHOLD

if RUN_VALIDATION:
    validation = (
        train.groupby(
            "target",
            group_keys=False
        )
        .sample(
            n=VALIDATION_N // 2,
            random_state=RANDOM_STATE
        )
        .sample(
            frac=1,
            random_state=RANDOM_STATE
        )
        .reset_index(drop=True)
    )

    calibration, evaluation = train_test_split(
        validation,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=validation["target"]
    )

    calibration = calibration.reset_index(drop=True)
    evaluation = evaluation.reset_index(drop=True)

    calibration_path = os.path.join(
        RESULT_DIR,
        "jev_calibration_predictions.csv"
    )

    evaluation_path = os.path.join(
        RESULT_DIR,
        "jev_evaluation_predictions.csv"
    )

    print()
    print(
        f"Calibration 데이터 {len(calibration)}개 예측 시작"
    )

    calibration_pred = predict_dataframe(
        calibration,
        calibration_path
    )

    if AUTO_TUNE_THRESHOLD:
        best_threshold = DEFAULT_THRESHOLD
        best_f1 = -1

        for t in np.arange(
            0.10,
            0.91,
            0.01
        ):
            pred = (
                calibration_pred[
                    "jev_probability"
                ].values >= t
            ).astype(int)

            score = f1_score(
                calibration_pred["target"],
                pred
            )

            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        threshold = best_threshold

        print()
        print(
            "최적 threshold:",
            round(threshold, 2)
        )
        print(
            "Calibration F1:",
            round(best_f1, 6)
        )

    print()
    print(
        f"Evaluation 데이터 {len(evaluation)}개 예측 시작"
    )

    evaluation_pred = predict_dataframe(
        evaluation,
        evaluation_path
    )

    evaluation_pred[
        "jev_prediction"
    ] = (
        evaluation_pred[
            "jev_probability"
        ] >= threshold
    ).astype(int)

    val_f1 = f1_score(
        evaluation_pred["target"],
        evaluation_pred["jev_prediction"]
    )

    val_accuracy = accuracy_score(
        evaluation_pred["target"],
        evaluation_pred["jev_prediction"]
    )

    cm = confusion_matrix(
        evaluation_pred["target"],
        evaluation_pred["jev_prediction"]
    )

    print()
    print("Jev Zero-shot Validation")
    print(
        "Threshold:",
        round(threshold, 2)
    )
    print(
        "F1 Score:",
        round(val_f1, 6)
    )
    print(
        "Accuracy:",
        round(val_accuracy, 6)
    )
    print()
    print("Confusion Matrix")
    print(cm)

    evaluation_pred.to_csv(
        evaluation_path,
        index=False,
        encoding="utf-8-sig"
    )

if RUN_TEST:
    print()
    print(
        f"전체 test {len(test)}개 Jev 예측 시작"
    )

    test_checkpoint = os.path.join(
        RESULT_DIR,
        "jev_test_predictions.csv"
    )

    test_pred = predict_dataframe(
        test,
        test_checkpoint
    )

    test_pred[
        "target"
    ] = (
        test_pred[
            "jev_probability"
        ] >= threshold
    ).astype(int)

    submission = pd.DataFrame({
        "id": test_pred["id"],
        "target": test_pred["target"]
    })

    submission_path = os.path.join(
        RESULT_DIR,
        "jev_zero_shot_submission.csv"
    )

    submission.to_csv(
        submission_path,
        index=False
    )

    test_pred.to_csv(
        os.path.join(
            RESULT_DIR,
            "jev_test_predictions_final.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("완료")
    print(
        "사용 Threshold:",
        round(threshold, 2)
    )
    print()
    print(
        submission["target"].value_counts()
    )
    print()
    print(
        "Kaggle 제출 파일:"
    )
    print(
        submission_path
    )
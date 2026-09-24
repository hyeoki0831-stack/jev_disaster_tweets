import os
import time
import random
import requests
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, classification_report

API_KEY = ""

BASE_DIR = ""

TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
TEST_PATH = os.path.join(BASE_DIR, "test.csv")

RESULT_DIR = os.path.join(BASE_DIR, "jev_result_v2")
os.makedirs(RESULT_DIR, exist_ok=True)

URL = "https://ai-gateway.vercel.sh/v1/evaluate"

BATCH_SIZE = 8
REQUEST_GAP = 1.0
MAX_RETRIES = 8
MAX_WAIT = 90
TIMEOUT = 90

RUN_VALIDATION = True
VALIDATION_N = 600
RANDOM_STATE = 42

RUN_TEST = True

DEFAULT_THRESHOLD = 0.50
AUTO_TUNE_THRESHOLD = True

session = requests.Session()

session.headers.update({
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
})

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

print("train shape:", train.shape)
print("test shape :", test.shape)

def clean_value(x):
    if pd.isna(x):
        return ""
    return str(x)

def build_payload(batch):
    tweets = []
    questions = {}

    for local_idx, (_, row) in enumerate(batch.iterrows()):
        tweet_key = f"tweet_{local_idx}"
        question_key = f"q_{local_idx}"

        tweets.append({
            "key": tweet_key,
            "text": clean_value(row["text"]),
            "keyword": clean_value(row.get("keyword", "")),
            "location": clean_value(row.get("location", ""))
        })

        questions[question_key] = {
            "type": "boolean",
            "instructions": (
                f"Evaluate only the tweet whose key is '{tweet_key}'. "
                "Determine whether it genuinely refers to a real-world disaster "
                "or emergency event. "
                "Return true when the tweet reports, describes, warns about, "
                "reacts to, or clearly refers to an actual fire, flood, earthquake, "
                "storm, explosion, serious accident, attack, casualty, evacuation, "
                "destruction, or comparable real-world emergency. "
                "Return false when disaster-related language is metaphorical, "
                "figurative, humorous, fictional, lyrical, promotional, hypothetical, "
                "or otherwise not about a genuine real-world disaster. "
                "Use the text as the primary evidence. Keyword and location are "
                "supporting context only. Do not classify something as a disaster "
                "merely because a disaster-related keyword appears."
            )
        }

    payload = {
        "model": "typesafe-ai/jev",
        "state": {
            "tweets": tweets
        },
        "questions": questions
    }

    return payload

def get_retry_after(response):
    value = response.headers.get("Retry-After")

    if value is None:
        return None

    try:
        return float(value)
    except:
        return None

def extract_probabilities(data, n_items):
    if "answers" not in data:
        raise RuntimeError(
            "응답에 answers가 없습니다:\n"
            + str(data)[:1000]
        )

    answers = data["answers"]
    probabilities = []

    for i in range(n_items):
        key = f"q_{i}"

        if key not in answers:
            raise RuntimeError(
                f"{key}가 Jev 응답에 없습니다:\n"
                + str(data)[:1000]
            )

        answer = answers[key]

        if isinstance(answer, dict):
            if "probability" in answer:
                probability = float(answer["probability"])
            elif "trueProbability" in answer:
                probability = float(answer["trueProbability"])
            elif "value" in answer and isinstance(answer["value"], bool):
                probability = 1.0 if answer["value"] else 0.0
            else:
                raise RuntimeError(
                    f"알 수 없는 Jev 응답 형식: {answer}"
                )

        elif isinstance(answer, bool):
            probability = 1.0 if answer else 0.0

        else:
            raise RuntimeError(
                f"알 수 없는 Jev 응답 형식: {answer}"
            )

        probability = np.clip(probability, 0.0, 1.0)
        probabilities.append(float(probability))

    return probabilities

class TemporaryAPIError(Exception):
    pass

def call_batch_once(batch):
    payload = build_payload(batch)

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = session.post(
                URL,
                json=payload,
                timeout=TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()

                probabilities = extract_probabilities(
                    data,
                    len(batch)
                )

                time.sleep(REQUEST_GAP)

                return probabilities

            if response.status_code == 401:
                raise RuntimeError(
                    "401 인증 실패입니다. API Key를 확인하세요.\n"
                    + response.text
                )

            if response.status_code == 403:
                raise RuntimeError(
                    "403 접근 거부입니다.\n"
                    + response.text
                )

            if response.status_code == 429:
                retry_after = get_retry_after(response)

                if retry_after is not None:
                    wait = min(
                        retry_after + random.uniform(0.5, 1.5),
                        MAX_WAIT
                    )
                else:
                    wait = min(
                        3 * (2 ** attempt)
                        + random.uniform(0.5, 2.0),
                        MAX_WAIT
                    )

                print(
                    f"429 Rate Limit | "
                    f"batch={len(batch)} | "
                    f"{wait:.1f}초 대기 | "
                    f"재시도 {attempt + 1}/{MAX_RETRIES}"
                )

                time.sleep(wait)
                last_error = response.text
                continue

            if response.status_code in [500, 502, 503, 504, 529]:
                wait = min(
                    5 * (2 ** attempt)
                    + random.uniform(0.5, 2.0),
                    MAX_WAIT
                )

                print(
                    f"{response.status_code} 서버 오류 | "
                    f"batch={len(batch)} | "
                    f"{wait:.1f}초 대기 | "
                    f"재시도 {attempt + 1}/{MAX_RETRIES}"
                )

                time.sleep(wait)
                last_error = response.text
                continue

            raise RuntimeError(
                f"예상하지 못한 API 오류\n"
                f"status={response.status_code}\n"
                f"{response.text}"
            )

        except requests.Timeout:
            wait = min(
                5 * (2 ** attempt),
                MAX_WAIT
            )

            print(
                f"Timeout | "
                f"batch={len(batch)} | "
                f"{wait:.1f}초 대기"
            )

            time.sleep(wait)

            last_error = "Timeout"

        except requests.ConnectionError as e:
            wait = min(
                5 * (2 ** attempt),
                MAX_WAIT
            )

            print(
                f"Connection Error | "
                f"{wait:.1f}초 대기"
            )

            time.sleep(wait)

            last_error = str(e)

    raise TemporaryAPIError(
        f"{MAX_RETRIES}회 재시도 후에도 실패: {last_error}"
    )

def call_batch_adaptive(batch):
    try:
        return call_batch_once(batch)

    except TemporaryAPIError:
        if len(batch) == 1:
            print()
            print("단일 요청까지 실패했습니다.")
            print("현재 Jev 서버가 불안정할 가능성이 큽니다.")
            print("이미 처리된 데이터는 저장되어 있습니다.")
            print("잠시 후 코드를 다시 실행하면 이어서 진행됩니다.")
            raise

        middle = len(batch) // 2

        left = batch.iloc[:middle].copy()
        right = batch.iloc[middle:].copy()

        print()
        print(
            f"Batch {len(batch)}개가 계속 실패 → "
            f"{len(left)} + {len(right)}개로 자동 분할"
        )

        left_result = call_batch_adaptive(left)

        time.sleep(2)

        right_result = call_batch_adaptive(right)

        return left_result + right_result

def restore_checkpoint(result, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return result

    old = pd.read_csv(checkpoint_path)

    if "id" not in old.columns:
        return result

    if "jev_probability" not in old.columns:
        return result

    old_map = (
        old.dropna(subset=["jev_probability"])
        .drop_duplicates("id")
        .set_index("id")["jev_probability"]
        .to_dict()
    )

    result["jev_probability"] = result.apply(
        lambda row: (
            old_map[row["id"]]
            if row["id"] in old_map
            else row["jev_probability"]
        ),
        axis=1
    )

    restored = result["jev_probability"].notna().sum()

    if restored > 0:
        print(
            f"체크포인트 복구: "
            f"{restored}/{len(result)}개 완료 상태"
        )

    return result

def predict_dataframe(df, checkpoint_path):
    result = df.copy()
    result["jev_probability"] = np.nan

    result = restore_checkpoint(
        result,
        checkpoint_path
    )

    while True:
        remaining_indices = result.index[
            result["jev_probability"].isna()
        ].tolist()

        if len(remaining_indices) == 0:
            break

        current_indices = remaining_indices[:BATCH_SIZE]

        batch = result.loc[current_indices].copy()

        try:
            probabilities = call_batch_adaptive(batch)

        except TemporaryAPIError:
            result.to_csv(
                checkpoint_path,
                index=False,
                encoding="utf-8-sig"
            )

            raise

        for idx, probability in zip(
            current_indices,
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

        completed = result[
            "jev_probability"
        ].notna().sum()

        print(
            f"{completed}/{len(result)} 완료 "
            f"({completed / len(result) * 100:.1f}%)"
        )

    return result

threshold = DEFAULT_THRESHOLD

if RUN_VALIDATION:
    n_validation = min(
        VALIDATION_N,
        len(train)
    )

    validation = train.sample(
        n=n_validation,
        random_state=RANDOM_STATE
    )

    calibration, evaluation = train_test_split(
        validation,
        test_size=0.5,
        random_state=RANDOM_STATE,
        stratify=validation["target"]
    )

    calibration = calibration.reset_index(drop=True)
    evaluation = evaluation.reset_index(drop=True)

    calibration_checkpoint = os.path.join(
        RESULT_DIR,
        "jev_v2_calibration.csv"
    )

    evaluation_checkpoint = os.path.join(
        RESULT_DIR,
        "jev_v2_evaluation.csv"
    )

    print()
    print(
        f"Calibration {len(calibration)}개 시작"
    )

    calibration_pred = predict_dataframe(
        calibration,
        calibration_checkpoint
    )

    if AUTO_TUNE_THRESHOLD:
        thresholds = np.arange(
            0.10,
            0.91,
            0.01
        )

        scores = []

        for t in thresholds:
            prediction = (
                calibration_pred[
                    "jev_probability"
                ].values >= t
            ).astype(int)

            score = f1_score(
                calibration_pred["target"],
                prediction
            )

            scores.append(score)

        best_idx = int(
            np.argmax(scores)
        )

        threshold = float(
            thresholds[best_idx]
        )

        print()
        print(
            "Calibration 최적 threshold:",
            round(threshold, 2)
        )

        print(
            "Calibration F1:",
            round(scores[best_idx], 6)
        )

    print()
    print(
        f"Evaluation {len(evaluation)}개 시작"
    )

    evaluation_pred = predict_dataframe(
        evaluation,
        evaluation_checkpoint
    )

    evaluation_pred["jev_prediction"] = (
        evaluation_pred[
            "jev_probability"
        ] >= threshold
    ).astype(int)

    f1 = f1_score(
        evaluation_pred["target"],
        evaluation_pred["jev_prediction"]
    )

    accuracy = accuracy_score(
        evaluation_pred["target"],
        evaluation_pred["jev_prediction"]
    )

    cm = confusion_matrix(
        evaluation_pred["target"],
        evaluation_pred["jev_prediction"]
    )

    print()
    print("Jev Zero-shot Validation 결과")
    print("Threshold :", round(threshold, 2))
    print("F1 Score  :", round(f1, 6))
    print("Accuracy  :", round(accuracy, 6))

    print()
    print("Confusion Matrix")
    print(cm)

    print()
    print(
        classification_report(
            evaluation_pred["target"],
            evaluation_pred["jev_prediction"],
            digits=4
        )
    )

    evaluation_pred.to_csv(
        os.path.join(
            RESULT_DIR,
            "jev_v2_evaluation_final.csv"
        ),
        index=False,
        encoding="utf-8-sig"
    )

if RUN_TEST:
    print()
    print(
        f"Kaggle test {len(test)}개 예측 시작"
    )

    test_checkpoint = os.path.join(
        RESULT_DIR,
        "jev_v2_test_checkpoint.csv"
    )

    test_pred = predict_dataframe(
        test,
        test_checkpoint
    )

    test_pred["target"] = (
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
        "jev_v2_submission.csv"
    )

    prediction_path = os.path.join(
        RESULT_DIR,
        "jev_v2_test_predictions.csv"
    )

    submission.to_csv(
        submission_path,
        index=False
    )

    test_pred.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig"
    )

    print()
    print("전체 작업 완료")
    print("사용 Threshold:", round(threshold, 2))

    print()
    print("예측값 분포")
    print(
        submission["target"].value_counts()
    )

    print()
    print("Kaggle 제출 파일")
    print(submission_path)

    print()
    print("Jev 확률 포함 파일")
    print(prediction_path)
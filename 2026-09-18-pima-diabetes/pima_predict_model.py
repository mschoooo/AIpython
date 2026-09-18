"""Pima 당뇨 예측 입문 예제. 실행: python pima_predict_model.py"""

from pathlib import Path

import joblib
import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
OUTPUT = Path(__file__).resolve().parent / "pima_analysis"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    data_dir = Path(kagglehub.dataset_download("kumargh/pimaindiansdiabetescsv"))
    df = pd.read_csv(data_dir / "pima-indians-diabetes.csv", header=None, names=COLUMNS)

    X = df.drop(columns="Outcome")
    y = df["Outcome"]

    # stratify는 양성/음성 비율을 두 집합에서 비슷하게 유지합니다.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 다섯 측정 컬럼에서는 0을 결측으로 처리하고, 나머지 컬럼의 0은 유지합니다.
    # 중앙값과 표준화 기준은 학습 데이터만으로 계산됩니다.
    other_columns = [column for column in X.columns if column not in ZERO_AS_MISSING]
    preprocessing = ColumnTransformer([
        ("zero_as_missing", SimpleImputer(missing_values=0, strategy="median"), ZERO_AS_MISSING),
        ("other", SimpleImputer(strategy="median"), other_columns),
    ])
    model = make_pipeline(
        preprocessing,
        StandardScaler(),
        LogisticRegression(max_iter=1000, random_state=42),
    )
    model.fit(X_train, y_train)

    predicted = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    scores = {
        "accuracy": accuracy_score(y_test, predicted),
        "precision": precision_score(y_test, predicted),
        "recall": recall_score(y_test, predicted),
        "roc_auc": roc_auc_score(y_test, probability),
    }
    cm = confusion_matrix(y_test, predicted)
    print(f"학습: {len(X_train)}행 / 평가: {len(X_test)}행")
    print(f"항상 0으로 예측할 때 정확도: {(y_test == 0).mean():.3f}")
    for name, value in scores.items():
        print(f"{name}: {value:.3f}")
    print("혼동행렬 [[TN, FP], [FN, TP]]:")
    print(cm)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    ConfusionMatrixDisplay(cm, display_labels=["No diabetes", "Diabetes"]).plot(
        ax=axes[0], cmap="Blues", colorbar=False
    )
    axes[0].set_title("Confusion matrix (test set)")
    RocCurveDisplay.from_predictions(y_test, probability, ax=axes[1])
    axes[1].plot([0, 1], [0, 1], "--", color="gray")
    axes[1].set_title("ROC curve (test set)")
    fig.savefig(OUTPUT / "model_evaluation.png", dpi=180)
    plt.close(fig)

    pd.DataFrame([scores]).to_csv(OUTPUT / "model_metrics.csv", index=False)
    joblib.dump(model, OUTPUT / "pima_logistic_model.joblib")
    print(f"결과 저장: {OUTPUT}")
    print("학습 예제이며 의료 진단에 사용해서는 안 됩니다.")


if __name__ == "__main__":
    main()

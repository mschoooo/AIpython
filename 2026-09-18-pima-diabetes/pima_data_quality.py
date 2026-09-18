"""Pima Indians Diabetes 데이터 품질 점검: python pima_data_quality.py"""

from pathlib import Path

import kagglehub
import matplotlib

matplotlib.use("Agg")  # 창을 열지 않고 PNG로 저장
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATASET = "kumargh/pimaindiansdiabetescsv"
FILE_NAME = "pima-indians-diabetes.csv"
COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
# 아래 측정치의 0은 생리적으로 불가능하거나 결측 기록으로 취급하는 관례가 있습니다.
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
OUTPUT = Path(__file__).resolve().parent / "pima_analysis"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    data_dir = Path(kagglehub.dataset_download(DATASET))
    df = pd.read_csv(data_dir / FILE_NAME, header=None, names=COLUMNS)

    print(f"데이터 크기: {len(df)}행 × {len(df.columns)}열")
    print("\n첫 5행:\n", df.head().to_string(index=False))

    # 1. 원본의 NaN과, 결측으로 의심되는 0을 따로 집계합니다.
    actual_na = df.isna().sum()
    zero_missing = pd.Series(0, index=COLUMNS, dtype=int)
    zero_missing.loc[ZERO_AS_MISSING] = df[ZERO_AS_MISSING].eq(0).sum()
    missing = pd.DataFrame({"NaN": actual_na, "의심 결측(0)": zero_missing})
    missing["합계"] = missing.sum(axis=1)
    print("\n결측치 현황:\n", missing.to_string())

    # 2. 모든 컬럼 값이 동일한 행을 중복으로 정의합니다.
    duplicate_rows = int(df.duplicated().sum())
    print(f"\n완전히 동일한 중복 행: {duplicate_rows}개")

    # 3. 의심 결측을 제외하고 IQR 방식으로 이상치 후보를 셉니다.
    cleaned = df.copy()
    cleaned[ZERO_AS_MISSING] = cleaned[ZERO_AS_MISSING].replace(0, float("nan"))
    numeric_cols = [column for column in COLUMNS if column != "Outcome"]
    outliers = {}
    limits = []
    for column in numeric_cols:
        values = cleaned[column].dropna()
        q1, q3 = values.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers[column] = int(((values < lower) | (values > upper)).sum())
        limits.append({"컬럼": column, "하한": lower, "상한": upper,
                       "이상치 후보": outliers[column]})
    outlier_table = pd.DataFrame(limits)
    print("\nIQR 이상치 후보 (결측 제외):\n", outlier_table.to_string(index=False))

    # 결과표도 저장해 나중에 다시 확인할 수 있게 합니다.
    missing.to_csv(OUTPUT / "missing_summary.csv", encoding="utf-8-sig")
    outlier_table.to_csv(OUTPUT / "outlier_summary.csv", index=False, encoding="utf-8-sig")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), layout="constrained")
    missing[["NaN", "의심 결측(0)"]].rename(
        columns={"의심 결측(0)": "Zero as missing"}
    ).plot.bar(stacked=True, ax=axes[0], color=["#4C78A8", "#F58518"])
    axes[0].set_title("Missing values by column")
    axes[0].set_ylabel("Rows")
    axes[0].tick_params(axis="x", rotation=60)

    axes[1].bar(["Unique", "Duplicate"], [len(df) - duplicate_rows, duplicate_rows],
                color=["#54A24B", "#E45756"])
    axes[1].set_title("Exact duplicate rows")
    axes[1].set_ylabel("Rows")
    axes[1].bar_label(axes[1].containers[0])

    axes[2].bar(list(outliers), list(outliers.values()), color="#B279A2")
    axes[2].set_title("IQR outlier candidates")
    axes[2].set_ylabel("Rows")
    axes[2].tick_params(axis="x", rotation=60)
    axes[2].bar_label(axes[2].containers[0])
    fig.savefig(OUTPUT / "data_quality_overview.png", dpi=180)
    plt.close(fig)

    # 스케일이 다른 변수는 개별 축의 상자그림으로 봅니다.
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), layout="constrained")
    for ax, column in zip(axes.flat, numeric_cols):
        sns.boxplot(y=cleaned[column], ax=ax, color="#72B7B2", fliersize=3)
        ax.set_title(column)
        ax.set_ylabel("")
    fig.suptitle("Distributions and IQR outlier candidates", fontsize=15)
    fig.savefig(OUTPUT / "outlier_boxplots.png", dpi=180)
    plt.close(fig)
    print(f"\n그래프와 요약표 저장 위치: {OUTPUT}")


if __name__ == "__main__":
    main()

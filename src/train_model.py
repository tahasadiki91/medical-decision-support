import os
import joblib
import numpy as np
import pandas as pd

from scipy.io import arff

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "bone-marrow.arff")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "model_columns.pkl")


# =========================
# FEATURE CONFIGURATION
# =========================
# We keep only pre-transplant / transplant-time predictors.
FEATURES = [
    "Recipientgender",
    "Stemcellsource",
    "Donorage",
    "Gendermatch",
    "DonorABO",
    "RecipientABO",
    "RecipientRh",
    "CMVstatus",
    "DonorCMV",
    "RecipientCMV",
    "Disease",
    "Riskgroup",
    "Diseasegroup",
    "HLAmatch",
    "HLAgrI",
    "Recipientage",
    "CD34kgx10d6",
    "Rbodymass",
]

TARGET = "survival_status"

NUMERIC_FEATURES = [
    "Donorage",
    "Recipientage",
    "CD34kgx10d6",
    "Rbodymass",
]

CATEGORICAL_FEATURES = [
    "Recipientgender",
    "Stemcellsource",
    "Gendermatch",
    "DonorABO",
    "RecipientABO",
    "RecipientRh",
    "CMVstatus",
    "DonorCMV",
    "RecipientCMV",
    "Disease",
    "Riskgroup",
    "Diseasegroup",
    "HLAmatch",
    "HLAgrI",
]


# =========================
# DATA LOADING
# =========================
def load_arff_dataset(path: str) -> pd.DataFrame:
    """Load ARFF dataset and decode byte strings."""
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)

    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

    df.replace(["?", "None", "none", "nan", "NaN", ""], np.nan, inplace=True)
    return df


def cast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns into appropriate types."""
    df = df.copy()

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].astype("string")

    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    return df


# =========================
# PIPELINE
# =========================
def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


# =========================
# TRAINING
# =========================
def main() -> None:
    print("Loading dataset...")
    df = load_arff_dataset(DATA_PATH)
    df = cast_columns(df)

    required_columns = FEATURES + [TARGET]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    df = df[required_columns].copy()
    df = df.dropna(subset=[TARGET])

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    print("Dataset shape:", df.shape)
    print("Target distribution:")
    print(y.value_counts(dropna=False))
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("Training model...")
    model = build_pipeline()
    model.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"ROC-AUC:  {roc:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\nSaving artifacts...")
    joblib.dump(model, MODEL_PATH)
    joblib.dump(FEATURES, COLUMNS_PATH)

    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved feature list to: {COLUMNS_PATH}")


if __name__ == "__main__":
    main()
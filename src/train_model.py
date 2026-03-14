import os
from src.feature_engineering import ClinicalFeatureEngineer
import warnings
import joblib
import numpy as np
import pandas as pd

from scipy.io import arff

from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "bone-marrow.arff")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# On garde ce nom pour rester compatible avec l'app
MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
COLUMNS_PATH = os.path.join(MODELS_DIR, "model_columns.pkl")
INFO_PATH = os.path.join(MODELS_DIR, "model_info.pkl")


# =========================
# FEATURES DE BASE
# =========================
# Uniquement les variables pré-greffe / au moment de la greffe
BASE_FEATURES = [
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


# =========================
# FEATURE ENGINEERING CLINIQUE
# =========================


# =========================
# CHARGEMENT DONNÉES
# =========================
def load_arff_dataset(path: str) -> pd.DataFrame:
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)

    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: x.decode("utf-8") if isinstance(x, bytes) else x
        )

    df.replace(["?", "None", "none", "nan", "NaN", ""], np.nan, inplace=True)
    return df


def cast_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    numeric_base = ["Donorage", "Recipientage", "CD34kgx10d6", "Rbodymass"]
    categorical_base = [col for col in BASE_FEATURES if col not in numeric_base]

    for col in numeric_base:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in categorical_base:
        if col in df.columns:
            df[col] = df[col].astype("object")

    if TARGET in df.columns:
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    return df


# =========================
# PREPROCESSOR
# =========================
def infer_feature_types_after_engineering(sample_df: pd.DataFrame):
    numeric_features = []
    categorical_features = []

    for col in sample_df.columns:
        if pd.api.types.is_numeric_dtype(sample_df[col]):
            numeric_features.append(col)
        else:
            categorical_features.append(col)

    return numeric_features, categorical_features


def build_preprocessor(engineered_sample_df: pd.DataFrame) -> ColumnTransformer:
    numeric_features, categorical_features = infer_feature_types_after_engineering(engineered_sample_df)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


# =========================
# CANDIDATS MODÈLES
# =========================
def build_candidate_search_spaces(preprocessor: ColumnTransformer):
    candidates = []

    # Logistic Regression
    lr_pipeline = Pipeline(steps=[
        ("feature_engineering", ClinicalFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    lr_params = {
        "classifier__C": np.logspace(-3, 2, 20),
        "classifier__solver": ["lbfgs"]
    }

    candidates.append(("logistic_regression", lr_pipeline, lr_params))

    # Random Forest
    rf_pipeline = Pipeline(steps=[
        ("feature_engineering", ClinicalFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    rf_params = {
        "classifier__n_estimators": [200, 400, 600, 800],
        "classifier__max_depth": [None, 4, 6, 8, 10, 12],
        "classifier__min_samples_split": [2, 4, 6, 8],
        "classifier__min_samples_leaf": [1, 2, 3, 4],
        "classifier__max_features": ["sqrt", "log2", None]
    }

    candidates.append(("random_forest", rf_pipeline, rf_params))

    # Extra Trees
    et_pipeline = Pipeline(steps=[
        ("feature_engineering", ClinicalFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", ExtraTreesClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])

    et_params = {
        "classifier__n_estimators": [200, 400, 600, 800],
        "classifier__max_depth": [None, 4, 6, 8, 10, 12],
        "classifier__min_samples_split": [2, 4, 6, 8],
        "classifier__min_samples_leaf": [1, 2, 3, 4],
        "classifier__max_features": ["sqrt", "log2", None]
    }

    candidates.append(("extra_trees", et_pipeline, et_params))

    # XGBoost
    xgb_pipeline = Pipeline(steps=[
        ("feature_engineering", ClinicalFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42
        ))
    ])

    xgb_params = {
        "classifier__n_estimators": [150, 250, 400, 600],
        "classifier__max_depth": [2, 3, 4, 5, 6],
        "classifier__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
        "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "classifier__min_child_weight": [1, 2, 3, 5],
        "classifier__gamma": [0, 0.1, 0.3, 0.5],
        "classifier__reg_alpha": [0, 0.01, 0.1, 1.0],
        "classifier__reg_lambda": [1.0, 2.0, 5.0, 10.0]
    }

    candidates.append(("xgboost", xgb_pipeline, xgb_params))

    return candidates


# =========================
# TUNING
# =========================
def tune_model(model_name, pipeline, param_distributions, X_train, y_train, cv):
    print(f"\n--- Tuning {model_name} ---")

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="roc_auc",
        cv=cv,
        verbose=1,
        n_jobs=-1,
        random_state=42,
        refit=True
    )

    search.fit(X_train, y_train)

    best_score = float(search.best_score_)
    best_params = search.best_params_
    best_estimator = search.best_estimator_

    print(f"Best CV ROC-AUC for {model_name}: {best_score:.4f}")
    print(f"Best params for {model_name}: {best_params}")

    return {
        "model_name": model_name,
        "best_cv_roc_auc": best_score,
        "best_params": best_params,
        "best_estimator": best_estimator
    }


def evaluate_holdout(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    roc = float(roc_auc_score(y_test, y_prob))
    report = classification_report(y_test, y_pred)

    return {
        "accuracy": acc,
        "roc_auc": roc,
        "classification_report": report
    }


# =========================
# MAIN
# =========================
def main():
    print("Loading dataset...")
    df = load_arff_dataset(DATA_PATH)
    df = cast_base_columns(df)

    required_columns = BASE_FEATURES + [TARGET]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in dataset: {missing_columns}")

    df = df[required_columns].copy()
    df = df.dropna(subset=[TARGET])

    X = df[BASE_FEATURES].copy()
    y = df[TARGET].astype(int).copy()

    if not isinstance(y, pd.Series):
        raise TypeError("Target y is not a pandas Series. Check duplicated target columns.")

    print("Dataset shape:", df.shape)
    print("Target distribution:")
    print(y.value_counts(dropna=False))
    print()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # On infère les types après feature engineering
    engineered_sample = ClinicalFeatureEngineer().fit_transform(X_train.head(20).copy())
    preprocessor = build_preprocessor(engineered_sample)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidate_spaces = build_candidate_search_spaces(preprocessor)

    tuning_results = []
    for model_name, pipeline, param_space in candidate_spaces:
        result = tune_model(model_name, pipeline, param_space, X_train, y_train, cv)
        tuning_results.append(result)

    tuning_results = sorted(
        tuning_results,
        key=lambda x: x["best_cv_roc_auc"],
        reverse=True
    )

    best_result = tuning_results[0]
    best_model_name = best_result["model_name"]
    best_model = best_result["best_estimator"]

    print("\n==============================")
    print("BEST MODEL SELECTED:", best_model_name)
    print(f"Best CV ROC-AUC: {best_result['best_cv_roc_auc']:.4f}")
    print("==============================\n")

    print("Calibrating best model...")
    calibrated_model = CalibratedClassifierCV(
        estimator=clone(best_model),
        method="sigmoid",
        cv=3
    )
    calibrated_model.fit(X_train, y_train)

    print("Evaluating calibrated model on hold-out test set...")
    holdout_metrics = evaluate_holdout(calibrated_model, X_test, y_test)

    print(f"Test Accuracy: {holdout_metrics['accuracy']:.4f}")
    print(f"Test ROC-AUC:  {holdout_metrics['roc_auc']:.4f}")
    print("\nClassification report:")
    print(holdout_metrics["classification_report"])

    print("Computing final cross-validated ROC-AUC with best estimator on full dataset...")
    final_cv_scores = cross_val_score(
        best_model,
        X,
        y,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )
    final_cv_mean = float(np.mean(final_cv_scores))
    final_cv_std = float(np.std(final_cv_scores))

    print(f"Final full-data CV ROC-AUC: {final_cv_mean:.4f} ± {final_cv_std:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)

    model_info = {
        "best_model_name": best_model_name,
        "candidate_results": [
            {
                "model_name": r["model_name"],
                "best_cv_roc_auc": float(r["best_cv_roc_auc"]),
                "best_params": r["best_params"]
            }
            for r in tuning_results
        ],
        "test_accuracy": holdout_metrics["accuracy"],
        "test_roc_auc": holdout_metrics["roc_auc"],
        "final_cv_roc_auc_mean": final_cv_mean,
        "final_cv_roc_auc_std": final_cv_std,
        "target_name": TARGET,
        "target_meaning": {
            0: "alive",
            1: "dead"
        },
        "base_features": BASE_FEATURES
    }

    print("\nSaving artifacts...")
    joblib.dump(calibrated_model, MODEL_PATH)
    joblib.dump(BASE_FEATURES, COLUMNS_PATH)
    joblib.dump(model_info, INFO_PATH)

    print(f"Saved calibrated model to: {MODEL_PATH}")
    print(f"Saved feature list to: {COLUMNS_PATH}")
    print(f"Saved model info to: {INFO_PATH}")


if __name__ == "__main__":
    main()
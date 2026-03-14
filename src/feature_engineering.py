import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Transformer sklearn pour créer des variables cliniques dérivées
    à partir des variables pré-greffe.
    Compatible avec sklearn Pipeline + joblib serialization.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # ---------- Nettoyage simple ----------
        for col in ["Donorage", "Recipientage", "CD34kgx10d6", "Rbodymass"]:
            if col in X.columns:
                X[col] = pd.to_numeric(X[col], errors="coerce")

        # ---------- Feature engineering clinique ----------

        # Différence d'âge donneur / receveur
        if "Donorage" in X.columns and "Recipientage" in X.columns:
            X["AgeDifference"] = X["Donorage"] - X["Recipientage"]

        # Ratio CD34 par masse corporelle
        if "CD34kgx10d6" in X.columns and "Rbodymass" in X.columns:
            X["CD34_per_kg"] = X["CD34kgx10d6"] / (X["Rbodymass"] + 1e-6)

        # Flag mismatch CMV
        if "DonorCMV" in X.columns and "RecipientCMV" in X.columns:
            X["CMV_mismatch"] = (X["DonorCMV"] != X["RecipientCMV"]).astype(int)

        # Flag mismatch ABO
        if "DonorABO" in X.columns and "RecipientABO" in X.columns:
            X["ABO_mismatch"] = (X["DonorABO"] != X["RecipientABO"]).astype(int)

        # Gender mismatch
        if "Recipientgender" in X.columns and "Gendermatch" in X.columns:
            X["Gender_mismatch_flag"] = (X["Gendermatch"] == "No").astype(int)

        # Risk simplification
        if "Riskgroup" in X.columns:
            X["HighRiskFlag"] = X["Riskgroup"].astype(str).str.contains(
                "high", case=False, na=False
            ).astype(int)

        # Age bucket
        if "Recipientage" in X.columns:
            X["Recipient_age_group"] = pd.cut(
                X["Recipientage"],
                bins=[0, 20, 40, 60, 120],
                labels=["young", "adult", "middle", "senior"],
            )

        return X
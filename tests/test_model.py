import joblib
import pandas as pd


def test_model_can_predict():
    model = joblib.load("models/rf_model.pkl")

    sample = pd.DataFrame([{
        "Recipientgender": "0",
        "Stemcellsource": "1",
        "Donorage": 18,
        "Gendermatch": "0",
        "DonorABO": "1",
        "RecipientABO": "1",
        "RecipientRh": "1",
        "CMVstatus": "Donor-/Recipient-",
        "DonorCMV": "0",
        "RecipientCMV": "0",
        "Disease": "ALL",
        "Riskgroup": "0",
        "Diseasegroup": "1",
        "HLAmatch": "0",
        "HLAgrI": "0",
        "Recipientage": 5,
        "CD34kgx10d6": 8.1,
        "Rbodymass": 25
    }])

    pred = model.predict(sample)
    proba = model.predict_proba(sample)

    assert pred.shape == (1,)
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0][0] <= 1.0
    assert 0.0 <= proba[0][1] <= 1.0
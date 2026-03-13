# Pediatric BMT Success Predictor

A decision-support application developed to assist physicians in predicting the success rate of bone marrow transplants in pediatric patients. This project emphasizes model accuracy, explainability (via SHAP), and a clean web interface.

## How to Run the Project
This project is fully reproducible. To run the application locally, follow these steps:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
2. **Train the machine learning model:**
   ```bash
   python src/train_model.py
3. **Launch the Streamlit web application:**
   ```bash
   streamlit run app/app.py

## Critical Analysis & Findings

1. Was the dataset balanced? If not, how did you handle imbalance? and what was the impact?
The dataset was moderately imbalanced (~60% survived, 40% not survived). I addressed this by implementing a Class-weight adjustment strategy during model training. The impact was highly positive, allowing the model to prioritize the minority class without requiring synthetic data generation, resulting in balanced precision and recall.

2. Which ML model performed best? Provide performance metrics.
After removing target leakage to ensure an honest predictive model, XGBoost slightly outperformed Random Forest in overall ROC-AUC, though they tied in accuracy.

    XGBoost ROC-AUC: 0.745

    Random Forest ROC-AUC: 0.732

    Accuracy: 76.3% (Both)

    F1-Score: 0.727 (Both)

    Precision: 0.750 (Both)
   (ROC-AUC est un outil pour juger la qualité d’un modèle de classification. C’est très utilisé en machine learning, surtout en médecine, finance, détection de fraude… partout où un modèle doit décider entre deux classes : par exemple malade / pas malade.)

4. Which medical features most influenced predictions (SHAP results)?
After removing target leakage (survival_time), the SHAP explainability analysis revealed that the top three pre-transplant features influencing the model's predictions are:

    Relapse: The patient's relapse history.

    PLTrecovery: Platelet recovery time.

    CD34kgx10d6: The CD34+ cell dose.

5. What insights did prompt engineering provide for your selected task?
Prompt engineering was critical for debugging complex library versioning issues and handling data formatting. For example, I used iterative prompting to troubleshoot a matrix dimensionality error in the shap library caused by a recent update to TreeExplainer. I also used it to quickly write a data parser using scipy.io to automatically decode the raw .arff dataset format and dynamically clean missing values (NaNs) so the SVM model wouldn't crash during pipeline execution.
DATA ANALYSIS :
Feature Analysis and Selection for Survival Prediction

The dataset initially contains 36 features describing characteristics of the donor, recipient, transplant procedure, and post-transplant clinical outcomes. The goal of the analysis is to determine which variables should be used to maximize the predictive performance of a survival prediction model, while avoiding redundancy, noise, and information leakage.

To achieve this, several preprocessing and feature-selection steps were applied.

1. Removing Post-Transplant Variables (Data Leakage)

The first step was to remove variables that describe events occurring after the transplantation, because they cannot be used for prediction at the time of transplantation. Including them would introduce data leakage, meaning the model would indirectly use information about the future outcome.

The following variables were excluded for this reason:

Relapse – recurrence of the disease after transplantation

aGvHDIIIIV – acute graft versus host disease stage III or IV

extcGvHD – extensive chronic graft versus host disease

ANCrecovery – time to neutrophil recovery

PLTrecovery – time to platelet recovery

time_to_aGvHD_III_IV – time until severe graft versus host disease

These variables occur after the transplantation procedure and are therefore not appropriate predictors when estimating survival beforehand.

2. Removing Redundant Variables

Several variables describe the same biological information in different formats. Keeping all of them would introduce redundancy and multicollinearity, which can negatively affect the stability of machine learning models.

Examples include:

Donor age variables

Donorage (continuous age)

Donorage35 (binary threshold at 35 years)

Since the binary variable is directly derived from the continuous one, the continuous variable was preferred and the derived variable can be removed.

Recipient age variables

Recipientage (continuous)

Recipientage10 (binary threshold)

Recipientageint (age intervals)

These three variables encode the same information. The continuous variable Recipientage contains the most detail and therefore is retained.

3. Removing Derived Compatibility Variables

Some compatibility indicators are calculated directly from other variables.

Examples:

ABO compatibility

DonorABO

RecipientABO

ABOmatch

Since ABOmatch is derived from the donor and recipient blood groups, it becomes redundant if both groups are already included.

HLA compatibility

Several variables describe the same immunological compatibility:

HLAmatch

HLAmismatch

Antigen

Allele

HLAgrI

These represent different ways of quantifying the mismatch between donor and recipient HLA markers. Because they are highly correlated, typically one detailed representation (such as HLAmatch or HLAgrI) is sufficient.

4. Handling Variables with Missing Values

Some variables contain missing data:

RecipientABO

RecipientRh

CMVstatus

DonorCMV

RecipientCMV

Antigen

Allele

extcGvHD

CD3dCD34

CD3dkgx10d8

Rbodymass

When the proportion of missing values is high, the reliability of these features decreases. They can either be imputed or removed depending on the preprocessing strategy.

5. Reducing Highly Correlated Biological Variables

Certain variables describe related biological quantities and are strongly correlated.

Examples:

CD34kgx10d6 – dose of CD34+ stem cells

CD3dCD34 – ratio between CD3+ and CD34+ cells

CD3dkgx10d8 – dose of CD3+ cells

Because these variables measure similar immunological characteristics, keeping all of them may introduce redundancy. Typically one representative variable such as CD34kgx10d6 is retained.

6. Selecting Clinically Relevant Predictors

After removing post-event variables, redundant features, and variables with excessive missing data, the remaining features represent clinically meaningful predictors available before transplantation.

The most informative predictors typically include:

Recipientage – recipient age at transplantation

Donorage – donor age

Riskgroup – disease risk classification

Disease or Diseasegroup – type of hematological disease

HLAmatch or HLAgrI – immunological compatibility between donor and recipient

Stemcellsource – source of hematopoietic stem cells

CD34kgx10d6 – transplanted stem cell dose

Gendermatch – donor-recipient gender compatibility

CMVstatus – cytomegalovirus serological compatibility (if data quality allows)

These features represent biologically plausible factors influencing transplantation outcomes, making them appropriate predictors for survival estimation.

Conclusion

The feature selection process aimed to construct a robust predictive model by:

Removing variables that occur after transplantation to prevent data leakage.

Eliminating redundant or derived variables that encode the same information.

Handling variables with missing values.

Reducing highly correlated biological variables.

Retaining clinically meaningful pre-transplant predictors.

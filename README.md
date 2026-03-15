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

## Critical Analysis & Findings:

  ##   1.   Handling Class Imbalance
The dataset is moderately imbalanced (about 60% survival vs 40% non-survival). Instead of generating synthetic data with SMOTE or dropping data via undersampling, we decided to use class-weight adjustment.
By penalizing the model more heavily for misclassifying the minority class, we kept the original dataset intact. This approach gave us a solid balance between precision and recall, which is crucial for medical predictions where missing a high-risk case is problematic.


## 2.	Best Model and Performance Metrics
After preprocessing the data and dropping features that would cause data leakage, XGBoost
emerged as our best performing model.

Metrics:

•	ROC-AUC: 0.745 (vs 0.732 for Random Forest)

•	Accuracy: 76.3%

•	Precision: 0.750

•	Recall: 0.705

•	F1-Score: 0.727

## Why XGBoost? 
While Random Forest builds trees independently, XGBoost builds them sequentially, actively learning from the mistakes of previous trees. This helped it capture the complex medical relationships in our dataset a bit better.
## Why ROC-AUC? 
We prioritized ROC-AUC over raw accuracy. Accuracy just gives a percentage at a fixed threshold, but ROC-AUC evaluates the model's fundamental ability to separate survivors from non-survivors. In a clinical setting with imbalanced data, this is a much more reliable metric.

3. Which medical features most influenced predictions (SHAP results)?
  After removing post-transplant variables to prevent data leakage, SHAP analysis showed that the most influential features were clinically meaningful pre-transplant predictors, including risk group, recipient age, donor age, HLA compatibility, stem cell source, disease category, gender match, CMV status, and CD34+ cell dose. These features are medically plausible because they reflect the patient profile, donor-recipient compatibility, and transplant characteristics known before the procedure.

4. What insights did prompt engineering provide for your selected task?
  Prompt engineering was highly useful for debugging technical issues and accelerating preprocessing and implementation. It helped resolve SHAP compatibility problems, generate code to parse the ARFF dataset, and improve handling of missing values and formatting issues. For example, I used iterative prompting to troubleshoot a matrix dimensionality error in the shap library caused by a recent update to TreeExplainer. I also used it to quickly write a data parser using scipy.io to automatically decode the raw.arff dataset format and dynamically clean missing values (NaNs) so the SVM model wouldn't crash during pipeline execution.


## DATA ANALYSIS :Feature Analysis and Selection for Survival Prediction
   
   The dataset initially contains 36 features describing characteristics of the donor, recipient, transplant procedure, and     post-transplant clinical outcomes. The goal of the analysis is to determine which variables should be used to maximize       the predictive performance of a survival prediction model, while avoiding redundancy, noise, and information leakage.
   
   To achieve this, several preprocessing and feature-selection steps were applied.
   
   1. Removing Post-Transplant Variables (Data Leakage)

   
   2. Removing Redundant Variables
      Several variables describe the same biological information in different formats. Keeping all of them would introduce         redundancy and multicollinearity, which can negatively affect the stability of machine learning models. Examples             include:
   
      Donor age variables:
   
      Donorage (continuous age)
   
      Donorage35 (binary threshold at 35 years)
   
      Action: Since the binary variable is directly derived from the continuous one, the continuous variable was preferred         and the derived variable can be removed.
   
      Recipient age variables:
   
      Recipientage (continuous)
   
      Recipientage10 (binary threshold)
   
      Recipientageint (age intervals)
   
      Action: These three variables encode the same information. The continuous variable Recipientage contains the most            detail     and therefore is retained.
   
   3. Removing Derived Compatibility Variables
      Some compatibility indicators are calculated directly from other variables. Examples include:
   
      ABO compatibility: DonorABO, RecipientABO, and ABOmatch. Since ABOmatch is derived from the donor and recipient blood        groups, it becomes redundant if both groups are already included.
   
      HLA compatibility: Several variables describe the same immunological compatibility (HLAmatch, HLAmismatch, Antigen,          Allele, HLAgrI). These represent different ways of quantifying the mismatch between donor and recipient HLA markers.         Because they are highly correlated, typically one detailed representation (such as HLAmatch or HLAgrI) is sufficient.
   
   4. Handling Variables with Missing Values
       Some variables contain missing data: RecipientABO, RecipientRh, CMVstatus, DonorCMV, RecipientCMV, Antigen, Allele,          extcGvHD, CD3dCD34, CD3dkgx10d8, and Rbodymass.
   
      When the proportion of missing values is high, the reliability of these features decreases. They can either be imputed       or removed depending on the preprocessing strategy.
   
   5. Reducing Highly Correlated Biological Variables
      Certain variables describe related biological quantities and are strongly correlated. Examples include:
   
      CD34kgx10d6: dose of CD34+ stem cells
   
      CD3dCD34: ratio between CD3+ and CD34+ cells
   
      CD3dkgx10d8: dose of CD3+ cells
   
      Because these variables measure similar immunological characteristics, keeping all of them may introduce redundancy.         Typically, one representative variable such as CD34kgx10d6 is retained.
   
   6. Selecting Clinically Relevant Predictors
      After removing post-event variables, redundant features, and variables with excessive missing data, the remaining            features represent clinically meaningful predictors available before transplantation. The most informative predictors        typically include:
   
      Recipientage: recipient age at transplantation
   
      Donorage: donor age
   
      Riskgroup: disease risk classification
   
      Disease / Diseasegroup: type of hematological disease
    
      HLAmatch / HLAgrI: immunological compatibility between donor and recipient
   
      Stemcellsource: source of hematopoietic stem cells
    
      CD34kgx10d6: transplanted stem cell dose
   
      Gendermatch: donor-recipient gender compatibility
   
      CMVstatus: cytomegalovirus serological compatibility (if data quality allows)
   
      These features represent biologically plausible factors influencing transplantation outcomes, making them appropriate        predictors for survival estimation.
   
Conclusion

   The feature selection process aimed to construct a robust predictive model by:
   
   Removing variables that occur after transplantation to prevent data leakage.
   
   Eliminating redundant or derived variables that encode the same information.
   
   Handling variables with missing values.
   
   Reducing highly correlated biological variables

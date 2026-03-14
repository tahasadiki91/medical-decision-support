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
The initial analysis revealed that the pediatric bone marrow transplant dataset was moderately imbalanced, with approximately 60% of cases representing survival and 40% representing non-survival. To correct this bias without altering the integrity of the original clinical data, a class-weight adjustment strategy was prioritized during model training. This mathematical approach allowed for heavier penalization of prediction errors on the minority class, thereby eliminating the need to rely on methods that generate synthetic data, such as oversampling (SMOTE) , or that discard valuable information, such as undersampling. The impact was highly positive: the algorithm was able to effectively prioritize the detection of minority cases while learning from the entirety of the real dataset. This resulted in an excellent balance between precision and recall, an absolute requirement for the reliability of this medical decision-support application.
   ## Handling Class Imbalance Using Class Weight Adjustment
   The dataset shows a moderate class imbalance, with about 60% survived and 40% not survived. This imbalance can cause the     model to favor the majority class and reduce its ability to correctly predict the minority class. To address this issue,     we used class weight adjustment, which assigns a higher weight to the minority class during training so that the model       pays more attention to it. This method was chosen because it keeps the original dataset unchanged, unlike oversampling       (SMOTE) which creates synthetic data, and undersampling, which removes some majority class data and may lead to              information loss. 

3. Which ML model performed best? Provide performance metrics.
     After cleaning the data to remove any target leakage and ensuring our model is honest, XGBoost provided the best             results.

    Here are our performance metrics:
 
   ROC-AUC: 0.745 for XGBoost (compared to 0.732 for Random Forest)

   Accuracy: 76.3% (for both)
 
   F1-Score: 0.727 (for both)

     Precision: 0.750 (for both)
   
     Difference between Random Forest and XGBoost:
     Both models use "decision trees", but their learning methods are different:
   
     Random Forest: It builds many trees at the same time, completely independently. To make a prediction, all the trees          "vote"   and the majority wins. It is very robust, but the trees do not learn from each other.
   
     XGBoost: 
     It builds trees one after the other. Each new tree is created specifically to correct the mistakes of the previous one.      Since it learns from its errors step by step, it can often understand complex relationships in medical data better,          which    explains why it performed slightly better here.

     Why use ROC-AUC?
       
      Even though both models have the same overall accuracy of 76.3%, we chose XGBoost because of its higher ROC-AUC score.       Accuracy simply calculates the percentage of correct answers at a fixed threshold (for example, 50% probability). ROC-       AUC, on the other hand, evaluates the model's overall ability to separate the two groups (patients who survive and           those     who do not), regardless of the chosen threshold. In medicine, this is a much more important metric because         it proves         that the model can fundamentally distinguish between a high-risk profile and a safe one.

3. Which medical features most influenced predictions (SHAP results)?
  After removing post-transplant variables to prevent data leakage, SHAP analysis showed that the most influential features were clinically meaningful pre-transplant predictors, including risk group, recipient age, donor age, HLA compatibility, stem cell source, disease category, gender match, CMV status, and CD34+ cell dose. These features are medically plausible because they reflect the patient profile, donor-recipient compatibility, and transplant characteristics known before the procedure.

4. What insights did prompt engineering provide for your selected task?
  Prompt engineering was highly useful for debugging technical issues and accelerating preprocessing and implementation. It helped resolve SHAP compatibility problems, generate code to parse the ARFF dataset, and improve handling of missing values and formatting issues. For example, I used iterative prompting to troubleshoot a matrix dimensionality error in the shap library caused by a recent update to TreeExplainer. I also used it to quickly write a data parser using scipy.io to automatically decode the raw.arff dataset format and dynamically clean missing values (NaNs) so the SVM model wouldn't crash during pipeline execution.


## DATA ANALYSIS :Feature Analysis and Selection for Survival Prediction
   
   The dataset initially contains 36 features describing characteristics of the donor, recipient, transplant procedure, and     post-transplant clinical outcomes. The goal of the analysis is to determine which variables should be used to maximize       the predictive performance of a survival prediction model, while avoiding redundancy, noise, and information leakage.
   
   To achieve this, several preprocessing and feature-selection steps were applied.
   
   1. Removing Post-Transplant Variables (Data Leakage)
    The first step was to remove variables that describe events occurring after the transplantation, because they cannot be      used for prediction at the time of transplantation. Including them would introduce data leakage, meaning the model
     would indirectly use information about the future outcome.
   
     The following variables were excluded for this reason:
   
     Relapse: recurrence of the disease after transplantation
   
     aGvHDIIIIV: acute graft versus host disease stage III or IV
   
     extcGvHD: extensive chronic graft versus host disease
   
     ANCrecovery: time to neutrophil recovery
   
     PLTrecovery: time to platelet recovery
   
     time_to_aGvHD_III_IV: time until severe graft versus host disease
   
     These variables occur after the transplantation procedure and are therefore not appropriate predictors when estimating       survival beforehand.
   
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

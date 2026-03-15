# Pediatric BMT Success Predictor

A decision-support application developed to assist physicians in predicting the success rate of bone marrow transplants in pediatric patients. This project emphasizes model accuracy, explainability (via SHAP), and a clean web interface.

## How to Run the Project
This project is fully reproducible. To run the application locally, follow these steps:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
2. **Train the machine learning model:**
   ```bash
   python -m src.train_model
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

## 3.	Most Influential Features (SHAP)
Based on our SHAP summary plots, the most critical pre-transplant predictors are:

•	Risk group
 
•	Recipient age & Donor age

•	HLA compatibility

•	Stem cell source

•	Disease category

•	Gender match

•	CMV status

•	CD34+ cell dose
These results make clinical sense, as they are all standard risk factors and compatibility metrics that physicians evaluate before authorizing a transplant.


## 4.	Prompt Engineering Insights

Prompt engineering was highly useful for debugging technical issues and accelerating preprocessing and implementation. It helped resolve SHAP compatibility problems, generate code to parse the ARFF dataset, and improve handling of missing values and formatting issues. For example, I used iterative prompting to troubleshoot a matrix dimensionality error in the shap library caused by a recent update to TreeExplainer. I also used it to quickly write a data parser using scipy.io to automatically decode the raw.arff dataset format and dynamically clean missing values (NaNs) so the SVM model wouldn't crash during pipeline execution.


## DATA ANALYSIS :Feature Analysis and Selection for Survival Prediction
   
We initially started with 36 features describing donor and recipient characteristics, the transplant procedure, and post-transplant clinical outcomes. To build a robust predictive model, our goal was to maximize performance while actively eliminating noise, redundancy, and information leakage.
Here is the step-by-step preprocessing strategy we applied:

1.	Preventing Data Leakage (Post-Transplant Variables) We strictly removed all variables that recorded events occurring after the transplant (e.g., acute GvHD, survival time). Keeping them would introduce severe data leakage, allowing the model to "cheat" by looking at future outcomes rather than predicting from pre-transplant conditions.
2.	Removing Redundant Variables Several features described the exact same biological information but in different formats. Keeping them all would introduce multicollinearity.
 	Age variables: We had continuous age ( Donorage , Recipientage ) and binned thresholds ( Donorage35 , Recipientage10 , Recipientageint ). We kept the continuous variables because they contain the most precise detail, and we dropped the derived categorical ones.
3.	Removing Derived Compatibility Variables Some compatibility indicators were mathematically derived from other existing columns:
ABO compatibility: ABOmatch is completely derived from DonorABO and RecipientABO .
 	HLA compatibility: We had multiple features representing the same immunological mismatch ( HLAmatch , HLAmismatch , Antigen , Allele , HLAgrI ). We selected one detailed representation for HLA and dropped the rest to stabilize the model.
4.	Handling Missing Values Several variables contained missing data (e.g., RecipientABO , CMVstatus , CD3dCD34 ). We evaluated the proportion of missing values for each: features with an overwhelmingly high percentage of missing data were dropped because they were unreliable. For the remaining features, we imputed the missing values so we wouldn't have to discard valuable patient records.
5.	Reducing Highly Correlated Biological Data Certain variables measured very similar immunological characteristics. For example, the dose of CD34+ cells ( CD34kgx10d6 ), the dose of CD3+ cells ( CD3dkgx10d8 ), and their ratio ( CD3dCD34 ) were highly correlated. To avoid redundancy, we retained CD34kgx10d6 as our primary representative feature.
## Conclusion & Final Predictors
Through this selection process, we successfully reduced the dataset to only the most clinically relevant, pre-transplant predictors. Our final features include:
Demographics: Recipientage , Donorage
 
Clinical Profile: Riskgroup , Diseasegroup

Compatibility: HLAmatch (or HLAgrI ), Gendermatch , CMVstatus

 Transplant Details: Stemcellsource , CD34kgx10d6

These represent biologically plausible factors that influence transplantation outcomes, making them the perfect inputs for our survival estimation model.



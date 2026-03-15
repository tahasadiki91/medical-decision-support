# Pediatric BMT Success Predictor

A decision-support application developed to assist physicians in predicting the success rate of bone marrow transplants in pediatric patients. This project emphasizes model accuracy, explainability with SHAP, and a clean web interface for transparent clinical decision support.

---

## Project Overview

The goal of this project is to build an accurate, interpretable, and reproducible medical decision-support system for pediatric bone marrow transplant survival prediction.

The application:
- predicts transplant success probability,
- explains predictions using SHAP,
- provides a clean Streamlit interface for physicians,
- follows a structured machine learning workflow from preprocessing to deployment.

---

## Dataset

This project uses the **Pediatric Bone Marrow Transplant Children** dataset from the UCI Machine Learning Repository.

The dataset contains pre-transplant and transplant-related clinical variables describing:
- donor and recipient demographics,
- disease and risk characteristics,
- compatibility indicators,
- transplant procedure information.

Our objective is to predict the binary target **`survival_status`**.

---

## How to Run the Project

This project is fully reproducible.

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the machine learning model
```bash
python src/train_model.py
```

### 3. Launch the Streamlit web application
```bash
streamlit run app/app.py
```

---

## Critical Questions for the README

### 1. Was the dataset balanced? If not, how was imbalance handled and what was the impact?

The dataset was not perfectly balanced, but it showed a **moderate class imbalance**, with about **60% survived patients** and **40% non-survived patients**.

This imbalance can affect machine learning performance because a model may naturally favor the majority class. To reduce this effect, we used **class-weight adjustment** instead of SMOTE or undersampling.

This choice was made for two reasons:
- undersampling would remove valuable clinical observations,
- synthetic oversampling can introduce artificial patterns into medical data.

By penalizing the model more heavily for misclassifying the minority class, we kept the original dataset intact while improving balance between classes during training. The main impact was a more reliable evaluation in terms of **precision, recall, F1-score, and ROC-AUC**, which are more informative than accuracy alone in a medical classification setting.

---

### 2. Which ML model performed best? Provide performance metrics.

After preprocessing the data and dropping features that would cause data leakage, **XGBoost** emerged as our best-performing model.

#### Final Performance Metrics

| Model | Accuracy | ROC-AUC | Precision | Recall | F1-Score |
|------|----------|---------|-----------|--------|----------|
| XGBoost | 0.763 | 0.745 | 0.750 | 0.705 | 0.727 |
| Random Forest | — | 0.732 | — | — | — |
| SVM | — | — | — | — | — |

### Why XGBoost?

While Random Forest builds trees independently, XGBoost builds them sequentially and learns from the mistakes of previous trees. This helped it capture the complex medical relationships in the dataset more effectively.

### Why ROC-AUC?

We prioritized **ROC-AUC** over raw accuracy. Accuracy reflects performance at one decision threshold, while ROC-AUC evaluates the model's overall ability to separate survivors from non-survivors. In a clinical setting with moderately imbalanced data, this is a more reliable metric.

---

### 3. Which medical features most influenced predictions (SHAP results)?

Based on our SHAP summary plots, the most critical pre-transplant predictors were:

- **Risk group**
- **Recipient age and donor age**
- **HLA compatibility**
- **Stem cell source**
- **Disease category**
- **Gender match**
- **CMV status**
- **CD34+ cell dose**

These results are clinically meaningful because they reflect standard medical risk factors and compatibility measures considered before authorizing a transplant.

---

### 4. What insights did prompt engineering provide for the selected task?

Prompt engineering was useful for debugging technical issues and accelerating preprocessing and implementation.

It helped us:
- resolve SHAP compatibility problems,
- generate code to parse the ARFF dataset,
- improve handling of missing values,
- fix formatting and pipeline issues,
- generate memory optimization logic.

#### Example Prompt 1
**Prompt used:**
> Write a Python function `optimize_memory(df)` that reduces memory usage by converting numeric columns to smaller valid dtypes without changing the data values.

**Result obtained:**
A reusable function that downcasted integer and float columns and allowed us to demonstrate memory usage before and after optimization.

**Insight:**
This prompt was effective because it was precise and task-oriented. A possible improvement would be to explicitly request memory reduction reporting in MB and percentage.

#### Example Prompt 2
**Prompt used:**
> Help me debug a SHAP TreeExplainer dimensionality error after updating the SHAP library. Adapt the code so it works whether `shap_values` is returned as a list or a 3D NumPy array.

**Result obtained:**
A more robust SHAP plotting block compatible with newer SHAP versions.

**Insight:**
Iterative prompting was especially useful for solving library-version issues that were difficult to diagnose manually.

---

## Data Analysis: Feature Analysis and Selection for Survival Prediction

We initially started with **36 features** describing donor and recipient characteristics, the transplant procedure, and post-transplant clinical outcomes. To build a robust predictive model, our goal was to maximize performance while eliminating noise, redundancy, and information leakage.

### Step-by-step preprocessing strategy

1. **Preventing data leakage**  
   We removed all variables recording events occurring after the transplant, such as acute GvHD variables and survival time. Keeping them would allow the model to "cheat" by using future outcomes instead of pre-transplant information.

2. **Removing redundant variables**  
   Some features described the same biological information in different formats. For age, we had continuous variables (`Donorage`, `Recipientage`) and grouped thresholds (`Donorage35`, `Recipientage10`, `Recipientageint`). We kept the continuous variables because they contain the most precise information.

3. **Removing derived compatibility variables**  
   Some compatibility features were derived from others. For example:
   - `ABOmatch` is derived from `DonorABO` and `RecipientABO`
   - multiple HLA-related features (`HLAmatch`, `HLAmismatch`, `Antigen`, `Allele`, `HLAgrI`) described overlapping immunological mismatch information

   We simplified these variables to stabilize the model and improve interpretability.

4. **Handling missing values**  
   Several features contained missing data, such as `RecipientABO`, `CMVstatus`, and `CD3dCD34`. Features with extremely high missingness were dropped when considered unreliable. The remaining missing values were imputed to preserve as many patient records as possible.

5. **Reducing highly correlated biological data**  
   Some transplant biology variables were strongly related. For example, `CD34kgx10d6`, `CD3dkgx10d8`, and `CD3dCD34` carried overlapping information. To reduce redundancy, we retained `CD34kgx10d6` as the main representative predictor.

---

## Correlation Analysis

We computed a correlation matrix on numerical features to identify redundancy before training.

### Most Correlated Feature Pairs
- `aGvHDIIIIV` ↔ `time_to_aGvHD_III_IV` : **0.969**
- `HLAmatch` ↔ `HLAgrI` : **0.947**
- `Recipientageint` ↔ `Recipientage` : **0.917**
- `Allele` ↔ `HLAmatch` : **0.904**

### Interpretation

These results show that several variables encode very similar information. Instead of keeping all correlated variables equally, we used this analysis to guide feature selection and reduce redundancy.

### Impact on the project
- simpler and cleaner feature space,
- lower redundancy between predictors,
- better interpretability,
- more reliable downstream SHAP explanations.

> See `notebooks/eda.ipynb` for the complete correlation matrix, visualizations, and full preprocessing logic.

---

## Conclusion and Final Predictors

Through this selection process, we reduced the dataset to the most clinically relevant pre-transplant predictors.

The final feature set includes:
- **Demographic variables:** Recipient Age, Donor Age
- **Clinical profile:** Risk Group, Disease Group
- **Compatibility factors:** HLA Match or HLAgrI, Gender Match, CMV Status
- **Transplant characteristics:** Stem Cell Source, CD34kgx10⁶

These variables represent biologically meaningful factors that can influence transplantation outcomes, making them appropriate inputs for survival prediction.

---

## Project Structure

```bash
project/
│
├── notebooks/
│   └── eda.ipynb
├── src/
│   ├── data_processing.py
│   ├── train_model.py
│   └── evaluate_model.py
├── app/
│   └── app.py
├── tests/
├── requirements.txt
└── README.md
```

---

## Reproducibility

To reproduce the full project:

```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run app/app.py
```

The notebook documents the exploratory analysis, the `src/` folder contains preprocessing and training logic, and the `app/` folder contains the web interface.

---

## Notes

- SHAP is used to improve transparency and trust in predictions.
- The application is designed for decision support, not to replace clinical judgment.
- The final goal is to provide accurate, interpretable, and reproducible machine learning support for transplant outcome prediction.

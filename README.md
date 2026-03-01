#  Stroke Prediction using Machine Learning

A machine learning study comparing four classification algorithms for predicting stroke occurrence in patients, using an imbalanced medical dataset with undersampling techniques.

> **Course:** Data Warehouses and Data Mining  
> **Institution:** Ionian University — Department of Informatics  
> **Author:** Στέργιος Μουτζίκος   
> **Date:** May 2025

---

##  Project Overview

Stroke is one of the leading causes of death and long-term disability worldwide. This project applies supervised machine learning on clinical and demographic patient data to predict stroke risk. Special focus is placed on handling **class imbalance** (only ~5% positive cases) using **Random Undersampling**, and evaluating models with a comprehensive set of metrics.

---

##  Algorithms Compared

| Model | Accuracy | Recall | F1 Score | ROC AUC |
|---|---|---|---|---|
| Logistic Regression | **0.811** | **0.920** | **0.830** | 0.847 |
| SVM | 0.807 | 0.884 | 0.821 | **0.851** |
| Random Forest | 0.777 | 0.908 | 0.804 | 0.826 |
| XGBoost | 0.773 | 0.892 | 0.798 | 0.821 |

> Evaluated using **10-Fold Stratified Cross-Validation** with optimal threshold tuning per fold.

---

##  Repository Structure

```
stroke-prediction/
└── Code/
    ├── Stroke_Prediction.ipynb        # main code
    └── Stroke_Prediction.pdf          # notebook as PDF
└── Data/
    ├── stroke-data.csv                # Original raw dataset
    ├── stroke-data_updated.csv        # Preprocessed dataset
    └── stroke-data-undersampled.csv   # Balanced dataset (undersampled)
└── Paper/
    ├── Stroke_prediction_paper.pdf 
    └── Stroke_prediction.tex
├── README.md
├── requirements.txt                                  


```

---

##  Pipeline Summary

1. **Data Loading & EDA** — 5,110 patients, 12 features
2. **Missing Value Handling** — BMI mean imputation (201 missing values)
3. **Encoding** — Label encoding for binary columns, One-Hot for multi-category
4. **Outlier Handling** — Z-score capping (threshold = 3) on `avg_glucose_level` and `bmi`
5. **Dimensionality Reduction** — PCA visualization with K-Means clustering
6. **Class Balancing** — RandomUnderSampler (249 samples per class)
7. **Model Training & Evaluation** — 10-Fold CV with dynamic threshold optimization

---

##  Dataset

- **Source:** Electronic medical records
- **Size:** 5,110 patients × 12 features
- **Target:** `stroke` (0 = No stroke, 1 = Stroke)
- **Class imbalance:** ~95% No stroke / ~5% Stroke

**Features include:** age, gender, hypertension, heart disease, marital status, work type, residence type, average glucose level, BMI, smoking status.

---

##  Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
scipy
umap-learn
imbalanced-learn
xgboost
jupyter
```

---

##  Report

The full academic paper is available in [`Stroke_prediction_paper.pdf`](./Paper/Stroke_prediction_paper.pdf), covering:
- Literature review on stroke prediction with ML
- Detailed methodology and preprocessing steps
- Experimental results and model comparison
- Conclusions and future improvement suggestions

---

##  Key Findings

- **Logistic Regression** offers the best combination of recall (0.920), F1 (0.830), and interpretability — ideal for medical applications where minimizing false negatives is critical.
- **SVM** achieves the highest ROC AUC (0.851) and lowest RMSE, making it best for probability estimation.
- Undersampling successfully addressed the severe class imbalance and produced balanced, reliable evaluation results.

---

##  Future Improvements

- Feature engineering to create more informative variables
- Test LightGBM and CatBoost classifiers
- Hyperparameter tuning with Grid Search or Bayesian Optimization
- External dataset validation for generalizability
- Model explainability with SHAP or LIME

---

##  References

- Scikit-learn: Pedregosa et al. (2011)
- Imbalanced-learn: Lemaitre et al. (2017)  
- XGBoost: Chen & Guestrin (2016)
- Kokkotis et al. (2022) — Stroke Prediction on Imbalanced Data
- Hassan et al. (2024) — Key Risk Factors for Stroke using ML

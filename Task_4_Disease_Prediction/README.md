# Task 4: Disease Prediction from Medical Data
Author: Harpreet Singh

# Objective
To build a machine learning model capable of predicting the likelihood of heart disease based on clinical patient data and lifestyle habits.

# Dataset
* *Source:* Custom `heart_disease.csv` dataset containing 10,000 patient records.
* *Features:* The dataset includes 21 medical indicators. Through feature importance analysis, the top 5 most critical predictors were identified as: Sleep Hours, BMI, Homocysteine Level, CRP Level, and Triglyceride Level.

# Approach & Methodology
1.  *Data Preprocessing:* Handled missing values by imputing the median for numerical data and the mode for categorical data. Applied One-Hot Encoding for text variables and utilized `StandardScaler` to normalize clinical measurements with varying ranges.
2.  *Algorithm Selection:* Implemented a **Random Forest Classifier**. The model was explicitly configured with `class_weight='balanced'` to address the 80/20 target class imbalance present in the medical data.
3.  *Model Training:* Split the dataset into an 80% training set and a 20% testing set to ensure an unbiased evaluation of the model's pipeline.

# Evaluation Metrics
The machine learning pipeline successfully processed the clinical data and achieved the following baseline metrics:
* *Accuracy:* 0.81
* *ROC-AUC Score:* 0.5006
* *Primary Predictive Feature:* Sleep Hours (9.64% relative importance)

## Files in this Folder
* `disease_prediction.py`: The complete Python script containing the data preprocessing, model training, and evaluation code.
* `heart_disease.csv`: The clinical dataset used to train the model.

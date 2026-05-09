Task 1: Credit Scoring Model
Author: Harpreet Singh

# Objective
To build a machine learning model capable of predicting an individual's creditworthiness based on their past financial history.

# Dataset
   Source: "https://www.kaggle.com/datasets/mpwolke/cusersmarildownloadsgermancsv?select=german.csv"
   Features: The dataset includes financial indicators such as income, existing debts, payment history, and loan duration.

# Approach & Methodology
1.  Data Preprocessing: Handled missing values and encoded categorical variables (e.g., payment history) into numerical formats using `LabelEncoder`.
2.  Algorithm Selection: Implemented a *Random Forest Classifier* due to its robustness with tabular data and ability to handle both linear and non-linear relationships.
3.  Model Training: Split the dataset into 80% training and 20% testing sets to ensure unbiased evaluation.

# Evaluation Metrics
The model was evaluated using the following metrics to ensure a balance between identifying good creditors and minimizing the risk of approving bad loans:
*   *Accuracy:* 0.76
*   *Precision:* 0.78
*   *Recall:* 0.91
*   *F1-Score:* 0.84
*   *ROC-AUC:* 0.7712

#  Insights from Model
Top 5 Financial Features Influencing Credit:
*  Credit Amount: 13.01%
*  Account Balance: 11.96%
*  Duration of Credit (Monthly): 10.51
*  Age (Years): 10.28%
*  Payment Status of Previous Credit: 7.02%

# Files in this Folder
*   `task1.py` : The main script containing the model training and evaluation code.
*   `german.csv` : The financial data used for training.

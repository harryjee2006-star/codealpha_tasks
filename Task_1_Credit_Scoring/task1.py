import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report , roc_auc_score

data = pd.read_csv("code alpha/data/german.csv" , sep= ";")
X = data.drop("Creditability" , axis = 1 )
y = data["Creditability"]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size= 0.2 , random_state= 42)

model = RandomForestClassifier(n_estimators= 100 , max_depth= 10 , random_state= 42)
model.fit(X_train , y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Task 1: Credit Scoring Model Results")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 5 Financial Features Influencing Credit:")
print(importances.head(5))
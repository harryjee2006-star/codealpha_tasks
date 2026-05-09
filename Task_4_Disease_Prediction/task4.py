import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

print("Loading dataset...")
data = pd.read_csv('code alpha/data/heart_disease.csv')

data = data.dropna(subset=['Heart Disease Status'])

X = data.drop('Heart Disease Status', axis=1)
y = data['Heart Disease Status']

le_y = LabelEncoder()
y = le_y.fit_transform(y)

num_cols = X.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())
for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

X = pd.get_dummies(X, columns=cat_cols, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n--- Model Evaluation ---")
print("Target Classes:", le_y.classes_)
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le_y.classes_))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(5)
print("\nTop 5 Predictive Features:\n", importances)
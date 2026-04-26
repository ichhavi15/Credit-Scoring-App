import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, "scaler.pkl")

data = pd.read_csv("loan_data.csv")

# clean column names
data.columns = data.columns.str.strip()

# 👇 DEBUG (very important)
print(data.columns)

# ✅ correct column name check
# (maybe it's 'loan_status' OR 'Loan_Status')
# so use this:

data['loan_status'] = data['loan_status'].astype(str).str.strip()

# convert target
data['loan_status'] = data['loan_status'].map({'Approved': 1, 'Rejected': 0})

# 🔥 check after mapping
print(data['loan_status'].value_counts())

# ❗ drop only if still NaN
data = data.dropna(subset=['loan_status'])

# convert text
data['education'] = data['education'].map({'Graduate': 1, 'Not Graduate': 0})
data['self_employed'] = data['self_employed'].map({'Yes': 1, 'No': 0})

# dependents fix
data['no_of_dependents'] = pd.to_numeric(data['no_of_dependents'], errors='coerce')

# features
X = data.drop(['loan_id', 'loan_status'], axis=1)
X = X.fillna(0)

y = data['loan_status']

# ❗ FINAL CHECK
print("Data shape:", X.shape)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
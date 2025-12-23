import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================
# 1. Load dataset hasil preprocessing
# ===============================
data = pd.read_csv("namadataset_preprocessing/titanic_clean.csv")

# ===============================
# 2. Pisahkan fitur dan label
# ===============================
X = data.drop(columns=["Survived"])
y = data["Survived"]

# ===============================
# 3. Split data
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 4. Set MLflow experiment
# ===============================
mlflow.set_experiment("Titanic_Classification")

# ===============================
# 5. AKTIFKAN AUTOLOG
# ===============================
mlflow.sklearn.autolog()

# ===============================
# 6. Training model
# ===============================
model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

# ===============================
# 7. Evaluasi manual (INI NILAI PLUS)
# ===============================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# ===============================
# 8. Manual logging (nilai tambah)
# ===============================
mlflow.log_metric("accuracy_manual", acc)
mlflow.log_metric("precision_manual", prec)
mlflow.log_metric("recall_manual", rec)
mlflow.log_metric("f1_manual", f1)

print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1-score :", f1)
print("✅ Training via MLflow Project berhasil")

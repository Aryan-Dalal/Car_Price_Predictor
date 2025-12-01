# ==============================
# Car Price & Performance Prediction Script (Enhanced with Accuracy Display)
# ==============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

sns.set(style="whitegrid")

# ------------------------------
# 1️⃣ Create folders
# ------------------------------
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("accuracy/plots", exist_ok=True)

# ------------------------------
# 2️⃣ Load Dataset
# ------------------------------
try:
    data = pd.read_csv("data/CarPrice_Assignment.csv")
except Exception as e:
    print("Error loading dataset:", e)
    exit()

# ------------------------------
# 3️⃣ Data Preprocessing
# ------------------------------
categorical_cols = ['fueltype', 'aspiration', 'doornumber', 'carbody',
                    'drivewheel', 'enginetype', 'cylindernumber', 'fuelsystem']

le = LabelEncoder()
for col in categorical_cols:
    if col in data.columns:
        data[col] = le.fit_transform(data[col])

# ------------------------------
# 3️⃣1️⃣ Performance Score Scaling
# ------------------------------
perf_raw = ((data['price'] / (data['citympg'] + 1)) * (2025 - data['symboling']))
scaler = MinMaxScaler(feature_range=(1, 100))
data['Performance_Score'] = scaler.fit_transform(perf_raw.values.reshape(-1, 1))
joblib.dump(scaler, "models/perf_scaler.pkl")

# ------------------------------
# 3️⃣2️⃣ Features and Targets
# ------------------------------
feature_cols = ['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody',
                'drivewheel', 'enginetype', 'cylindernumber', 'horsepower',
                'peakrpm', 'citympg', 'highwaympg']

X = data[feature_cols]
y_price = data['price']
y_perf = data['Performance_Score']

# Train-test split
X_train, X_test, y_price_train, y_price_test = train_test_split(X, y_price, test_size=0.2, random_state=42)
_, _, y_perf_train, y_perf_test = train_test_split(X, y_perf, test_size=0.2, random_state=42)

# ------------------------------
# 4️⃣ Base Regressors
# ------------------------------
lr = LinearRegression()
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)

lr.fit(X_train, y_price_train)
dt.fit(X_train, y_price_train)
rf.fit(X_train, y_price_train)

# ------------------------------
# 5️⃣ Voting Regressor (Price)
# ------------------------------
voting = VotingRegressor([('lr', lr), ('dt', dt), ('rf', rf)])
voting.fit(X_train, y_price_train)

# ------------------------------
# 6️⃣ Multi-output Regressor (Price + Performance)
# ------------------------------
multi_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
multi_rf.fit(X_train, pd.concat([y_price_train, y_perf_train], axis=1))

# ------------------------------
# 7️⃣ Model Evaluation
# ------------------------------
models = {
    "Linear Regression": lr,
    "Decision Tree": dt,
    "Random Forest": rf,
    "Voting Regressor": voting
}

accuracy_results = []

print("\n📊 Model Accuracy Results\n" + "="*35)

for name, model in models.items():
    y_pred = model.predict(X_test)
    r2 = r2_score(y_price_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_price_test, y_pred))
    accuracy_results.append((name, r2, rmse))

    # --- Print on terminal ---
    print(f"\n{name}:")
    print(f"R² Score : {r2:.4f}")
    print(f"RMSE     : {rmse:.4f}")

    # Plot 1: Actual vs Predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(y_price_test, y_pred, alpha=0.6, color='blue')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"{name} - Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"accuracy/plots/{name.replace(' ', '_')}_actual_vs_pred.png")
    plt.close()

    # Plot 2: Error Distribution
    errors = y_price_test - y_pred
    plt.figure(figsize=(6, 4))
    sns.histplot(errors, bins=20, kde=True, color='red')
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.title(f"{name} - Error Distribution")
    plt.tight_layout()
    plt.savefig(f"accuracy/plots/{name.replace(' ', '_')}_error_dist.png")
    plt.close()

# ------------------------------
# 8️⃣ Multi-output Accuracy
# ------------------------------
multi_pred = multi_rf.predict(X_test)
multi_price_pred = multi_pred[:, 0]
multi_perf_pred = multi_pred[:, 1]

r2_price_multi = r2_score(y_price_test, multi_price_pred)
r2_perf_multi = r2_score(y_perf_test, multi_perf_pred)

rmse_price_multi = np.sqrt(mean_squared_error(y_price_test, multi_price_pred))
rmse_perf_multi = np.sqrt(mean_squared_error(y_perf_test, multi_perf_pred))

accuracy_results.append(("MultiOutput RF (Price)", r2_price_multi, rmse_price_multi))
accuracy_results.append(("MultiOutput RF (Performance)", r2_perf_multi, rmse_perf_multi))

# --- Print MultiOutput scores on terminal ---
print("\nMultiOutput RandomForest Results:")
print(f"Price Model     → R²: {r2_price_multi:.4f}, RMSE: {rmse_price_multi:.4f}")
print(f"Performance Model → R²: {r2_perf_multi:.4f}, RMSE: {rmse_perf_multi:.4f}")

# ------------------------------
# 9️⃣ Save Accuracy Report
# ------------------------------
os.makedirs("accuracy", exist_ok=True)
with open("accuracy/metrics.txt", "w") as f:
    f.write("Model Accuracy Metrics\n")
    f.write("=" * 40 + "\n\n")
    for name, r2, rmse in accuracy_results:
        f.write(f"{name}\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write("-" * 40 + "\n")

print("\n✅ Accuracy metrics saved to 'accuracy/metrics.txt'")
print("✅ All accuracy plots saved to 'accuracy/plots/'")

# ------------------------------
# 🔟 Save Models
# ------------------------------
joblib.dump(lr, "models/lr_model.pkl")
joblib.dump(dt, "models/dt_model.pkl")
joblib.dump(rf, "models/rf_model.pkl")
joblib.dump(voting, "models/voting_model.pkl")
joblib.dump(multi_rf, "models/multi_rf_model.pkl")

print("\n✅ All models saved successfully.\n")

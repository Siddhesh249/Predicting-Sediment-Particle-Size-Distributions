import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores.")
sns.set_theme(style="whitegrid", context="paper")

# Ensure Results directory exists
os.makedirs('Results', exist_ok=True)

# -------------------------------
# Data Loading & Preprocessing
# -------------------------------
data = pd.read_csv("Dataset.csv")
print(f"Dataset loaded. Initial shape: {data.shape}")

if 'time' in data.columns:
    data = data.drop(columns=['time'])
data = data.dropna()

# Remove strong outliers
data = data[data['d50'] < data['d50'].quantile(0.99)]
data = data[data['sigma2'] < data['sigma2'].quantile(0.99)]
print(f"Dataset shape after preprocessing: {data.shape}")

features = ['S', 'ub', 'np', 'T', 'a676_a650', 'a450_a676', 'chl_a', 'u']
X = data[features]
y_d50 = data['d50']
y_sigma2 = data['sigma2']

# Train-test split
X_train, X_test, y_train_d50, y_test_d50 = train_test_split(X, y_d50, test_size=0.3, random_state=42)
X_train2, X_test2, y_train_sigma2, y_test_sigma2 = train_test_split(X, y_sigma2, test_size=0.3, random_state=42)

# ============================================================
# RANDOM FOREST REGRESSION
# ============================================================
print("\n--- Training Random Forest Models ---")

# OOB Curve Generation
n_trees = [10, 20, 40, 60, 100, 200, 400]
oob_scores_d50, oob_scores_sigma2 = [], []

for n in n_trees:
    rf_d50_tmp = RandomForestRegressor(n_estimators=n, oob_score=True, random_state=42, n_jobs=-1)
    rf_d50_tmp.fit(X_train, y_train_d50)
    oob_scores_d50.append(rf_d50_tmp.oob_score_)

    rf_sigma2_tmp = RandomForestRegressor(n_estimators=n, oob_score=True, random_state=42, n_jobs=-1)
    rf_sigma2_tmp.fit(X_train2, y_train_sigma2)
    oob_scores_sigma2.append(rf_sigma2_tmp.oob_score_)

plt.figure(figsize=(6, 4))
plt.plot(n_trees, oob_scores_d50, 'k-', label='d50')
plt.plot(n_trees, oob_scores_sigma2, 'r--', label='σ²')
plt.xlabel("Number of Trees")
plt.ylabel("OOB Score")
plt.title("OOB Score vs Number of Trees")
plt.legend()
plt.tight_layout()
plt.savefig("Results/oob_score_curve.png", dpi=150)
plt.close()

# Train Final RF Models
rf_d50 = RandomForestRegressor(n_estimators=400, random_state=42, oob_score=True, n_jobs=-1)
rf_sigma2 = RandomForestRegressor(n_estimators=400, random_state=42, oob_score=True, n_jobs=-1)

rf_d50.fit(X_train, y_train_d50)
rf_sigma2.fit(X_train2, y_train_sigma2)

y_pred_d50_rf = rf_d50.predict(X_test)
y_pred_sigma2_rf = rf_sigma2.predict(X_test2)

r2_d50_rf = r2_score(y_test_d50, y_pred_d50_rf)
r2_sigma2_rf = r2_score(y_test_sigma2, y_pred_sigma2_rf)
print("Random Forest Performance:")
print(f"  R² for d50: {r2_d50_rf:.4f}")
print(f"  R² for σ²:  {r2_sigma2_rf:.4f}")

# Feature Importance Plot
importances = pd.Series(rf_d50.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(7, 4))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette='viridis', legend=False)
plt.title("Feature Importance for d50 (Random Forest)")
plt.tight_layout()
plt.savefig("Results/feature_importance_d50.png", dpi=150)
plt.close()

# ============================================================
# SUPPORT VECTOR REGRESSION
# ============================================================
print("\n--- Training SVR Models ---")

# As per the paper, SVR is trained only on the top-4 important features
top4 = ['S', 'ub', 'np', 'T']
Xtr_d50_top4 = pd.DataFrame(X_train, columns=features)[top4]
Xte_d50_top4 = pd.DataFrame(X_test, columns=features)[top4]
Xtr_s2_top4  = pd.DataFrame(X_train2, columns=features)[top4]
Xte_s2_top4  = pd.DataFrame(X_test2, columns=features)[top4]

svr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf'))
])

param_grid = {
    'svr__C': [1, 10, 100, 500, 1000, 2048, 3000],
    'svr__epsilon': [0.1, 0.5, 1, 2, 4]
}

# Grid Search for d50
print("Running Grid Search for SVR (d50)...")
grid_d50 = GridSearchCV(svr_pipe, param_grid, scoring='r2', cv=5, n_jobs=-1)
grid_d50.fit(Xtr_d50_top4, y_train_d50)
svr_d50_best = grid_d50.best_estimator_

# Grid Search for σ²
print("Running Grid Search for SVR (σ²)...")
grid_sigma2 = GridSearchCV(svr_pipe, param_grid, scoring='r2', cv=5, n_jobs=-1)
grid_sigma2.fit(Xtr_s2_top4, y_train_sigma2)
svr_sigma2_best = grid_sigma2.best_estimator_

print(f"Best Hyperparameters (d50): {grid_d50.best_params_}")
print(f"Best Hyperparameters (σ²):  {grid_sigma2.best_params_}")

y_pred_d50_svr = svr_d50_best.predict(Xte_d50_top4)
y_pred_sigma2_svr = svr_sigma2_best.predict(Xte_s2_top4)

r2_d50_svr = r2_score(y_test_d50, y_pred_d50_svr)
r2_sigma2_svr = r2_score(y_test_sigma2, y_pred_sigma2_svr)
print("SVR Performance (Top-4 Features):")
print(f"  R² for d50: {r2_d50_svr:.4f}")
print(f"  R² for σ²:  {r2_sigma2_svr:.4f}")

# Generate R² vs C and R² vs ε Plots
print("\nGenerating R² sensitivity plots for SVR...")
cv = KFold(n_splits=5, shuffle=True, random_state=42)

C_values = np.logspace(-3, 3, 20)
eps_values = np.logspace(-3, 3, 20)
C_fixed = 2048
eps_fixed = 4.0

r2_d50_vsC, r2_sigma2_vsC = [], []
r2_d50_vseps, r2_sigma2_vseps = [], []

# Scale the top-4 features manually for this cross-validation sweep
scaler_top4 = StandardScaler()
Xtr_d50_scaled = scaler_top4.fit_transform(Xtr_d50_top4)
Xtr_s2_scaled = scaler_top4.fit_transform(Xtr_s2_top4)

for C in C_values:
    svr_tmp_d50 = SVR(kernel='rbf', C=C, epsilon=eps_fixed, gamma='scale')
    svr_tmp_s2 = SVR(kernel='rbf', C=C, epsilon=eps_fixed, gamma='scale')

    r2_d50 = cross_val_score(svr_tmp_d50, Xtr_d50_scaled, y_train_d50, cv=5, scoring='r2').mean()
    r2_sigma2 = cross_val_score(svr_tmp_s2, Xtr_s2_scaled, y_train_sigma2, cv=5, scoring='r2').mean()

    r2_d50_vsC.append(r2_d50)
    r2_sigma2_vsC.append(r2_sigma2)

for eps in eps_values:
    svr_tmp_d50 = SVR(kernel='rbf', C=C_fixed, epsilon=eps, gamma='scale')
    svr_tmp_s2 = SVR(kernel='rbf', C=C_fixed, epsilon=eps, gamma='scale')

    r2_d50 = cross_val_score(svr_tmp_d50, Xtr_d50_scaled, y_train_d50, cv=5, scoring='r2').mean()
    r2_sigma2 = cross_val_score(svr_tmp_s2, Xtr_s2_scaled, y_train_sigma2, cv=5, scoring='r2').mean()

    r2_d50_vseps.append(r2_d50)
    r2_sigma2_vseps.append(r2_sigma2)

plt.figure(figsize=(12, 4.5))

plt.subplot(1, 2, 1)
plt.plot(C_values, r2_d50_vsC, 'k-', label=r'$d_{50}$')
plt.plot(C_values, r2_sigma2_vsC, 'r--', label=r'$\sigma^2$')
plt.xscale('log')
plt.xlabel("C")
plt.ylabel(r"$R^2$")
plt.title("(a) R² vs C")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(eps_values, r2_d50_vseps, 'k-', label=r'$d_{50}$')
plt.plot(eps_values, r2_sigma2_vseps, 'r--', label=r'$\sigma^2$')
plt.xscale('log')
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$R^2$")
plt.title("(b) R² vs ε")
plt.legend()

plt.tight_layout()
plt.savefig("Results/svr_r2_vs_C_eps.png", dpi=150)
plt.close()

# ============================================================
# RESULTS SUMMARY
# ============================================================
print("\n--- Summary ---")

results_summary = pd.DataFrame({
    'Model': ['Random Forest', 'SVR (Top-4 Features)'],
    'R² (d50)': [r2_d50_rf, r2_d50_svr],
    'R² (σ²)': [r2_sigma2_rf, r2_sigma2_svr]
})
print(results_summary.to_string(index=False))

# Predicted vs Actual Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(y_test_d50, y_pred_d50_rf, c='b', alpha=0.6, label='RF')
axes[0].scatter(y_test_d50, y_pred_d50_svr, c='r', alpha=0.6, label='SVR')
axes[0].plot([y_test_d50.min(), y_test_d50.max()],
             [y_test_d50.min(), y_test_d50.max()], 'k--', lw=1)
axes[0].set_title("Predicted vs Actual (d50)")
axes[0].set_xlabel("Actual d50")
axes[0].set_ylabel("Predicted d50")
axes[0].legend()

axes[1].scatter(y_test_sigma2, y_pred_sigma2_rf, c='b', alpha=0.6, label='RF')
axes[1].scatter(y_test_sigma2, y_pred_sigma2_svr, c='r', alpha=0.6, label='SVR')
axes[1].plot([y_test_sigma2.min(), y_test_sigma2.max()],
             [y_test_sigma2.min(), y_test_sigma2.max()], 'k--', lw=1)
axes[1].set_title("Predicted vs Actual (σ²)")
axes[1].set_xlabel("Actual σ²")
axes[1].set_ylabel("Predicted σ²")
axes[1].legend()

plt.tight_layout()
plt.savefig("Results/predicted_vs_actual.png", dpi=150)
plt.close()

print("\nAll figures and summary saved to the 'Results' directory.")

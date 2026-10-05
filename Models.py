# -------------------------------
# 1️ Imports
# -------------------------------
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, KFold

sns.set(style="whitegrid", context="notebook")

# -------------------------------
# 2️ Load and preprocess dataset
# -------------------------------
data = pd.read_csv("/content/synthetic_data_improved.csv")
print(" Dataset loaded:", data.shape)

if 'time' in data.columns:
    data = data.drop(columns=['time'])
data = data.dropna()

# Remove strong outliers
data = data[data['d50'] < data['d50'].quantile(0.99)]
data = data[data['sigma2'] < data['sigma2'].quantile(0.99)]

features = ['S','ub','np','T','a676_a650','a450_a676','chl_a','u']
X = data[features]
y_d50 = data['d50']
y_sigma2 = data['sigma2']

# Train-test split
X_train, X_test, y_train_d50, y_test_d50 = train_test_split(X, y_d50, test_size=0.3, random_state=42)
X_train2, X_test2, y_train_sigma2, y_test_sigma2 = train_test_split(X, y_sigma2, test_size=0.3, random_state=42)

# Scaling for SVR
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train2_scaled = scaler.fit_transform(X_train2)
X_test2_scaled = scaler.transform(X_test2)

# ============================================================
#  3️ RANDOM FOREST: Training + OOB Curve
# ============================================================

print("\n Training Random Forest models...")
n_trees = [10, 20, 40, 60, 100, 200, 400]
oob_scores_d50, oob_scores_sigma2 = [], []

import os
os.makedirs('figures', exist_ok=True)

for n in n_trees:
    rf_tmp1 = RandomForestRegressor(n_estimators=n, oob_score=True, random_state=42, n_jobs=-1)
    rf_tmp1.fit(X_train, y_train_d50)
    oob_scores_d50.append(rf_tmp1.oob_score_)

    rf_tmp2 = RandomForestRegressor(n_estimators=n, oob_score=True, random_state=42, n_jobs=-1)
    rf_tmp2.fit(X_train2, y_train_sigma2)
    oob_scores_sigma2.append(rf_tmp2.oob_score_)

plt.figure(figsize=(6,4))
plt.plot(n_trees, oob_scores_d50, 'k-', label='d50')
plt.plot(n_trees, oob_scores_sigma2, 'r--', label='σ²')
plt.xlabel("Number of Trees")
plt.ylabel("OOB Score")
plt.title("OOB Score vs Number of Trees")
plt.legend()
plt.tight_layout()
plt.savefig("figures/oob_score_curve.png")
plt.show()

# Final tuned RF
rf_d50 = RandomForestRegressor(
    n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    random_state=42, oob_score=True, n_jobs=-1
)
rf_sigma2 = RandomForestRegressor(
    n_estimators=400, max_depth=None, min_samples_split=2, min_samples_leaf=1,
    random_state=42, oob_score=True, n_jobs=-1
)

rf_d50.fit(X_train, y_train_d50)
rf_sigma2.fit(X_train2, y_train_sigma2)

y_pred_d50_rf = rf_d50.predict(X_test)
y_pred_sigma2_rf = rf_sigma2.predict(X_test2)

r2_d50_rf = r2_score(y_test_d50, y_pred_d50_rf)
r2_sigma2_rf = r2_score(y_test_sigma2, y_pred_sigma2_rf)
print(f"RF R² (d50): {r2_d50_rf:.3f} | RF R² (σ²): {r2_sigma2_rf:.3f}")

# Feature importance
importances = pd.Series(rf_d50.feature_importances_, index=features).sort_values(ascending=False)
plt.figure(figsize=(7,4))
sns.barplot(x=importances.values, y=importances.index, palette='viridis')
plt.title("Feature Importance for d50 (Random Forest)")
plt.tight_layout()
plt.savefig("figures/feature_importance_d50.png")
plt.show()

# ============================================================
#  4️ SVR: Hyperparameter tuning (top-4 features only: S, ub, np, T)
# ============================================================
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

print("\n Training SVR models with grid search (top-4 features: S, ub, np, T)...")

top4 = ['S', 'ub', 'np', 'T']   # features chosen as in the paper

# Subset training/test sets
Xtr_d50 = X_train[top4]
Xte_d50 = X_test[top4]
Xtr_s2  = X_train2[top4]
Xte_s2  = X_test2[top4]

# SVR pipeline: scale -> SVR
svr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf'))
])

# Grid search over C and epsilon
param_grid = {
    'svr__C': [1, 10, 100, 500, 1000, 2000],
    'svr__epsilon': [0.1, 0.5, 1, 2]
}

# Grid search for d50
grid_d50 = GridSearchCV(svr_pipe, param_grid, scoring='r2', cv=5, n_jobs=-1)
grid_d50.fit(Xtr_d50, y_train_d50)
svr_d50 = grid_d50.best_estimator_

# Grid search for σ²
grid_sigma2 = GridSearchCV(svr_pipe, param_grid, scoring='r2', cv=5, n_jobs=-1)
grid_sigma2.fit(Xtr_s2, y_train_sigma2)
svr_sigma2 = grid_sigma2.best_estimator_

# Report best params
print("Best Params (d50, top-4):", grid_d50.best_params_)
print("Best Params (σ², top-4):", grid_sigma2.best_params_)

# Test-set predictions
y_pred_d50_svr = svr_d50.predict(Xte_d50)
y_pred_sigma2_svr = svr_sigma2.predict(Xte_s2)

# R² scores
r2_d50_svr = r2_score(y_test_d50, y_pred_d50_svr)
r2_sigma2_svr = r2_score(y_test_sigma2, y_pred_sigma2_svr)
print(f"SVR (top-4) R² (d50): {r2_d50_svr:.3f} | SVR (top-4) R² (σ²): {r2_sigma2_svr:.3f}")

# ============================================================
#  5️ SVR: R² vs C and vs ε (Figure 3 style)
# ============================================================



# Cross-validation setup
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Sweep values (log scale like the paper)
C_values = np.logspace(-3, 3, 20)      # 1e-3 to 1e3
eps_values = np.logspace(-3, 3, 20)    # 1e-3 to 1e3

# Fixed params from paper’s choice
C_fixed = 2048
eps_fixed = 4.0

# Store results
r2_d50_vsC, r2_sigma2_vsC = [], []
r2_d50_vseps, r2_sigma2_vseps = [], []

# ---------- Sweep C (fix epsilon)
for C in C_values:
    svr_d50 = SVR(kernel='rbf', C=C, epsilon=eps_fixed, gamma='scale')
    svr_sigma2 = SVR(kernel='rbf', C=C, epsilon=eps_fixed, gamma='scale')

    r2_d50 = cross_val_score(svr_d50, X_train_scaled, y_train_d50, cv=cv, scoring='r2').mean()
    r2_sigma2 = cross_val_score(svr_sigma2, X_train2_scaled, y_train_sigma2, cv=cv, scoring='r2').mean()

    r2_d50_vsC.append(r2_d50)
    r2_sigma2_vsC.append(r2_sigma2)

# ---------- Sweep epsilon (fix C)
for eps in eps_values:
    svr_d50 = SVR(kernel='rbf', C=C_fixed, epsilon=eps, gamma='scale')
    svr_sigma2 = SVR(kernel='rbf', C=C_fixed, epsilon=eps, gamma='scale')

    r2_d50 = cross_val_score(svr_d50, X_train_scaled, y_train_d50, cv=cv, scoring='r2').mean()
    r2_sigma2 = cross_val_score(svr_sigma2, X_train2_scaled, y_train_sigma2, cv=cv, scoring='r2').mean()

    r2_d50_vseps.append(r2_d50)
    r2_sigma2_vseps.append(r2_sigma2)

# ---------- Plot like the paper
plt.figure(figsize=(12, 4.5))

# (a) R² vs C
plt.subplot(1, 2, 1)
plt.plot(C_values, r2_d50_vsC, 'k-', label=r'$d_{50}$')
plt.plot(C_values, r2_sigma2_vsC, 'r--', label=r'$\sigma^2$')
plt.xscale('log')
plt.xlabel("C")
plt.ylabel(r"$R^2$")
plt.title("(a)")
plt.legend()

# (b) R² vs ε
plt.subplot(1, 2, 2)
plt.plot(eps_values, r2_d50_vseps, 'k-', label=r'$d_{50}$')
plt.plot(eps_values, r2_sigma2_vseps, 'r--', label=r'$\sigma^2$')
plt.xscale('log')
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$R^2$")
plt.title("(b)")

plt.tight_layout()
plt.savefig("figures/svr_r2_vs_C_eps.png", dpi=150)
plt.show()

# ============================================================
# 5️ Final Results + Visualization
# ============================================================
results = pd.DataFrame({
    'Model': ['Random Forest', 'SVR'],
    'R² (d50)': [r2_d50_rf, r2_d50_svr],
    'R² (σ²)': [r2_sigma2_rf, r2_sigma2_svr]
})
display(results)
results.to_csv("figures/final_results_summary.csv", index=False)

# Predicted vs Actual plots
fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].scatter(y_test_d50, y_pred_d50_rf, c='b', alpha=0.6, label='RF')
axes[0].scatter(y_test_d50, y_pred_d50_svr, c='r', alpha=0.6, label='SVR')
axes[0].plot([y_test_d50.min(), y_test_d50.max()],
             [y_test_d50.min(), y_test_d50.max()], 'k--', lw=1)
axes[0].set_title("Predicted vs Actual (d50)")
axes[0].set_xlabel("Actual")
axes[0].set_ylabel("Predicted")
axes[0].legend()

axes[1].scatter(y_test_sigma2, y_pred_sigma2_rf, c='b', alpha=0.6, label='RF')
axes[1].scatter(y_test_sigma2, y_pred_sigma2_svr, c='r', alpha=0.6, label='SVR')
axes[1].plot([y_test_sigma2.min(), y_test_sigma2.max()],
             [y_test_sigma2.min(), y_test_sigma2.max()], 'k--', lw=1)
axes[1].set_title("Predicted vs Actual (σ²)")
axes[1].set_xlabel("Actual")
axes[1].set_ylabel("Predicted")
axes[1].legend()
plt.tight_layout()
plt.savefig("figures/predicted_vs_actual.png")
plt.show()

print("\n All final figures and summary saved in /figures ")
print(results)

import seaborn as sns 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

inv = np.expm1  # Inverse log-transform for predictions

# Load dataset
tips = sns.load_dataset("tips")

# === Data Preprocessing ===
# Log-transform total_bill and tip to reduce skew
for col in ['total_bill', 'tip']:
    tips[f'log_{col}'] = np.log1p(tips[col])

# Feature engineering: tip percentage
tips['tip_percent'] = tips['tip'] / tips['total_bill']

# One-hot encode categorical variables
categorical_cols = ['sex', 'smoker', 'day', 'time']
tips_encoded = pd.get_dummies(tips, columns=categorical_cols, drop_first=False)

# Add engineered features
tips_encoded['log_tip'] = tips['log_tip']
tips_encoded['log_total_bill'] = tips['log_total_bill']
tips_encoded['tip_percent'] = tips['tip_percent']

# Feature columns
sex_cols = [col for col in tips_encoded.columns if col.startswith('sex_')]
smoker_cols = [col for col in tips_encoded.columns if col.startswith('smoker_')]
day_cols = [col for col in tips_encoded.columns if col.startswith('day_')]
time_cols = [col for col in tips_encoded.columns if col.startswith('time_')]

# Final feature list
features = ['log_tip', 'size', 'tip_percent'] 
X_base = tips_encoded[features]

# Polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_base)
X = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(features))

y = tips_encoded['log_total_bill']

# Train/val/test split (60/20/20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# === Correlation Matrix Heatmap ===
plt.figure(figsize=(10, 8))
correlation_matrix = tips_encoded.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# --- Experiment A: Original Features ---
# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Neural Network
nn = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=3000, early_stopping=True, random_state=42)
nn.fit(X_train_scaled, y_train)
nn_pred = nn.predict(X_test_scaled)

# --- Experiment B: PCA Features ---
pca = PCA(n_components=0.80)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Linear Regression (PCA)
lr_pca = LinearRegression()
lr_pca.fit(X_train_pca, y_train)
lr_pca_pred = lr_pca.predict(X_test_pca)

# Neural Network (PCA)
nn_pca = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=3000, early_stopping=True, random_state=42)
nn_pca.fit(X_train_pca, y_train)
nn_pca_pred = nn_pca.predict(X_test_pca)

# --- Evaluation ---
inv = np.expm1
y_true = inv(y_test)

results = {
    'Linear Regression (Original)': inv(lr_pred),
    'Neural Network (Original)': inv(nn_pred),
    'Linear Regression (PCA)': inv(lr_pca_pred),
    'Neural Network (PCA)': inv(nn_pca_pred)
}

print('\n=== Model Evaluation (MAE in dollars) ===')
for name, preds in results.items():
    mae = mean_absolute_error(y_true, preds)
    print(f'{name}: MAE = {mae:.2f}')

# Optionally, plot a bar chart for these four results
plt.figure(figsize=(8,5))
plt.bar(results.keys(), [mean_absolute_error(y_true, preds) for preds in results.values()], color=['#4c72b0', '#55a868', '#c44e52', '#8172b3'])
plt.ylabel('MAE (in dollars)')
plt.title('Model Comparison: MAE on Test Set')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('model_comparison_mae_linear_nn_pca.png', dpi=150)
plt.show()

# === Additional Visualizations for White Paper ===

# 1. Model Comparison Bar Chart (MAE)
mae_scores = [
    mean_absolute_error(y_true, inv(lr_pred)),
    mean_absolute_error(y_true, inv(nn_pred)),
    mean_absolute_error(y_true, inv(lr_pca_pred)),
    mean_absolute_error(y_true, inv(nn_pca_pred)),
]
model_names = ['Linear Regression (Original)', 'Neural Network (Original)', 'Linear Regression (PCA)', 'Neural Network (PCA)']
plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=mae_scores, palette='viridis')
plt.ylabel('MAE (in dollars)', fontsize=12)
plt.title('Model Comparison: MAE on Test Set', fontsize=14)
for i, v in enumerate(mae_scores):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=10)
plt.legend(['MAE'], fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig('model_comparison_mae_linear_nn_pca.png', dpi=150)
plt.close()

# 2. Predicted vs Actual Scatter Plots for Each Model
plt.figure(figsize=(16, 10))
for i, (name, preds) in enumerate(results.items(), 1):
    plt.subplot(2, 2, i)
    plt.scatter(y_true, inv(preds), alpha=0.6, color='#1f77b4', edgecolor='k', s=60, label='Predicted')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Ideal')
    plt.xlabel('Actual Total Bill', fontsize=11)
    plt.ylabel('Predicted Total Bill', fontsize=11)
    plt.title(f'{name}: Predicted vs Actual', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('predicted_vs_actual_all_models.png', dpi=150)
plt.close()

# 3. Pairplot for relevant features (hue by party size)
pairplot_data = tips[['total_bill', 'tip', 'size', 'tip_percent']]
pairplot_data['size'] = pairplot_data['size'].astype(str)  # treat as categorical for hue
g = sns.pairplot(pairplot_data, hue='size', palette='Set1', diag_kind='kde', plot_kws={'alpha':0.6})
g.fig.suptitle('Pairplot of Key Features Colored by Party Size', y=1.04, fontsize=16)
g._legend.set_title('Party Size')
g._legend.set_bbox_to_anchor((1.05, 1))
g._legend.set_loc('upper left')
g.savefig('pairplot_key_features.png', dpi=150, bbox_inches='tight')
plt.close()

# 4. Model Comparison Bar Chart (MAE) with PCA models
all_mae_scores = mae_scores
all_model_names = model_names

plt.figure(figsize=(10, 5))
sns.barplot(x=all_model_names, y=all_mae_scores, palette='viridis')
plt.ylabel('MAE (in dollars)', fontsize=12)
plt.title('Model Comparison: MAE on Test Set (Original & PCA)', fontsize=14)
for i, v in enumerate(all_mae_scores):
    plt.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=10)
plt.legend(['MAE'], fontsize=10, loc='upper right')
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig('model_comparison_mae_with_pca.png', dpi=150)
plt.close()

# 5. Predicted vs Actual for PCA models
plt.figure(figsize=(12, 5))
for i, (name, preds) in enumerate({'Linear Regression (PCA)': lr_pca_pred, 'Neural Network (PCA)': nn_pca_pred}.items(), 1):
    plt.subplot(1, 2, i)
    plt.scatter(y_true, inv(preds), alpha=0.6, color='#2ca02c', edgecolor='k', s=60, label='Predicted')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Ideal')
    plt.xlabel('Actual Total Bill', fontsize=11)
    plt.ylabel('Predicted Total Bill', fontsize=11)
    plt.title(f'{name}: Predicted vs Actual', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('predicted_vs_actual_pca_models.png', dpi=150)
plt.close()

# 6. Actual vs Predicted for Both Models (Original Features Only)
plt.figure(figsize=(8, 6))
plt.scatter(y_true, inv(lr_pred), alpha=0.6, color='#4c72b0', edgecolor='k', s=60, label='Linear Regression Predicted')
plt.scatter(y_true, inv(nn_pred), alpha=0.6, color='#55a868', edgecolor='k', s=60, label='Neural Network Predicted')
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Ideal')
plt.xlabel('Actual Total Bill', fontsize=12)
plt.ylabel('Predicted Total Bill', fontsize=12)
plt.title('Actual vs Predicted: Linear Regression & Neural Network (Original Features)', fontsize=13)
plt.legend(fontsize=10, loc='upper left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('actual_vs_predicted_lr_nn.png', dpi=150)
plt.close()
                
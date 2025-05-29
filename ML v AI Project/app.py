import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os

# Create graphs directory at the top
graphs_dir = './graphs'
os.makedirs(graphs_dir, exist_ok=True)

# === 1. Import, preprocess, clean, and EDA ===
sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")

tips = tips[tips['total_bill'] > 0].copy()

# Add day and time as numeric codes for correlation analysis
# day_code: Thur=0, Fri=1, Sat=2, Sun=3
# time_code: Lunch=0, Dinner=1
tips['day_code'] = tips['day'].astype('category').cat.codes
tips['time_code'] = tips['time'].astype('category').cat.codes

# Only keep relevant features for modeling
feature_cols = ['total_bill', 'size', 'day_code', 'time_code']
X = tips[feature_cols]
y = tips['tip']

# EDA: Correlation matrix heatmap
numeric_tips = tips[feature_cols + ['tip']]
plt.figure(figsize=(10, 8))
correlation_matrix = numeric_tips.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.gcf().text(1.02, 0.5, 'day_code: Thur=0, Fri=1, Sat=2, Sun=3\n'
                          'time_code: Lunch=0, Dinner=1', 
               fontsize=10, va='center', ha='left', transform=plt.gca().transAxes)
plt.tight_layout()
plt.savefig(f'{graphs_dir}/correlation_matrix.png')
plt.close()

# === 2. Modeling Experiments ===
# Features for Experiment A (original)
feature_cols = ['total_bill', 'size'] + [col for col in tips.columns if col.startswith('day_') or col.startswith('time_')]
X = tips[feature_cols] 
y = tips['tip']

# Train/val/test split (60/20/20)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# --- Experiment A: Original Features ---
# Linear Regression
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)

# Neural Network
nn = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=2000, random_state=1, early_stopping=True)
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
nn_pca = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=2000, random_state=2, early_stopping=True)
nn_pca.fit(X_train_pca, y_train)
nn_pca_pred = nn_pca.predict(X_test_pca)

# --- 4. Evaluation ---
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

results = {
    'Linear Regression (Original)': lr_pred,
    'Neural Network (Original)': nn_pred,
    'Linear Regression (PCA)': lr_pca_pred,
    'Neural Network (PCA)': nn_pca_pred
}

print('\n=== Model Evaluation (Test Set) ===')
mae_scores = []
for name, preds in results.items():
    mse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    mape = mean_absolute_percentage_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mae_scores.append(mae)
    print(f'{name}:')
    print(f'  MSE  = {mse:.3f}')
    print(f'  MAE  = {mae:.3f}')
    print(f'  MAPE = {mape:.2f}%')
    print(f'  R^2  = {r2:.3f}')

# 5. Model Comparison Bar Chart (MAE)
model_names = list(results.keys())
plt.figure(figsize=(8, 5))
sns.barplot(x=model_names, y=mae_scores, palette='viridis')
plt.ylabel('MAE (in dollars)', fontsize=12)
plt.title('Model Comparison: MAE on Test Set', fontsize=14)
for i, v in enumerate(mae_scores):
    plt.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=10)
plt.legend(['MAE'], fontsize=10, loc='upper right')
plt.tight_layout()
plt.savefig(f'{graphs_dir}/model_comparison_mae.png', dpi=150)
plt.close()

# 6. Predicted vs Actual Scatter Plots for Each Model
plt.figure(figsize=(12, 8))
for i, (name, preds) in enumerate(results.items(), 1):
    plt.subplot(2, 2, i)
    plt.scatter(y_test, preds, alpha=0.6, color='#1f77b4', edgecolor='k', s=60, label='Predicted')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
    plt.xlabel('Actual Tip', fontsize=11)
    plt.ylabel('Predicted Tip', fontsize=11)
    plt.title(f'{name}: Predicted vs Actual', fontsize=12)
    plt.legend(fontsize=10, loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{graphs_dir}/predicted_vs_actual_all_models.png', dpi=150)
plt.close()

# 7. Pairplot for relevant features (hue by party size)
pairplot_data = tips[['total_bill', 'tip', 'size']].copy()
pairplot_data['size'] = pairplot_data['size'].astype(str)  # treat as categorical for hue
g = sns.pairplot(pairplot_data, hue='size', palette='Set1', diag_kind='kde', plot_kws={'alpha':0.6})
g.fig.suptitle('Pairplot of Key Features Colored by Party Size', y=1.04, fontsize=16)
g._legend.set_title('Party Size')
g._legend.set_bbox_to_anchor((1.05, 1))
g._legend.set_loc('upper left')
g.savefig(f'{graphs_dir}/pairplot_key_features.png', dpi=150, bbox_inches='tight')
plt.close()

# 8. PCA Scree Plot
plt.figure(figsize=(7,4))
plt.plot(np.arange(1, len(pca.explained_variance_ratio_)+1), np.cumsum(pca.explained_variance_ratio_), marker='o')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Scree Plot')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig(f'{graphs_dir}/pca_scree_plot.png', dpi=150)
plt.close()

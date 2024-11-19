import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Load the Data
data = pd.read_csv('dataset.csv')

# Convert date columns to datetime
data['subscription_start'] = pd.to_datetime(data['subscription_start'])
data['subscription_expiration'] = pd.to_datetime(data['subscription_expiration'])
data['last_activity'] = pd.to_datetime(data['last_activity'])

# Create time-based features
data['subscription_duration'] = (data['subscription_expiration'] - data['subscription_start']).dt.days
data['days_since_last_activity'] = (pd.to_datetime('today') - data['last_activity']).dt.days
data['days_until_expiration'] = (data['subscription_expiration'] - pd.to_datetime('today')).dt.days

# Data Exploration
print(data.info())
print(data.describe())

# Create output directory if it doesn't exist
os.makedirs('output', exist_ok=True)

# Visualize churn distribution
sns.countplot(x='churned', data=data)
plt.title("Churn Distribution")
plt.savefig('output/churn_distribution.png')
plt.clf()

# Correlation heatmap for numeric features
corr = data.corr()
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig('output/correlation_heatmap.png')
plt.clf()

# Feature Engineering
data['avg_session_duration'] = data['total_session_time'] / data['num_sessions']
data['interaction_frequency'] = data['num_support_interactions'] / (data['subscription_length_days'] + 1)
data['recent_activity'] = (pd.to_datetime(data['subscription_expiration']) - pd.to_datetime(data['last_activity'])).dt.days

# Fill missing values for new features
data.fillna(0, inplace=True)

# Drop irrelevant columns
data.drop(['customer_id', 'subscription_start', 'subscription_expiration', 'last_activity'], axis=1, inplace=True)

# Handle Imbalanced Dataset
X = data.drop('churned', axis=1)
y = data['churned']

# Balance the dataset using SMOTE
smote = SMOTE(random_state=42, k_neighbors=3)  # Adjust k_neighbors to a value <= number of samples in the minority class
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Models

# Logistic Regression
log_model = LogisticRegression(random_state=42)
log_model.fit(X_train, y_train)
log_preds = log_model.predict(X_test)

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# XGBoost
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Evaluate Models
def evaluate_model(model_name, y_true, y_pred):
  print(f"--- {model_name} ---")
  print(classification_report(y_true, y_pred))
  cm = confusion_matrix(y_true, y_pred)
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
  plt.title(f"Confusion Matrix - {model_name}")
  plt.savefig(f'output/confusion_matrix_{model_name.replace(" ", "_").lower()}.png')
  plt.clf()

evaluate_model("Logistic Regression", y_test, log_preds)
evaluate_model("Random Forest", y_test, rf_preds)
evaluate_model("XGBoost", y_test, xgb_preds)

# Hyperparameter Tuning (Random Forest)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=3, scoring='f1', verbose=2)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
best_rf_model = grid_search.best_estimator_

# Re-evaluate best model
best_rf_preds = best_rf_model.predict(X_test)
evaluate_model("Tuned Random Forest", y_test, best_rf_preds)

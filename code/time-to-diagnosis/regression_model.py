import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, cohen_kappa_score, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import statsmodels.api as sm
import shap
from sklearn.inspection import PartialDependenceDisplay
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

file_name = '../data/data_time_to_diagnosis/new_output.csv'
df = pd.read_csv(file_name)

# remove Other Pacific Islander/American Indian or Alaska Native/Asian Indian
df['race'] = df['race'].replace('Other Pacific Islander', 'No matching concept')
df['race'] = df['race'].replace('American Indian or Alaska Native', 'No matching concept')
df['race'] = df['race'].replace('Asian Indian', 'No matching concept')

# imputation
df['income'] = df['income'].fillna(72284)
df['education'] = df['education'].fillna(0.339)
insurance_mean = df['insurance'].mean()
df['insurance'] = df['insurance'].fillna(insurance_mean)
pm25_mean = df['PM2.5_mean'].mean()
df['PM2.5_mean'] = df['PM2.5_mean'].fillna(pm25_mean)

# features and labels
X = df[['race', 'gender', 'diagnose_age', 'income', 'education', 'insurance', 'PM2.5_mean']]
y = 1-df['longer_diagnosis_flag']

# Step 1: Preprocess the data
# Encode categorical variables using one-hot encoding
X_encoded = pd.get_dummies(X, columns=['race', 'gender'])
# Standardize continuous variables
scaler = StandardScaler()
X_encoded[['diagnose_age', 'income', 'education', 'insurance', 'PM2.5_mean']] = scaler.fit_transform(X_encoded[['diagnose_age', 'income', 'education','insurance', 'PM2.5_mean']])

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)

# class weight adjustment
log_reg = LogisticRegression(random_state=0, max_iter=1000, class_weight='balanced').fit(X_train, y_train)

# # resampling oversampling the minority class
# smote = SMOTE()
# X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
# # resampling undersampling the majority class
# undersampler = RandomUnderSampler()
# X_resampled, y_resampled = undersampler.fit_resample(X_train, y_train)
# log_reg = LogisticRegression(random_state=0, max_iter=1000).fit(X_resampled, y_resampled)

y_prob = log_reg.predict_proba(X_test)[:, 1]  # Probability for class 1
optimal_threshold = 0.5  # Example of using a threshold lower than 0.5
y_pred = (y_prob >= optimal_threshold).astype(int)

# Step 5: Compute model performance metrics
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
f1 = f1_score(y_test, y_pred)
average_precision = average_precision_score(y_test, y_prob)
kappa = cohen_kappa_score(y_test, y_pred)

# Step 6: Compute the odds ratios
coefficients = log_reg.coef_[0]
odds_ratios = np.exp(coefficients)

# Create a dataframe for the coefficients and odds ratios
feature_names = X_encoded.columns
odds_ratio_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Odds Ratio': odds_ratios
})


# Print out the performance metrics
print(f"Precision-Recall AUC: {pr_auc}")
print(f"F1 Score: {f1}")
print(f"Average Precision: {average_precision}")
print(f"Cohen's Kappa: {kappa}")

print(odds_ratio_df)
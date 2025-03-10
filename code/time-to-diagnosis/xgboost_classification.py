import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, cohen_kappa_score, auc
from xgboost import XGBClassifier
from sklearn.utils import resample
import statsmodels.api as sm
import shap
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from xgboost import plot_importance


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
X = df[['race','gender','diagnose_age','income','education','insurance','PM2.5_mean']]
df['shorter_diagnosis_flag'] = 1-df['longer_diagnosis_flag']
y = df['shorter_diagnosis_flag']

# Step 1: Preprocess the data
# Encode categorical variables using one-hot encoding
X_encoded = pd.get_dummies(X, columns=['race', 'gender'])
X_encoded_original = X_encoded.copy()
# Standardize continuous variables
scaler = StandardScaler()
X_encoded[['diagnose_age', 'income', 'education','insurance', 'PM2.5_mean']] = scaler.fit_transform(X_encoded[['diagnose_age', 'income', 'education','insurance', 'PM2.5_mean']])

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, stratify=y, random_state=42)

# Calculate scale_pos_weight
pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
# XGBoost classifier
xgb_model = XGBClassifier(use_label_encoder=False, n_estimators=50, scale_pos_weight=pos_weight, eval_metric='aucpr')
# xgb_model = XGBClassifier(use_label_encoder=False, n_estimators=1000, eval_metric='aucpr')
xgb_model.fit(X_train, y_train)

# Predict on test set
y_prob = xgb_model.predict_proba(X_test)[:, 1]
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

# Step 5: Compute model performance metrics
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
pr_auc = auc(recall, precision)
f1 = f1_score(y_test, y_pred)
average_precision = average_precision_score(y_test, y_prob)
kappa = cohen_kappa_score(y_test, y_pred)

# Print out the performance metrics
print(f"Precision-Recall AUC: {pr_auc}")
print(f"F1 Score: {f1}")
print(f"Average Precision: {average_precision}")
print(f"Cohen's Kappa: {kappa}")

# # partial_dependence plot
# features = ['diagnose_age']
# fig, ax = plt.subplots(figsize=(8, 6))
# display = PartialDependenceDisplay.from_estimator(xgb_model, X_train, features, ax=ax)
# plt.title("Partial Dependence of diagnose_age")
# plt.show()

# # feature importance from XGboost
# plot_importance(xgb_model, importance_type='weight')
# plt.show()

def bootstrap_risk_ratio(X_test, group_A_indices, group_base_indices, model, n_bootstrap=1000):
    risk_ratios = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        group_A_resample = resample(group_A_indices)
        group_base_resample = resample(group_base_indices)

        # # without resampling, n_bootstrap=1
        # group_A_resample = group_A_indices
        # group_base_resample = group_base_indices

        # Get predictions for resampled data
        group_A_probs_boot = model.predict_proba(X_test.loc[group_A_resample])[:, 1]
        group_base_probs_boot = model.predict_proba(X_test.loc[group_base_resample])[:, 1]

        # Compute risks for the bootstrap sample
        risk_A_boot = np.mean(group_A_probs_boot)
        risk_base_boot = np.mean(group_base_probs_boot)

        # Calculate the risk ratio for the bootstrap sample
        risk_ratio_boot = risk_A_boot / risk_base_boot
        risk_ratios.append(risk_ratio_boot)

    # Return the risk ratios
    return np.percentile(risk_ratios, [2.5, 97.5]), np.mean(risk_ratios)

# adjusted risk ratio
# age
X_test_original = X_encoded_original.loc[X_test.index]
X_test_original['less_then_55'] = X_test_original['diagnose_age'] < 55
X_test_original['55_65'] = (X_test_original['diagnose_age'] >= 55) & (X_test_original['diagnose_age'] < 65)
X_test_original['65_75'] = (X_test_original['diagnose_age'] >= 65) & (X_test_original['diagnose_age'] < 75)
X_test_original['75_85'] = (X_test_original['diagnose_age'] >= 75) & (X_test_original['diagnose_age'] < 85)
X_test_original['older_than_85'] = X_test_original['diagnose_age'] >=85
group_A = X_test_original[X_test_original['less_then_55']==True]
group_B = X_test_original[X_test_original['55_65']==True]
group_C = X_test_original[X_test_original['65_75']==True]
group_D = X_test_original[X_test_original['75_85']==True]
group_E = X_test_original[X_test_original['older_than_85']==True]
print('diagnose_age Adjusted Risk Ratio:')
print('65_75 is base')
bound, mean = bootstrap_risk_ratio(X_test, group_A.index, group_C.index, xgb_model)
print(f"less_then_55, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_B.index, group_C.index, xgb_model)
print(f"55_65, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_C.index, group_C.index, xgb_model)
print(f"65_75, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_D.index, group_C.index, xgb_model)
print(f"75_85, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_E.index, group_C.index, xgb_model)
print(f"older_than_85, mean: {mean}, {bound[0], bound[1]}")
print('################################################')


# gender
group_A = X_test[X_test['gender_FEMALE']==True]
group_B = X_test[X_test['gender_MALE']==True]
print('gender Adjusted Risk Ratio:')
print('female is base')
bound, mean = bootstrap_risk_ratio(X_test, group_B.index, group_A.index, xgb_model)
print(f"male, mean: {mean}, {bound[0], bound[1]}")
print('################################################')

# race
group_A = X_test[X_test['race_White']==True]
group_B = X_test[X_test['race_Hispanic or Latino']==True]
group_C = X_test[X_test['race_Black or African American']==True]
group_D = X_test[X_test['race_Asian']==True]
group_E = X_test[X_test['race_No matching concept']==True]
print('race Adjusted Risk Ratio:')
print('Black or African American is base')
bound, mean = bootstrap_risk_ratio(X_test, group_A.index, group_C.index, xgb_model)
print(f"White, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_B.index, group_C.index, xgb_model)
print(f"Hispanic or Latino, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_C.index, group_C.index, xgb_model)
print(f"Black or African American, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_D.index, group_C.index, xgb_model)
print(f"Asian, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_E.index, group_C.index, xgb_model)
print(f"No matching concept, mean: {mean}, {bound[0], bound[1]}")
print('################################################')

# income
X_test_original = X_encoded_original.loc[X_test.index]
X_test_original['low_class'] = X_test_original['income'] < 30000
X_test_original['low_middle_class'] = (X_test_original['income'] >= 30000) & (X_test_original['income'] <= 58020)
X_test_original['middle_class'] = (X_test_original['income'] > 58020) & (X_test_original['income'] <= 94000)
X_test_original['upper_middle_class'] = (X_test_original['income'] > 94000) & (X_test_original['income'] <= 153000)
X_test_original['upper_class'] = X_test_original['income'] > 153000
group_A = X_test_original[X_test_original['low_class']==True]
group_B = X_test_original[X_test_original['low_middle_class']==True]
group_C = X_test_original[X_test_original['middle_class']==True]
group_D = X_test_original[X_test_original['upper_middle_class']==True]
group_E = X_test_original[X_test_original['upper_class']==True]
print('income Adjusted Risk Ratio:')
print('middle_class is base')
bound, mean = bootstrap_risk_ratio(X_test, group_A.index, group_C.index, xgb_model)
print(f"low_class, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_B.index, group_C.index, xgb_model)
print(f"lower_middle_class, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_C.index, group_C.index, xgb_model)
print(f"middle_class, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_D.index, group_C.index, xgb_model)
print(f"upper_middle_class, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_E.index, group_C.index, xgb_model)
print(f"upper_class, mean: {mean}, {bound[0], bound[1]}")
print('################################################')

# education
# two class
education_splits, split_points = pd.qcut(X_encoded_original['education'], q=2, retbins=True, labels=["Low", "High"])
X_test_original = X_encoded_original.loc[X_test.index]
X_test_original['low_education_class'] = X_test_original['education'] < split_points[1]
X_test_original['high_education_class'] = X_test_original['education'] >= split_points[1]
group_A = X_test_original[X_test_original['low_education_class']==True]
group_C = X_test_original[X_test_original['high_education_class']==True]
print(f'education: {split_points}')
print('education Adjusted Risk Ratio:')
print('low_education_class is base')
bound, mean = bootstrap_risk_ratio(X_test, group_A.index, group_A.index, xgb_model)
print(f"low_education_class, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_C.index, group_A.index, xgb_model)
print(f"high_education_class, mean: {mean}, {bound[0], bound[1]}")
print('################################################')


# insurance
# two class
insurance_splits, split_points = pd.qcut(X_encoded_original['insurance'], q=2, retbins=True, labels=["Low", "High"])
X_test_original = X_encoded_original.loc[X_test.index]
X_test_original['low_insurance_class'] = X_test_original['insurance'] < split_points[1]
X_test_original['high_insurance_class'] = X_test_original['insurance'] >= split_points[1]
group_A = X_test_original[X_test_original['low_insurance_class']==True]
group_C = X_test_original[X_test_original['high_insurance_class']==True]
print(f'insurance: {split_points}')
print('insurance Adjusted Risk Ratio:')
print('low_insurance_class is base')
bound, mean = bootstrap_risk_ratio(X_test, group_A.index, group_A.index, xgb_model)
print(f"low_insurance_class, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_C.index, group_A.index, xgb_model)
print(f"high_insurance_class, mean: {mean}, {bound[0], bound[1]}")
print('################################################')

# pm2.5
# two class
education_splits, split_points = pd.qcut(X_encoded_original['PM2.5_mean'], q=2, retbins=True, labels=["Low", "High"])
X_test_original = X_encoded_original.loc[X_test.index]
X_test_original['low_PM2.5_mean_class'] = X_test_original['PM2.5_mean'] < split_points[1]
X_test_original['high_PM2.5_mean_class'] = X_test_original['PM2.5_mean'] >= split_points[1]
group_A = X_test_original[X_test_original['low_PM2.5_mean_class']==True]
group_C = X_test_original[X_test_original['high_PM2.5_mean_class']==True]
print(f'PM2.5_mean: {split_points}')
print('PM2.5_mean Adjusted Risk Ratio:')
print('low_PM2.5_mean_class is base')
bound, mean = bootstrap_risk_ratio(X_test, group_A.index, group_A.index, xgb_model)
print(f"low_PM2.5_mean_class, mean: {mean}, {bound[0], bound[1]}")
bound, mean = bootstrap_risk_ratio(X_test, group_C.index, group_A.index, xgb_model)
print(f"high_PM2.5_mean_class, mean: {mean}, {bound[0], bound[1]}")
print('################################################')

# three class
# insurance_splits, split_points = pd.qcut(X_encoded_original['insurance'], q=3, retbins=True, labels=["Low", "Medium", "High"])
# X_test_original = X_encoded_original.loc[X_test.index]
# X_test_original['low_insurance_class'] = X_test_original['insurance'] < split_points[1]
# X_test_original['middle_insurance_class'] = (X_test_original['insurance'] >= split_points[1]) & (X_test_original['insurance'] < split_points[2])
# X_test_original['high_insurance_class'] = X_test_original['insurance'] >= split_points[1]
# group_A = X_test_original[X_test_original['low_insurance_class']==True]
# group_B = X_test_original[X_test_original['middle_insurance_class']==True]
# group_C = X_test_original[X_test_original['high_insurance_class']==True]
# print('insurance Adjusted Risk Ratio:')
# print('low_insurance_class is base')
# bound, mean = bootstrap_risk_ratio(X_test, group_A.index, group_B.index, xgb_model)
# print(f"low_insurance_class, mean: {mean}, {bound[0], bound[1]}")
# bound, mean = bootstrap_risk_ratio(X_test, group_B.index, group_B.index, xgb_model)
# print(f"middle_insurance_class, mean: {mean}, {bound[0], bound[1]}")
# bound, mean = bootstrap_risk_ratio(X_test, group_C.index, group_B.index, xgb_model)
# print(f"high_insurance_class, mean: {mean}, {bound[0], bound[1]}")









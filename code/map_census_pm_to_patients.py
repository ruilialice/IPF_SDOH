# this code is used to map the information in ipf_final with the
# income, education and insurance info from census
# the mapping is based on the age and race
# if the income value <0, such as '-666666666', it indicates that there is a missing
# if the education percentage level is 0 or 1, which indicates that there are very few people in this race in this area
# then use the percentage in the general population as the final result
# if the insurance percentage level is 0 or 1, which indicates that there are very few people in this race in this area
# then use the percentage in the general population as the final result

import pandas as pd
import math

data_file = '../data/data_time_to_diagnosis/ipf_patient_info.csv'
header_list = ['person_id', 'diagnosis_dt', 'occurrence_provider_id', 'visit_occurrence_id', 'race', 'gender', 'birth_dt',
               'diagnose_age', 'patient_location_zip', 'treatment_dt', 'antifibrotic_flag', 'symptom_dt', 'interval',
               'abnormal_flag', 'longer_diagnosis_flag']
df = pd.read_csv(data_file, names=header_list, dtype=str)
df['income'] = None
df['education'] = None
df['insurance'] = None

census_df = pd.read_csv('../data/data_time_to_diagnosis/distinct_zip_income_education_insurance.csv', dtype=str)
census_df.columns = census_df.columns.str.lower()

for idx, row in df.iterrows():
    gender = row['gender'].lower()
    race = row['race'].lower()
    age = math.floor(float(row['diagnose_age']))
    zip_code = str(row['patient_location_zip'])

    potential_rows = census_df.loc[census_df['distinct_zip']==zip_code]
    if potential_rows.shape[0]>1:
        print('multiple result with zip code {}'.format(zip_code))
    elif potential_rows.shape[0]==0:
        continue

    if race in ['white', 'hispanic or latino', 'black or african american', 'asian']:
        potential_column_dict = {'income': ['median household income ('+race+')', 'median household income'],
                                 'education': ['education_percentage_'+gender+'('+race+')', 'education_percentage_'+gender+'(general)'],
                                 'insurance': ['insurance_percentage, '+gender+' 54' if age<55
                                               else 'insurance_percentage, '+gender+' 64' if age<65
                                                else 'insurance_percentage, '+gender+' 74' if age<75
                                                else 'insurance_percentage, '+gender+' over 75']}

        for column in potential_column_dict:
            temp_result = -1
            flag = 0
            for temp_column in potential_column_dict[column]:
                temp_data = float(potential_rows[temp_column].values[0])
                if temp_data>0:
                    temp_result = temp_data
                    flag = 1
                    break
            if flag == 1:
                df.at[idx, column] = temp_result

    else:
        potential_column_dict = {'income': ['median household income'],
                                 'education': ['education_percentage_' + gender + '(general)'],
                                 'insurance': ['insurance_percentage, '+gender+' 54' if age<55
                                               else 'insurance_percentage, '+gender+' 64' if age<65
                                                else 'insurance_percentage, '+gender+' 74' if age<75
                                                else 'insurance_percentage, '+gender+' over 75']}
        for column in potential_column_dict:
            temp_result = -1
            flag = 0
            for temp_column in potential_column_dict[column]:
                temp_data = float(potential_rows[temp_column].values[0])
                if temp_data>0:
                    temp_result = temp_data
                    flag = 1
                    break
            if flag == 1:
                df.at[idx, column] = temp_result

# read pm25 info
pm25_file = '../data/zip_to_county/pm25.csv'
pm25_df = pd.read_csv(pm25_file, dtype=str)
pm25_df = pm25_df[['distinct_zip', 'PM2.5_mean', 'PM2.5_dev']]
df = pd.merge(df, pm25_df, how='left', left_on='patient_location_zip', right_on='distinct_zip')
df = df.drop(columns=['distinct_zip'])

df.to_csv('../data/data_time_to_diagnosis/new_output.csv', index=False)
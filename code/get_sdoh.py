import pandas as pd
import requests

data_file = '../data/data_treatment/ipf_final.csv'
header_list = ['person_id', 'diagnosis_dt', 'occurrence_provider_id', 'visit_occurrence_id', 'race', 'gender', 'birth_dt',
               'diagnose_age', 'patient_location_zip', 'treatment_dt', 'antifibrotic_flag', 'symptom_dt', 'interval', 'abnormal_flag']
df = pd.read_csv(data_file, names=header_list, dtype=str)
distinct_zip = df['patient_location_zip'].dropna().unique()

# Create a new DataFrame with the distinct ZIP codes
distinct_zip_df = pd.DataFrame(distinct_zip, columns=['distinct_zip'])

# the following code is used to get the avg income/avg education/avg pollution
census_api_key = 'xxxxxxxxxxxxxxxxxx'
census_base_url = 'https://api.census.gov/data/2022/acs/acs5'
previous_census_base_url = 'https://api.census.gov/data/2021/acs/acs5'
# B19013_001E Median Household Income in the Past 12 Months (in 2022 Inflation-Adjusted Dollars)
# B19013A_001E Median Household Income in the Past 12 Months (in 2022 Inflation-Adjusted Dollars) (White Alone Householder)
# B19013B_001E Median Household Income in the Past 12 Months (in 2022 Inflation-Adjusted Dollars) (Black or African American Alone Householder)
# B19013C_001E Median Household Income in the Past 12 Months (in 2022 Inflation-Adjusted Dollars) (American Indian and Alaska Native Alone Householder)
# B19013D_001E Median Household Income in the Past 12 Months (in 2022 Inflation-Adjusted Dollars) (Asian Alone Householder)
# B19013E_001E Median Household Income in the Past 12 Months (in 2022 Inflation-Adjusted Dollars) (Native Hawaiian and Other Pacific Islander Alone Householder)
# B19013I_001E Median Household Income in the Past 12 Months (in 2022 Inflation-Adjusted Dollars) (Hispanic or Latino Householder)

# B15002_002E Estimate!!Total:!!Male:
# B15002_015E Estimate!!Total:!!Male:!!Bachelor's degree
# B15002_019E Estimate!!Total:!!Female:
# B15002_032E Estimate!!Total:!!Female:!!Bachelor's degree
# C15002A_002E Estimate!!Total:!!Male:  (White Alone)
# C15002A_006E Estimate!!Total:!!Male:!!Bachelor's degree or higher (White Alone)
# C15002A_007E Estimate!!Total:!!Female: (White Alone)
# C15002A_011E Estimate!!Total:!!Female:!!Bachelor's degree or higher (White Alone)
# C15002B_002E Estimate!!Total:!!Male: (Black or African American Alone)
# C15002B_006E Estimate!!Total:!!Male:!!Bachelor's degree or higher (Black or African American Alone)
# C15002B_007E Estimate!!Total:!!Female: (Black or African American Alone)
# C15002B_011E Estimate!!Total:!!Female:!!Bachelor's degree or higher (Black or African American Alone)
# C15002C_002E Estimate!!Total:!!Male: (American Indian and Alaska Native Alone)
# C15002C_006E Estimate!!Total:!!Male:!!Bachelor's degree or higher (American Indian and Alaska Native Alone)
# C15002C_007E Estimate!!Total:!!Female: (American Indian and Alaska Native Alone)
# C15002C_011E Estimate!!Total:!!Female:!!Bachelor's degree or higher (American Indian and Alaska Native Alone)
# C15002D_002E Estimate!!Total:!!Male: (Asian Alone)
# C15002D_006E Estimate!!Total:!!Male:!!Bachelor's degree or higher (Asian Alone)
# C15002D_007E Estimate!!Total:!!Female: (Asian Alone)
# C15002D_011E Estimate!!Total:!!Female:!!Bachelor's degree or higher (Asian Alone)
# C15002E_002E Estimate!!Total:!!Male: (Native Hawaiian and Other Pacific Islander Alone)
# C15002E_006E Estimate!!Total:!!Male:!!Bachelor's degree or higher (Native Hawaiian and Other Pacific Islander Alone)
# C15002E_007E Estimate!!Total:!!Female: (Native Hawaiian and Other Pacific Islander Alone)
# C15002E_011E Estimate!!Total:!!Female:!!Bachelor's degree or higher (Native Hawaiian and Other Pacific Islander Alone)
# C15002I_002E Estimate!!Total:!!Male: (Hispanic or Latino)
# C15002I_006E Estimate!!Total:!!Male:!!Bachelor's degree or higher (Hispanic or Latino)
# C15002I_007E Estimate!!Total:!!Female: (Hispanic or Latino)
# C15002I_011E Estimate!!Total:!!Female:!!Bachelor's degree or higher (Hispanic or Latino)

# C27001A_002E Estimate!!Total:!!Under 19 years: (White Alone)
# C27001A_003E Estimate!!Total:!!Under 19 years:!!With health insurance coverag (White Alone)
# C27001A_005E Estimate!!Total:!!19 to 64 years: (White Alone)
# C27001A_006E Estimate!!Total:!!19 to 64 years:!!With health insurance coverage (White Alone)
# C27001A_008E Estimate!!Total:!!65 years and over: (White Alone)
# C27001A_009E Estimate!!Total:!!65 years and over:!!With health insurance coverage (White Alone)
#--------
# C27001B_002E Estimate!!Total:!!Under 19 years: (Black or African American Alone)
# C27001B_003E 	Estimate!!Total:!!Under 19 years:!!With health insurance coverage (Black or African American Alone)
# C27001B_005E Estimate!!Total:!!19 to 64 years: (Black or African American Alone)
# C27001B_006E Estimate!!Total:!!19 to 64 years:!!With health insurance coverage (Black or African American Alone)
# C27001B_008E Estimate!!Total:!!65 years and over: (Black or African American Alone)
# C27001B_009E Estimate!!Total:!!65 years and over:!!With health insurance coverage (Black or African American Alone)
#--------
# C27001C_002E Estimate!!Total:!!Under 19 years: (American Indian and Alaska Native Alone)
# C27001C_003E Estimate!!Total:!!Under 19 years:!!With health insurance coverage (American Indian and Alaska Native Alone)
# C27001C_005E Estimate!!Total:!!19 to 64 years: (American Indian and Alaska Native Alone)
# C27001C_006E Estimate!!Total:!!19 to 64 years:!!With health insurance coverage (American Indian and Alaska Native Alone)
# C27001C_008E Estimate!!Total:!!65 years and over: (American Indian and Alaska Native Alone)
# C27001C_009E Estimate!!Total:!!65 years and over:!!With health insurance coverage (American Indian and Alaska Native Alone)
#--------
# C27001D_002E Estimate!!Total:!!Under 19 years: (Asian Alone)
# C27001D_003E Estimate!!Total:!!Under 19 years:!!With health insurance coverage (Asian Alone)
# C27001D_005E Estimate!!Total:!!19 to 64 years: (Asian Alone)
# C27001D_006E Estimate!!Total:!!19 to 64 years:!!With health insurance coverage (Asian Alone)
# C27001D_008E Estimate!!Total:!!65 years and over: (Asian Alone)
# C27001D_009E Estimate!!Total:!!65 years and over:!!With health insurance coverage (Asian Alone)
#--------
# C27001E_002E Estimate!!Total:!!Under 19 years: (Native Hawaiian and Other Pacific Islander Alone)
# C27001E_003E Estimate!!Total:!!Under 19 years:!!With health insurance coverage (Native Hawaiian and Other Pacific Islander Alone)
# C27001E_005E Estimate!!Total:!!19 to 64 years: (Native Hawaiian and Other Pacific Islander Alone)
# C27001E_006E Estimate!!Total:!!19 to 64 years:!!With health insurance coverage (Native Hawaiian and Other Pacific Islander Alone)
# C27001E_008E Estimate!!Total:!!65 years and over: (Native Hawaiian and Other Pacific Islander Alone)
# C27001E_009E Estimate!!Total:!!65 years and over:!!With health insurance coverage (Native Hawaiian and Other Pacific Islander Alone)
#-------
# C27001I_002E Estimate!!Total:!!Under 19 years: (Hispanic or Latino)
# C27001I_003E Estimate!!Total:!!Under 19 years:!!With health insurance coverage (Hispanic or Latino)
# C27001I_005E Estimate!!Total:!!19 to 64 years: (Hispanic or Latino)
# C27001I_006E Estimate!!Total:!!19 to 64 years:!!With health insurance coverage (Hispanic or Latino)
# C27001I_008E Estimate!!Total:!!65 years and over: (Hispanic or Latino)
# C27001I_009E Estimate!!Total:!!65 years and over:!!With health insurance coverage (Hispanic or Latino)

# median income, White, African, Hispanic, Asian, American Indian, Native Hawaiian
# education, White, African, Hispanic, Asian, American Indian, Native Hawaiian
# insurance coverage, White, African, Hispanic, Asian, American Indian, Native Hawaiian
income_variables = 'B19013_001E,B19013A_001E,B19013B_001E,B19013I_001E,B19013D_001E,B19013C_001E,B19013E_001E'
income_rename = {'B19013_001E': 'Median Household Income', 
                 'B19013A_001E': 'Median Household Income (White)',
                 'B19013B_001E': 'Median Household Income (Black or African American)', 
                 'B19013I_001E': 'Median Household Income (Hispanic or Latino)',
                 'B19013D_001E': 'Median Household Income (Asian)', 
                 'B19013C_001E': 'Median Household Income (American Indian and Alaska Native)',
                 'B19013E_001E': 'Median Household Income (Native Hawaiian and Other Pacific Islander)'}
# get income info
results = []
income_missing = 0
for zip_code in distinct_zip_df['distinct_zip']:
    params = {
        'get': 'NAME,'+ income_variables,
        'for': f'zip code tabulation area:{zip_code}',
        'key': census_api_key
    }

    response = requests.get(census_base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        results.append(df)
    else:
        response = requests.get(previous_census_base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data[1:], columns=data[0])
            results.append(df)
        else:
            income_missing += 1
            print(f"Failed to retrieve income data for zip code: {zip_code}")
income_df = pd.concat(results, ignore_index=True)
income_df.rename(columns=income_rename, inplace=True)
print('income_missing: {}'.format(income_missing))



education_variable = ('B15002_002E,B15002_015E,B15002_019E,B15002_032E,'
                      'C15002A_002E,C15002A_006E,C15002A_007E,C15002A_011E,'
                      'C15002B_002E,C15002B_006E,C15002B_007E,C15002B_011E,'
                      'C15002I_002E,C15002I_006E,C15002I_007E,C15002I_011E,'
                      'C15002D_002E,C15002D_006E,C15002D_007E,C15002D_011E,'
                      'C15002C_002E,C15002C_006E,C15002C_007E,C15002C_011E,'
                      'C15002E_002E,C15002E_006E,C15002E_007E,C15002E_011E')
education_rename = {'B15002_002E': 'Total Male', 'B15002_015E': 'Male, Bachelor\'s degree', 'B15002_019E': 'Total, Female', 'B15002_032E': 'Female:!!Bachelor\'s degree',
                    'C15002A_002E': 'Male (White)', 'C15002A_006E': 'Male, Bachelor\'s degree or higher (White)', 'C15002A_007E': 'Female (White)', 'C15002A_011E': 'Female, Bachelor\'s degree or higher (White)',
                    'C15002B_002E': 'Male (Black or African American)', 'C15002B_006E': 'Male, Bachelor\'s degree or higher (Black or African American)', 'C15002B_007E': 'Female (Black or African American)', 'C15002B_011E': 'Female, Bachelor\'s degree or higher (Black or African American)',
                    'C15002I_002E': 'Male (Hispanic or Latino)', 'C15002I_006E': 'Male, Bachelor\'s degree or higher (Hispanic or Latino)', 'C15002I_007E': 'Female (Hispanic or Latino)', 'C15002I_011E': 'Female, Bachelor\'s degree or higher (Hispanic or Latino)',
                    'C15002D_002E': 'Male (Asian)', 'C15002D_006E': 'Male, Bachelor\'s degree or higher (Asian)', 'C15002D_007E': 'Female (Asian)', 'C15002D_011E': 'Female, Bachelor\'s degree or higher (Asian)',
                    'C15002C_002E': 'Male (American Indian and Alaska Native)', 'C15002C_006E': 'Male, Bachelor\'s degree or higher (American Indian and Alaska Native)', 'C15002C_007E': 'Female (American Indian and Alaska Native)', 'C15002C_011E': 'Female, Bachelor\'s degree or higher (American Indian and Alaska Native',
                    'C15002E_002E': 'Male (Native Hawaiian and Other Pacific Islander)', 'C15002E_006E': 'Male, Bachelor\'s degree or higher (Native Hawaiian and Other Pacific Islander)', 'C15002E_007E': 'Female (Native Hawaiian and Other Pacific Islander)', 'C15002E_011E': 'Female, Bachelor\'s degree or higher (Native Hawaiian and Other Pacific Islander)'}

# get education info
results = []
education_missing = 0
for zip_code in distinct_zip_df['distinct_zip']:
    params = {
        'get': education_variable,
        'for': f'zip code tabulation area:{zip_code}',
        'key': census_api_key
    }

    response = requests.get(census_base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        results.append(df)
    else:
        response = requests.get(previous_census_base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data[1:], columns=data[0])
            results.append(df)
        else:
            education_missing += 1
            print(f"Failed to retrieve education data for zip code: {zip_code}")
education_df = pd.concat(results, ignore_index=True)
education_df['Education_Percentage_Male(General)'] = 1.0 * education_df['B15002_015E'].astype(int)/education_df['B15002_002E'].astype(int)
education_df['Education_Percentage_Female(General)'] = 1.0 * education_df['B15002_032E'].astype(int)/education_df['B15002_019E'].astype(int)
education_df['Education_Percentage_Male(White)'] = 1.0 * education_df['C15002A_006E'].astype(int)/education_df['C15002A_002E'].astype(int)
education_df['Education_Percentage_Female(White)'] = 1.0 * education_df['C15002A_011E'].astype(int)/education_df['C15002A_007E'].astype(int)
education_df['Education_Percentage_Male(Black or African American)'] = 1.0 * education_df['C15002B_006E'].astype(int)/education_df['C15002B_002E'].astype(int)
education_df['Education_Percentage_Female(Black or African American)'] = 1.0 * education_df['C15002B_011E'].astype(int)/education_df['C15002B_007E'].astype(int)
education_df['Education_Percentage_Male(Hispanic or Latino)'] = 1.0 * education_df['C15002I_006E'].astype(int)/education_df['C15002I_002E'].astype(int)
education_df['Education_Percentage_Female(Hispanic or Latino)'] = 1.0 * education_df['C15002I_011E'].astype(int)/education_df['C15002I_007E'].astype(int)
education_df['Education_Percentage_Male(Asian)'] = 1.0 * education_df['C15002D_006E'].astype(int)/education_df['C15002D_002E'].astype(int)
education_df['Education_Percentage_Female(Asian)'] = 1.0 * education_df['C15002D_011E'].astype(int)/education_df['C15002D_007E'].astype(int)
education_df['Education_Percentage_Male(American Indian and Alaska Native)'] = 1.0 * education_df['C15002C_006E'].astype(int)/education_df['C15002C_002E'].astype(int)
education_df['Education_Percentage_Female(American Indian and Alaska Native)'] = 1.0 * education_df['C15002C_011E'].astype(int)/education_df['C15002C_007E'].astype(int)
education_df['Education_Percentage_Male(Native Hawaiian and Other Pacific Islander)'] = 1.0 * education_df['C15002E_006E'].astype(int)/education_df['C15002E_002E'].astype(int)
education_df['Education_Percentage_Female(Native Hawaiian and Other Pacific Islander)'] = 1.0 * education_df['C15002E_011E'].astype(int)/education_df['C15002E_007E'].astype(int)
education_df = education_df.drop(['B15002_002E','B15002_015E','B15002_019E','B15002_032E','C15002A_002E','C15002A_006E','C15002A_007E','C15002A_011E',
                                  'C15002B_002E','C15002B_006E','C15002B_007E','C15002B_011E','C15002I_002E','C15002I_006E','C15002I_007E','C15002I_011E',
                                  'C15002D_002E','C15002D_006E','C15002D_007E','C15002D_011E','C15002C_002E','C15002C_006E','C15002C_007E','C15002C_011E',
                                  'C15002E_002E','C15002E_006E','C15002E_007E','C15002E_011E'], axis=1)
print('education_missing: {}'.format(education_missing))



insurance_variable = ('B27001_018E,B27001_019E,B27001_021E,B27001_022E,B27001_024E,B27001_025E,B27001_027E,B27001_028E,'
                      'B27001_046E,B27001_047E,B27001_049E,B27001_050E,B27001_052E,B27001_053E,B27001_055E,B27001_056E')
insurance_rename = {'B27001_018E': '45-54 male total', 'B27001_019E': '45-54 male insurance',
                    'B27001_021E': '55-64 male total', 'B27001_022E': '55-64 male insurance',
                    'B27001_024E': '65-74 male total', 'B27001_025E': '65-74 male insurance',
                    'B27001_027E': '75 male total', 'B27001_028E': '75 male insurance',

                    'B27001_046E': '45-54 female total', 'B27001_047E': '45-54 female insurance',
                    'B27001_049E': '55-64 female total', 'B27001_050E': '55-64 female insurance',
                    'B27001_052E': '65-74 female total', 'B27001_053E': '65-74 female insurance',
                    'B27001_055E': '75 female total', 'B27001_056E': '75 female insurance'
                    }

# get insurance info
results = []
insurance_missing = 0
for zip_code in distinct_zip_df['distinct_zip']:
    params = {
        'get': insurance_variable,
        'for': f'zip code tabulation area:{zip_code}',
        'key': census_api_key
    }
    response = requests.get(census_base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data[1:], columns=data[0])
        results.append(df)
    else:
        response = requests.get(previous_census_base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data[1:], columns=data[0])
            results.append(df)
        else:
            insurance_missing += 1
            print(f"Failed to retrieve insurance data for zip code: {zip_code}")
insurance_df = pd.concat(results, ignore_index=True)
insurance_df['Insurance_Percentage, male 54'] = 1.0 * insurance_df['B27001_019E'].astype(int)/insurance_df['B27001_018E'].astype(int)
insurance_df['Insurance_Percentage, male 64'] = 1.0 * insurance_df['B27001_022E'].astype(int)/insurance_df['B27001_021E'].astype(int)
insurance_df['Insurance_Percentage, male 74'] = 1.0 * insurance_df['B27001_025E'].astype(int)/insurance_df['B27001_024E'].astype(int)
insurance_df['Insurance_Percentage, male over 75'] = 1.0 * insurance_df['B27001_028E'].astype(int)/insurance_df['B27001_027E'].astype(int)
insurance_df['Insurance_Percentage, female 54'] = 1.0 * insurance_df['B27001_047E'].astype(int)/insurance_df['B27001_046E'].astype(int)
insurance_df['Insurance_Percentage, female 64'] = 1.0 * insurance_df['B27001_050E'].astype(int)/insurance_df['B27001_049E'].astype(int)
insurance_df['Insurance_Percentage, female 74'] = 1.0 * insurance_df['B27001_053E'].astype(int)/insurance_df['B27001_052E'].astype(int)
insurance_df['Insurance_Percentage, female over 75'] = 1.0 * insurance_df['B27001_056E'].astype(int)/insurance_df['B27001_055E'].astype(int)

insurance_df = insurance_df.drop(['B27001_018E','B27001_019E','B27001_021E','B27001_022E','B27001_024E','B27001_025E','B27001_027E','B27001_028E',
                                  'B27001_046E','B27001_047E','B27001_049E','B27001_050E','B27001_052E','B27001_053E','B27001_055E','B27001_056E'], axis=1)
# outer join
result = pd.merge(income_df, education_df, on='zip code tabulation area', how='outer')
result = pd.merge(result, insurance_df, on='zip code tabulation area', how='outer')
final_merged_df = distinct_zip_df.merge(result, left_on='distinct_zip', right_on='zip code tabulation area', how='left')
# Save the dictionary to a file
final_merged_df.to_csv('../data/data_treatment/distinct_zip_income_education_insurance.csv', index=False)
print('insurance_missing: {}'.format(insurance_missing))

# IPF_SDOH
code for AMIA 2025 [Investigating the Impact of Social Determinants of Health on Diagnostic Delays and Access to Antifibrotic Treatment in Idiopathic Pulmonary Fibrosis](sdoh.pdf)

If you feel this code is useful, please cite our work
```
@article{li2024investigating,
  title={Investigating the Impact of Social Determinants of Health on Diagnostic Delays and Access to Antifibrotic Treatment in Idiopathic Pulmonary Fibrosis},
  author={Li, Rui and Lu, Qiuhao and Wen, Andrew and Wang, Jinlian and Fu, Sunyang and Ruan, Xiaoyang and Wang, Liwei and Liu, Hongfang},
  journal={medRxiv},
  pages={2024--11},
  year={2024},
  publisher={Cold Spring Harbor Laboratory Press}
}
```
## Prepare
Request the [Census Data API Key](https://api.census.gov/data/key_signup.html).
All census data used in this paper is using [American Community Survey 5-Year Data (2009-2023) API](https://www.census.gov/data/developers/data-sets/acs-5year.html).

Request the [Air Quality System (AQS) API Key](https://aqs.epa.gov/aqsweb/documents/data_api.html#lists).

## Running step
1. Get the census data

run [get_sdoh.py](code/get_sdoh.py), replace line 14 with your own census api key. The input is a csv file contains columns including 5-digit zip code. The output is the average household income, education and insurance within that area for patients with specific for race, age, and gender.

2. Get the PM2.5 data

Run [get_pm25_info.py](code/get_pm25_info.py), replace line 5 and 6 with your email and key.


3. Map the census and PM2.5 data to individual patients based on their gender , race and age.

run [map_census_pm_to_patients.py](code/map_census_pm_to_patients.py), to assign the sdoh features to every patients based on their age, gender and race. The input is three csv files, ipf_patient_info.csv records the patient demographic information and clinical outcomes of interest; distinct_zip_income_education_insurance.csv records the census information we obtained from step 1; pm25.csv records the pm2.5 data we obtained from step 2. The output is new_output.csv which contains patient demographics, clinical outcomes of interest, and sdohs.


4. Train machine learning models and analyze the association between the SDoH and clinical outcomes.

for two tasks, run regression_model.py and xgboost_classification.py under each folder.






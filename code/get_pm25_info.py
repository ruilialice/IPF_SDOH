import requests
import pandas as pd

file = '../data/zip_to_county/distinct_zip_county_code.csv'
email_address = 'rui.li.1@uth.tmc.edu'
key = 'amberram37'

# air pollution info
# the information is from website https://aqs.epa.gov/aqsweb/documents/data_api.html
air_url = 'https://aqs.epa.gov/data/api/annualData/byCounty?'

global no_data
global no_element1, no_element2
no_data = 0
no_element1 = 0
no_element2 = 0

def fetch_pm25_data(state_code, county_code):
    global no_data, no_element, no_element1, no_element2
    params = {
        'email': email_address,
        'key': key,
        'param': 88101,
        'bdate': '20230101',
        'edate': '20231231',
        'state': state_code,
        'county': county_code
    }
    response = requests.get(air_url, params=params)
    if response.status_code == 200:
        tmp = response.json()
        data = tmp['Data']

        if len(data)==0:
            no_data += 1
        # search the list, find the first row that
        # "sample_duration": "24 HOUR"
        # "method": "R & P Model 2025 PM-2.5 Sequential Air Sampler w/VSCC - Gravimetric"
        # "metric_used": "Quarterly Means of Daily Means"

        criteria_2 = {
            "sample_duration": "24 HOUR",
            "method": "R & P Model 2025 PM-2.5 Sequential Air Sampler w/VSCC - Gravimetric",
            "metric_used": "Quarterly Means of Daily Means"
        }
        criteria_1 = {
            "sample_duration": "24-HR BLK AVG",
            "metric_used": "Quarterly Means of Daily Means"
        }

        # Function to find the first matching element
        def find_first_matching_element(data, criteria):
            for element in data:
                if all(item in element.items() for item in criteria.items()):
                    return element
            return None
        element = find_first_matching_element(data, criteria_1)
        if element:
            mean = element['arithmetic_mean']
            deviation = element['standard_deviation']
            return pd.Series([mean, deviation])
        else:
            if len(data)>0:
                no_element1 += 1
                element2 = find_first_matching_element(data, criteria_2)
                if element2:
                    mean = element2['arithmetic_mean']
                    deviation = element2['standard_deviation']
                    # return pd.Series([mean, deviation])
                else:
                    no_element2 += 1
                #     return pd.Series([None, None])
            return pd.Series([None, None])
    else:
        return pd.Series([None, None])

zip_county_code_df = pd.read_csv(file, dtype={'state_code': str, 'county_code': str})
zip_county_code_df[['PM2.5_mean', 'PM2.5_dev']] = zip_county_code_df.apply(lambda row: fetch_pm25_data(row['state_code'], row['county_code']), axis=1)

zip_county_code_df.to_csv('../data/zip_to_county/pm25.csv', index=False)
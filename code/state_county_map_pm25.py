import requests
import pickle
import pandas as pd

# # the following code is used to map the state and county name into code
# # the code is further used to get the pm25 info
# email_address = 'rui.li.1@uth.tmc.edu'
# key = 'amberram37'
# # state code
# base_url = 'https://aqs.epa.gov/data/api/list/states?'
# params = {
#         'email': email_address,
#         'key': key
#     }
# response = requests.get(base_url, params=params)
# if response.status_code == 200:
#     data = response.json()['Data']
#     # Convert list to dictionary
#     state_code_dict = {item['value_represented']: item['code'] for item in data}
#
# county_url = 'https://aqs.epa.gov/data/api/list/countiesByState?'
# # for each state, find code of the county
# state_code_county_code = {}
# for state, state_code in state_code_dict.items():
#     params = {
#         'email': email_address,
#         'key': key,
#         'state': state_code
#     }
#     response = requests.get(county_url, params=params)
#     if response.status_code == 200:
#         data = response.json()['Data']
#         # Convert list to dictionary
#         county_code_dict = {item['value_represented']: item['code'] for item in data}
#         state_code_county_code[state] = [state_code, county_code_dict]
#     else:
#         print(f"Failed to retrieve data for state code: {state_code}")
#
# with open('../data/zip_to_county/state_code_county_code.pkl', 'wb') as file:
#     pickle.dump(state_code_county_code, file)


## the following code read ZIP_COUNTY_062024.xlsx
## the file is downloaded from https://www.huduser.gov/portal/datasets/usps_crosswalk.html
# read distinct_zip_income_education_insurance.csv containing zip codes
zip_file = '../data/data_treatment/distinct_zip_income_education_insurance.csv'
file_A = pd.read_csv(zip_file, dtype={'distinct_zip': str})

# zip to county fips
# if zip has multiple fips, select the one has the maximum total ratio
zip_to_county_file = '../data/zip_to_county/zip_to_county.csv'
file_B = pd.read_csv(zip_to_county_file, dtype={'ZIP': str, 'COUNTY': str, 'TOT_RATIO': float})
# For file_B, filter to keep only the row with the largest TOT_RATIO for each ZIP
file_B_filtered = file_B.loc[file_B.groupby('ZIP')['TOT_RATIO'].idxmax()]

# Perform the left join on the 'patient_location_zip' column in file_A with the 'ZIP' column in file_B
merged_df = pd.merge(file_A, file_B_filtered, how='left', left_on='distinct_zip', right_on='ZIP')
# Select only the necessary columns (assuming you want to keep all columns from file_A and the COUNTY column from file_B)
result_df = merged_df[['distinct_zip', 'COUNTY']]

# fips to county, with state and county name
fips_to_county_file = '../data/zip_to_county/FipsCountyCodes.csv'
file_C = pd.read_csv(fips_to_county_file, dtype={'FIPS': str, 'Name': str})
# Perform the second left join on the 'COUNTY' column in merged_df_1 with the 'FIPS' column in file_C
final_merged_df = pd.merge(result_df, file_C, how='left', left_on='COUNTY', right_on='FIPS')
# Split the 'NAME' column into 'state' and 'county_name' columns
final_merged_df[['state', 'county_name']] = final_merged_df['Name'].str.split(', ', expand=True)

# convert the abbreviations of states into full names
# Dictionary to map state abbreviations to full state names
state_abbreviation_to_full = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "IA": "Iowa",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "ME": "Maine",
    "MD": "Maryland",
    "MA": "Massachusetts",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MS": "Mississippi",
    "MO": "Missouri",
    "MT": "Montana",
    "NE": "Nebraska",
    "NV": "Nevada",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NY": "New York",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VT": "Vermont",
    "VA": "Virginia",
    "WA": "Washington",
    "WV": "West Virginia",
    "WI": "Wisconsin",
    "WY": "Wyoming",
    "DC": "New York"
}

# Function to map state abbreviation to full name
def map_state_abbreviation_to_full(abbreviation):
    if pd.isna(abbreviation):
        return None
    return state_abbreviation_to_full.get(abbreviation, "Unknown")

# Apply the function to create the 'full_state' column
final_merged_df['full_state'] = final_merged_df['state'].apply(map_state_abbreviation_to_full)

# Load the dictionary from the file
with open('../data/zip_to_county/state_code_county_code.pkl', 'rb') as file:
    state_code_county_code = pickle.load(file)


def map_state_county(row, state_county_dict):
    state = row['full_state']
    county = row['county_name']

    if state in state_county_dict:
        state_code = state_county_dict[state][0]
        county_dict = state_county_dict[state][1]
        if county in county_dict:
            county_code = county_dict[county]
            return pd.Series([state_code, county_code])

        # Attempt to concatenate words
        concatenated_county = county.replace(' ', '')
        if concatenated_county in county_dict:
            county_code = county_dict[concatenated_county]
            return pd.Series([state_code, county_code])

        # Attempt to replace 'St.' with 'Saint'
        saint_county = county.replace('St.', 'Saint')
        if saint_county in county_dict:
            county_code = county_dict[saint_county]
            return pd.Series([state_code, county_code])

        # Attempt to replace 'St.' with 'Saint'
        saint_county = county + ' City'
        if saint_county in county_dict:
            county_code = county_dict[saint_county]
            return pd.Series([state_code, county_code])

        saint_county = county.replace('District of ', '')
        if saint_county in county_dict:
            county_code = county_dict[saint_county]
            return pd.Series([state_code, county_code])

    return pd.Series([None, None])


final_merged_df[['state_code', 'county_code']] = final_merged_df.apply(map_state_county, axis=1, state_county_dict=state_code_county_code)

# Save the dictionary to a file
final_merged_df.to_csv('../data/zip_to_county/distinct_zip_county_code.csv', index=False)
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt

# Load the GeoJSON file (replace 'path_to_file.geojson' with your file path)
# texas_zip_shapes = gpd.read_file('../data/map_plot/tx_texas_zip_codes_geo.min.json')
texas_zip_shapes = gpd.read_file('../data/map_plot/texas-zip-codes-_1613.geojson')

# Load your ZIP code and patient data
zip_patient_file = '../data/ipf_zip_num_patients.csv'
zip_patient_data = pd.read_csv(zip_patient_file, names=['ZCTA5CE10', 'patients'], dtype=str)
# Merge patient data with the ZIP code geometries
merged_data = texas_zip_shapes.merge(zip_patient_data, on='ZCTA5CE10', how='left')

# df = pd.DataFrame({
#     'ZCTA5CE10': ['75001', '75002', '75006', '75007', '75010'],
#     'patients': [150, None, 100, None, 200]
# })
# merged_data = texas_zip_shapes.merge(df, on='ZCTA5CE10', how='left')

merged_data['patients'] = merged_data['patients'].fillna(0)
merged_data['patients'] = merged_data['patients'].astype(int)

# Plot the map
fig, ax = plt.subplots(1, 1, figsize=(10, 8))
merged_data.plot(column='patients', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.set_title('Number of Patients by ZIP Code in Texas', fontsize=15)

plt.show()

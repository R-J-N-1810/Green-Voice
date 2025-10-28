import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load datasets
img_features = pd.read_csv('../datasets/image_features.csv')
biosensor = pd.read_csv('../datasets/biosensor_with_id.csv')
smartfarm = pd.read_csv('../datasets/smartfarm_with_id.csv')

# Extract numeric suffix from img_features sample_id
img_features['numeric_id'] = img_features['sample_id'].str.extract(r'(\d+)$')

# Drop rows where numeric_id is NaN
img_features = img_features.dropna(subset=['numeric_id'])

# Convert numeric_id to integer
img_features['numeric_id'] = img_features['numeric_id'].astype('Int64')

# Convert biosensor and smartfarm sample_id to integer
biosensor['sample_id'] = biosensor['sample_id'].astype(int)
smartfarm['sample_id'] = smartfarm['sample_id'].astype(int)

# Extract numeric columns from biosensor and smartfarm
biosensor_num = biosensor.select_dtypes(include='number').copy()
biosensor_num['sample_id'] = biosensor['sample_id']
smartfarm_num = smartfarm.select_dtypes(include='number').copy()
smartfarm_num['sample_id'] = smartfarm['sample_id']

# Normalize numeric data
scaler = MinMaxScaler()
biosensor_scaled = pd.DataFrame(scaler.fit_transform(biosensor_num.drop(columns=['sample_id'])), 
                                columns=biosensor_num.drop(columns=['sample_id']).columns)
biosensor_scaled['sample_id'] = biosensor_num['sample_id']

smartfarm_scaled = pd.DataFrame(scaler.fit_transform(smartfarm_num.drop(columns=['sample_id'])), 
                                columns=smartfarm_num.drop(columns=['sample_id']).columns)
smartfarm_scaled['sample_id'] = smartfarm_num['sample_id']

# Merge biosensor and smartfarm on sample_id
tabular = pd.merge(biosensor_scaled, smartfarm_scaled, on='sample_id', how='inner')

# Merge with img_features
full_data = pd.merge(tabular, img_features, left_on='sample_id', right_on='numeric_id', how='inner')

# Drop BOTH the numeric_id AND the original sample_id string columns
full_data.drop(columns=['numeric_id', 'sample_id_y'], inplace=True)

# Rename sample_id_x to sample_id for clarity
full_data.rename(columns={'sample_id_x': 'sample_id'}, inplace=True)

# Print verification
print(f"final merged rows: {len(full_data)}")
print(f"Columns: {full_data.columns.tolist()}")

# Save merged dataset
full_data.to_csv('../datasets/full_merged_data.csv', index=False)
print("Datasets merged and saved.")

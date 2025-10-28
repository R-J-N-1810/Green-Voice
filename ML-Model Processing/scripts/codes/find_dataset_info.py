import os
import pandas as pd
import glob

search_paths = [
    '../datasets/',
    '../data/',
    '../../datasets/',
    'datasets/',
    'data/',
]

found_files = []

for search_path in search_paths:
    if os.path.exists(search_path):
        csv_files = glob.glob(os.path.join(search_path, '*.csv'))
        for csv_file in csv_files:
            found_files.append(csv_file)

for root, dirs, files in os.walk('.', topdown=True):
    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
    for file in files:
        if file.endswith('.csv'):
            full_path = os.path.join(root, file)
            if full_path not in found_files:
                found_files.append(full_path)

if not found_files:
    print("no csv files found")
    merged_csv = input("enter csv path: ").strip()
else:
    print(f"found {len(found_files)} csv files")
    merged_files = [f for f in found_files if 'merged' in f.lower() or 'full' in f.lower()]
    
    if merged_files:
        for i, f in enumerate(merged_files, 1):
            print(f"{i}. {f}")
        if len(merged_files) == 1:
            merged_csv = merged_files[0]
        else:
            choice = int(input("select file: ")) - 1
            merged_csv = merged_files[choice]
    else:
        for i, f in enumerate(found_files, 1):
            print(f"{i}. {f}")
        choice = int(input("select: ")) - 1
        merged_csv = found_files[choice]

df = pd.read_csv(merged_csv)
print(f"shape: {df.shape}")
print(f"columns: {list(df.columns[:5])}")

disease_keywords = ['disease', 'health', 'condition', 'status', 'label', 'class', 'type', 'category']
potential_label_cols = []
for col in df.columns:
    col_lower = col.lower()
    for keyword in disease_keywords:
        if keyword in col_lower:
            potential_label_cols.append(col)
            break

filtered_cols = []
for col in potential_label_cols:
    n_unique = df[col].nunique()
    is_numeric = df[col].dtype in ['int64', 'float64', 'int32', 'float32']
    if is_numeric and n_unique > 50:
        continue
    filtered_cols.append(col)

if filtered_cols:
    for i, col in enumerate(filtered_cols, 1):
        n_unique = df[col].nunique()
        unique_values = df[col].unique()[:5]
        print(f"{i}. {col} - {n_unique} unique")
        print(f"   examples: {list(unique_values)}")
    
    if len(filtered_cols) == 1:
        label_col = filtered_cols[0]
    else:
        choice = int(input("select label column: ")) - 1
        label_col = filtered_cols[choice]
else:
    for i, col in enumerate(df.columns, 1):
        print(f"{i}. {col}")
    choice = int(input("select: ")) - 1
    label_col = df.columns[choice]

class_distribution = df[label_col].value_counts()
num_classes = len(class_distribution)

print(f"\nclasses: {num_classes}")
for cls, count in class_distribution.items():
    percentage = (count / len(df)) * 100
    print(f"  {str(cls)}: {count} ({percentage:.1f}%)")

print(f"\nconfig:")
print(f"data_path = r'{merged_csv}'")
print(f"label_col = '{label_col}'")

config_file = 'dataset_config.txt'
config_content = f"""data_path = r'{merged_csv}'
label_col = '{label_col}'
num_classes = {num_classes}
classes = {list(class_distribution.index)}
"""

with open(config_file, 'w') as f:
    f.write(config_content)

print(f"\nconfig saved to {config_file}")

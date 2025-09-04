import os
import shutil
import pandas as pd

raw_dir = "../raw"
mapping_file = "../metadata/class_mapping.csv"
processed_dir = "../processed"

mapping_pd = pd.read_csv(mapping_file)

os.makedirs(processed_dir, exist_ok=True)

for _,row in mapping_pd.iterrows():
    last_class = row["last_class"]
    current_class = row["current_class"]
    dataset = row["dataset"]

    original_dir = os.path.join(raw_dir,dataset,current_class)
    final_dir = os.path.join(processed_dir,last_class)

    os.makedirs(final_dir, exist_ok=True)

    if not os.path.exists(original_dir):
        print(f" {original_dir} not found. Skipping...")
        continue

    for jpg_file in os.listdir(original_dir):
        original_path = os.path.join(original_dir, jpg_file)
        final_path = os.path.join(final_dir, jpg_file)

        if os.path.exists(final_path):
            first,second = os.path.splitext(jpg_file)
            i = 1
            while os.path.exists(final_path):
                final_path = os.path.join(final_dir, f"{first}_{i}{second}")
                i += 1

        shutil.copy2(original_path,final_path)

print(f"'processed' folder was done.")


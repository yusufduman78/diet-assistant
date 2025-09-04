import os
import shutil
import pandas as pd
from tqdm import tqdm

raw_dir = "../raw"
mapping_file = "../metadata/class_mapping.csv"
processed_dir = "../processed"

mapping_pd = pd.read_csv(mapping_file)

os.makedirs(processed_dir, exist_ok=True)

def find_class(base_path,target_class):

    for root,dirs,files in os.walk(base_path):
        if os.path.basename(root) == target_class:
            return root
    return None

for _,row in tqdm(mapping_pd.iterrows(), total = len(mapping_pd), desc="Processing"):
    last_class = row["last_class"]
    current_class = row["current_class"]
    dataset = row["dataset"]

    dataset_dir = os.path.join(raw_dir,dataset)
    class_dir = find_class(dataset_dir,current_class)

    if not class_dir:
        print(f"{current_class} folder not found in {dataset}")
        continue

    final_dir = os.path.join(processed_dir,last_class)
    os.makedirs(final_dir, exist_ok=True)

    file_list = os.listdir(class_dir)

    for jpg_file in tqdm(file_list,desc=f"Copying folder '{current_class}'"):
        if jpg_file.lower().endswith((".jpg",".png")):
            start_path = os.path.join(class_dir,jpg_file)
            final_path = os.path.join(final_dir,jpg_file)

            if os.path.exists(final_path):
                fst, snd = os.path.splitext(jpg_file)
                i = 1
                while os.path.exists(final_path):
                    final_path = os.path.join(final_dir, f"{fst}_{i}{snd}")
                    i += 1
            shutil.copy2(start_path,final_path)

print(f"'processed' was done.")
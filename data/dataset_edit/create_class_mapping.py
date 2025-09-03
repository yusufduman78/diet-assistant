import os
import csv


metadata_path = "../metadata"
classes_csv = os.path.join(metadata_path,"classes.csv")
mapping_csv = os.path.join(metadata_path,"class_mapping.csv")

os.makedirs(metadata_path,exist_ok=True)

all_classes = []

with open(classes_csv,"r",encoding="utf-8") as f: # reading classes.csv and getting datasets and classes
    reader = csv.DictReader(f)
    for nth_row in reader:
        dataset = nth_row["dataset_name"]
        class_name = nth_row["class_name"]
        all_classes.append((class_name,dataset))

with open(mapping_csv,"w",newline="",encoding="utf-8") as f: # class_mapping.csv and preparing the mapping template
    writer = csv.writer(f)
    writer.writerow(["last_class","current_class","dataset"])
    all_rows = [["",class_name,dataset] for class_name,dataset in all_classes]
    writer.writerows(all_rows)

print(f"Class mapping template created: '{mapping_csv}' ")
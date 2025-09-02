import os
import csv

root = "../raw"
metadata_path = "../metadata"
output_file = os.path.join(metadata_path,"classes.csv")

os.makedirs(metadata_path, exist_ok=True)

classes = []

for dataset_name in os.listdir(root):
    dataset_path = os.path.join(root,dataset_name)
    if os.path.isdir(dataset_path):
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path,class_name)
            if os.path.isdir(class_path):
                num_images = len([f for f in os.listdir(class_path) if f.lower().endswith((".jpg",".png"))])
                classes.append((dataset_name,class_name,num_images))

with open(output_file, "w",newline="",encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["dataset_name","class_name","num_images"])
    writer.writerows(classes)

print(f"Total {len(classes)} classes found. Saved to file '{output_file}' ")
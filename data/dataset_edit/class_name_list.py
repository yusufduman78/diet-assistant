import os
import csv

root = "../raw"
metadata_path = "../metadata"
output_file = os.path.join(metadata_path,"classes.csv")

os.makedirs(metadata_path, exist_ok=True)

classes = []

def find_classes(base_path,dataset_name):
    nth_layer = os.listdir(base_path)
    image_files = [f for f in nth_layer if f.lower().endswith((".jpg",".png"))]
    subfolders = [d for d in nth_layer if os.path.isdir(os.path.join(base_path,d))]

    if image_files:
        num_images = len(image_files)
        class_name = os.path.basename(base_path)
        classes.append((dataset_name,class_name,num_images))
    else:
        for sub in subfolders:
            find_classes(os.path.join(base_path,sub),dataset_name)

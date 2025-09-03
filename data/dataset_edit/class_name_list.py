import os
import csv

from sympy import wronskian

root = "../raw"
metadata_path = "../metadata"
output_file = os.path.join(metadata_path,"classes.csv")

os.makedirs(metadata_path, exist_ok=True)

classes = []

def find_classes(base_path,dataset_name):
    """
        Traverse with recursion method and find class folders
    :param base_path:
    :param dataset_name:
    :return:
    """
    nth_layer = os.listdir(base_path) # folders in the current folder
    image_files = [f for f in nth_layer if f.lower().endswith((".jpg",".png"))] # if there are files ending with '.jpg' and '.png' in the folder
    subfolders = [d for d in nth_layer if os.path.isdir(os.path.join(base_path,d))] # if there is a folder within the existing folder

    if image_files: # if there are images in this folder -> class folder
        num_images = len(image_files) # images number
        class_name = os.path.basename(base_path) # class name
        classes.append((dataset_name,class_name,num_images)) # to save .csv file
    else: # if not
        for sub in subfolders: #traverse in subfolders
            find_classes(os.path.join(base_path,sub),dataset_name)

for dataset_name in os.listdir(root): # folders of datasets
    dataset_path = os.path.join(root,dataset_name)
    if os.path.isdir(dataset_path): # if it is folder
        find_classes(dataset_path,dataset_name) # call the function

with open(output_file,"w",newline="",encoding="utf-8") as f: # open csv file in 'write' mod
    writer = csv.writer(f)
    writer.writerow(["dataset_name","class_name","num_images"])
    writer.writerows(classes)

print(f"Total {len(classes)} classes found. Saved to file '{output_file}' ")
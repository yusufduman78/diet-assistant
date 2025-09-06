import os
import csv

#file paths
final_dataset_path = "../processed/01_final_dataset"
train_path = os.path.join(final_dataset_path,"train")
test_path = os.path.join(final_dataset_path,"test")
val_path = os.path.join(final_dataset_path,"val")

#output csv
csv_file = os.path.join(final_dataset_path,"dataset_split.csv")

with open(csv_file,"w",newline="",encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["file_path","class_name","split"]) #header

    for name,dirs in [("train", train_path), ("val", val_path), ("test", test_path)]:
        if not os.path.exists(dirs):
            continue
        for clas_name in os.listdir(dirs):
            class_dir = os.path.join(dirs,clas_name)
            if not os.path.isdir(class_dir):
                continue
            for img_file in os.listdir(class_dir):
                if img_file.lower().endswith((".jpg",".png")):
                    file_path = os.path.join(class_dir,img_file)
                    writer.writerow([file_path,clas_name,name])

print(f"Created: {csv_file}")

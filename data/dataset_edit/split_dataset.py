import os
import shutil
import random
import pandas as pd

processed_dir = "../processed"
train_dir = os.path.join(processed_dir, "train")
test_dir = os.path.join(processed_dir, "test")
val_dir = os.path.join(processed_dir, "val")
final_classes = "../metadata/final_classes.csv"

os.makedirs(train_dir,exist_ok=True)
os.makedirs(test_dir,exist_ok=True)
os.makedirs(val_dir,exist_ok=True)

class_df = pd.read_csv(final_classes)

def find_images(base_path,class_name):
    """
        Finds all image files in the specified class directory.
    """

    class_path = os.path.join(base_path,class_name)

    if not os.path.exists(class_path):
        return []

    image_files = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg",".png"))]

    return image_files


for _,row in class_df.iterrows():
    class_name = row["class_name"]
    images = find_images(processed_dir,class_name)
    random.shuffle(images)
    total_images = len(images)

    # Creating target subdirectories for each class
    train_class_dir = os.path.join(train_dir,class_name)
    test_class_dir = os.path.join(test_dir,class_name)
    val_class_dir = os.path.join(val_dir,class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)


    #split --> %70 train / %15 val / %15 test
    train_count = int(total_images*0.7)
    val_count = int(total_images*0.15)
    test_count = total_images - train_count - val_count

    # Splitting the lists
    train_images = images[0:train_count]
    val_images = images[train_count:train_count+val_count]
    test_images = images[train_count+val_count:]

    #Copying the files
    for img in train_images:
        src_path = os.path.join(processed_dir, class_name, img)
        dst_path = os.path.join(train_class_dir, img)
        shutil.copy2(src_path, dst_path)

    for img in val_images:
        src_path = os.path.join(processed_dir, class_name, img)
        dst_path = os.path.join(val_class_dir, img)
        shutil.copy2(src_path, dst_path)

    for img in test_images:
        src_path = os.path.join(processed_dir, class_name, img)
        dst_path = os.path.join(test_class_dir, img)
        shutil.copy2(src_path, dst_path)

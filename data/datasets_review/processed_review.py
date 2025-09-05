import os
import pandas as pd

processed_dir = "../processed"
class_dist = "../metadata/class_distribution.csv"

data = []

for class_name in os.listdir(processed_dir):
    class_path = os.path.join(processed_dir,class_name)
    if os.path.isdir(class_path):
        num_images = len([f for f in os.listdir(class_path) if f.lower().endswith((".jpg",".png"))])
        data.append((class_name,num_images))

df = pd.DataFrame(data,columns=["class_name","num_images"])
df.to_csv(class_dist,index=False,encoding="utf-8")

print(f"Class Distribution was saved: {class_dist}")

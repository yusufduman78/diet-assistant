import os
import pandas as pd

mapping_file = "../metadata/class_mapping.csv"
finalclass_file = "../metadata/final_classes.csv"

mapping_pd = pd.read_csv(mapping_file)

#group by final class, then get, sort, and join unique source dataset names for each group
groups = mapping_pd.groupby("last_class")["dataset"].apply(lambda x: ";".join(sorted(set(x)))).reset_index()

#rename columns
groups = groups.reset_index().rename(columns={
    "index": "class_id",
    "last_class": "class_name",
    "dataset": "source_dataset"
})

os.makedirs(os.path.dirname(finalclass_file), exist_ok=True)
groups.to_csv(finalclass_file,index=False,encoding="utf-8")

print(f"Final class list saved: {finalclass_file}")

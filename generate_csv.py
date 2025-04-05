import os
import pandas as pd

def generate_csv(data_dir, output_csv):
    data = []
    classes = sorted(os.listdir(data_dir))  # Folder names are the class names

    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                relative_path = f"{class_name}/{file_name}"
                data.append([relative_path, label_idx])

    df = pd.DataFrame(data, columns=["filename", "category"])
    df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv} with {len(df)} entries.")

# Paths
base_dir = r"C:\Users\Sandhya\Deep_Learning\DL_Practice\Flower_Dataset"
generate_csv(os.path.join(base_dir, "train"), os.path.join(base_dir, "train.csv"))
generate_csv(os.path.join(base_dir, "val"), os.path.join(base_dir, "val.csv"))

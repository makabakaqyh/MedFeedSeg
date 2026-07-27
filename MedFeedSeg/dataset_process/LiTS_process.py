import os
import shutil
import random
import math
import pandas as pd
import cv2
import numpy as np


def split_lits_dataset(
    csv_path,
    source_root,
    target_dir,
    seed=666,
    val_ratio=0.15,
    test_ratio=0.15,
    mask_type="liver"
):
    """
    Split the LiTS dataset by study number.

    Args:
        csv_path: Path to lits_df.csv.
        source_root: Root directory of the LiTS PNG dataset (e.g., ../../datasets/lits-png/dataset_6).
        target_dir: Output directory (e.g., ../../datasets/LiTS).
        seed: Random seed for reproducible splitting.
        val_ratio: Validation set ratio.
        test_ratio: Test set ratio.
        mask_type: "tumor" to use tumor masks, or "liver" to use liver masks.
    """
    random.seed(seed)

    df = pd.read_csv(csv_path)

    # Remove slices containing only background
    df = df[(df["liver_mask_empty"])]
    # df = df[(df["tumor_mask_empty"])]

    if mask_type == "tumor":
        mask_col = "tumor_maskpath"
    elif mask_type == "liver":
        mask_col = "liver_maskpath"
    else:
        raise ValueError("mask_type must be either 'tumor' or 'liver'")

    # Create output directories
    dirs = {
        "train": {
            "image": os.path.join(target_dir, "train_set", "image"),
            "label": os.path.join(target_dir, "train_set", "label")
        },
        "val": {
            "image": os.path.join(target_dir, "val_set", "image"),
            "label": os.path.join(target_dir, "val_set", "label")
        },
        "test": {
            "image": os.path.join(target_dir, "test_set", "image"),
            "label": os.path.join(target_dir, "test_set", "label")
        }
    }

    for split in dirs.values():
        os.makedirs(split["image"], exist_ok=True)
        os.makedirs(split["label"], exist_ok=True)

    # Split the dataset by study rather than by slice
    studies = df["study_number"].unique().tolist()
    random.shuffle(studies)

    total = len(studies)
    val_num = math.floor(total * val_ratio)
    test_num = math.floor(total * test_ratio)
    train_num = total - val_num - test_num

    train_studies = studies[:train_num]
    val_studies = studies[train_num:train_num + val_num]
    test_studies = studies[train_num + val_num:]

    split_map = {
        "train": train_studies,
        "val": val_studies,
        "test": test_studies
    }

    for split_name, study_list in split_map.items():
        split_df = df[df["study_number"].isin(study_list)]

        for _, row in split_df.iterrows():
            image_name = os.path.basename(row["filepath"])
            label_name = os.path.basename(row[mask_col])

            src_image = os.path.join(source_root, image_name)
            src_label = os.path.join(source_root, label_name)

            dst_image = os.path.join(dirs[split_name]["image"], image_name)
            dst_label = os.path.join(dirs[split_name]["label"], label_name)

            if os.path.exists(src_image) and os.path.exists(src_label):
                # Copy the original image
                shutil.copy2(src_image, dst_image)

                # Convert the mask to binary values (0 or 255)
                mask = cv2.imread(src_label, cv2.IMREAD_GRAYSCALE)
                mask = (mask > 0).astype(np.uint8) * 255
                cv2.imwrite(dst_label, mask)

                # shutil.copy2(src_image, dst_image)
                # shutil.copy2(src_label, dst_label)
            else:
                print("File not found:", src_image, src_label)

        print(
            f"{split_name}: "
            f"{len(study_list)} studies, "
            f"{len(split_df)} slices"
        )


if __name__ == "__main__":
    csv_path = os.path.join("..", "..", "datasets", "lits_df.csv")
    source_root = os.path.join("..", "..", "datasets", "dataset_6", "dataset_6")
    target_dir = os.path.join("..", "..", "datasets", "LiTS")

    split_lits_dataset(
        csv_path=csv_path,
        source_root=source_root,
        target_dir=target_dir,
        seed=666,
        mask_type="liver"
    )

import os
import random
import shutil
from pathlib import Path


# ======================================================= Basic Configuration =======================================================
# Path to the original PolypGen dataset
src_root = Path("..") / ".." / "datasets" / "polypgen" / "PolypGen2021_MultiCenterData_v3"

# Output directory for the processed dataset
dst_root = Path("..") / ".." / "datasets" / "PolypGen"

# Dataset split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

seed = 666  # Fixed random seed for reproducible splits
random.seed(seed)

# ================================================== Collect Image-Mask Pairs ==================================================
pairs = []  # Store all image-mask pairs
skip = 0  # Number of skipped samples with missing masks

# Iterate through the six PolypGen centers (data_C1 to data_C6)
for c in range(1, 7):
    # Image directory of the current center
    image_dir = src_root / f"data_C{c}" / f"images_C{c}"

    # Mask directory of the current center
    mask_dir = src_root / f"data_C{c}" / f"masks_C{c}"

    # Get all image files in the current center
    image_files = sorted(list(image_dir.glob("*.jpg")))

    # Iterate through each image
    for img_path in image_files:
        # Filename without extension
        stem = img_path.stem

        # Find the corresponding mask
        mask_path = mask_dir / f"{stem}_mask.jpg"

        if mask_path.exists():
            # Store the image, mask, and center ID
            pairs.append((img_path, mask_path, c))
        else:
            skip += 1

# Print the number of valid image-mask pairs
print(f"Found {len(pairs)} image-mask pairs, skipped {skip} invalid samples.")

# ======================================================= Random Split =======================================================
# Shuffle all image-mask pairs
random.shuffle(pairs)

n_total = len(pairs)  # Total number of samples
n_train = int(n_total * train_ratio)  # Number of training samples
n_val = int(n_total * val_ratio)  # Number of validation samples

# Split the dataset into training, validation, and test sets
train_pairs = pairs[:n_train]
val_pairs = pairs[n_train:n_train + n_val]
test_pairs = pairs[n_train + n_val:]

# Store dataset splits in a dictionary
splits = {
    "train_set": train_pairs,
    "val_set": val_pairs,
    "test_set": test_pairs,
}

# ======================================================= Copy Data =======================================================
# Create output directories
for split in splits:
    (dst_root / split / "image").mkdir(parents=True, exist_ok=True)
    (dst_root / split / "label").mkdir(parents=True, exist_ok=True)

# Copy images and masks for each dataset split
for split_name, split_pairs in splits.items():
    # Iterate through all samples in the current split
    for img_path, mask_path, center_id in split_pairs:
        # New filename
        new_name = f"C{center_id}_{img_path.name}"

        # Destination paths
        dst_img = dst_root / split_name / "image" / new_name
        dst_mask = dst_root / split_name / "label" / new_name

        # Copy the image and mask
        shutil.copy2(img_path, dst_img)
        shutil.copy2(mask_path, dst_mask)

    print(f"{split_name}: {len(split_pairs)} samples")

print("Processing completed!")
print(f"Output directory: {dst_root}")
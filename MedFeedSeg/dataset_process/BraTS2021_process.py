import random
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from tqdm import tqdm


def normalize_to_uint8(image_slice):
    """
    Normalize a single MRI slice to the range [0, 255].
    """
    image_slice = image_slice.astype(np.float32)
    # Remove extreme bright outliers
    nonzero = image_slice[image_slice > 0]
    if nonzero.size > 0:
        low, high = np.percentile(nonzero, [1, 99])
        image_slice = np.clip(image_slice, low, high)

    min_val = image_slice.min()
    max_val = image_slice.max()

    if max_val - min_val < 1e-8:
        image_slice = np.zeros_like(image_slice, dtype=np.uint8)
    else:
        image_slice = (image_slice - min_val) / (max_val - min_val)
        image_slice = (image_slice * 255).astype(np.uint8)

    return image_slice


def slice_and_save(cases, save_path, save_only_tumor_slices=True):
    """
    Args:
        cases: List of case directories.
        save_path: Output directory.
        save_only_tumor_slices: Whether to save only slices containing tumors.
    Returns:
        num_slices: Number of saved slices.
    """
    num_slices = 0  # Number of saved slices
    for case in tqdm(cases):
        flair_path = case / f"{case.name}_flair.nii.gz"  # Path to the image volume
        seg_path = case / f"{case.name}_seg.nii.gz"  # Path to the segmentation label
        flair = nib.load(str(flair_path)).get_fdata()  # Load image volume as a NumPy array
        seg = nib.load(str(seg_path)).get_fdata()  # Load segmentation label as a NumPy array
        assert flair.shape == seg.shape, f"Shape mismatch: {case.name}"  # Ensure image and label have the same shape
        for i in range(flair.shape[2]):  # Iterate through all slices
            image_slice = normalize_to_uint8(flair[:, :, i])
            label_slice = (seg[:, :, i] > 0).astype(np.uint8) * 255
            # Skip slices without tumors if only tumor slices are required
            if save_only_tumor_slices and np.max(label_slice) == 0:
                continue
            # Save the image slice and corresponding label
            file_name = f"{case.name}_slice_{i:03d}.png"  # Slice filename
            image_save_path = save_path / "image" / file_name
            label_save_path = save_path / "label" / file_name
            cv2.imwrite(str(image_save_path), image_slice)
            cv2.imwrite(str(label_save_path), label_slice)
            num_slices += 1

    return num_slices


def split_dataset(source_dir, target_dir, seed=666):
    '''
    Split the dataset into training, validation, and test sets.

    Args:
        source_dir: Path to the source dataset.
        target_dir: Path to the output dataset.
        seed: Random seed for reproducible splitting.
    '''
    # Set the random seed to ensure reproducible splits
    random.seed(seed)

    # Collect valid case directories
    cases = []
    for item in source_dir.iterdir():  # Iterate through all files and subdirectories
        if item.is_dir() and item.name.startswith("BraTS2021_"):  # Select valid BraTS case directories
            flair_file = item / f"{item.name}_flair.nii.gz"  # Image file path ("/" works on both Linux and Windows)
            seg_file = item / f"{item.name}_seg.nii.gz"  # Label file path ("/" works on both Linux and Windows)
            if flair_file.exists() and seg_file.exists():
                cases.append(item)

    # Split by case to avoid slice-level data leakage
    total = len(cases)
    val_num = int(0.15 * total)  # 15%
    test_num = int(0.15 * total)  # 15%
    train_num = total - test_num - val_num  # 70%
    cases = sorted(cases)  # Sort first to ensure consistent ordering across platforms
    random.shuffle(cases)  # Shuffle the cases
    train_cases = cases[:train_num]  # Training cases
    val_cases = cases[train_num:train_num + val_num]  # Validation cases
    test_cases = cases[train_num + val_num:]  # Test cases

    # Slice the volumes and save them
    train_set_path = target_dir / "train_set"
    val_set_path = target_dir / "val_set"
    test_set_path = target_dir / "test_set"
    train_slices = slice_and_save(train_cases, train_set_path)
    val_slices = slice_and_save(val_cases, val_set_path)
    test_slices = slice_and_save(test_cases, test_set_path)

    print(f"total cases:{total}, train cases:{train_num}, val cases:{val_num}, test cases:{test_num}")
    print(f"train slices:{train_slices}, val slices:{val_slices}, test slices:{test_slices}")


if __name__ == "__main__":
    # Path configuration
    dataset_path = Path("../../datasets/BraTS")  # Path to the original dataset
    target_path = Path("../../datasets/BraTS_2021")  # Output directory

    # Create output directories
    for split in ["train_set", "val_set", "test_set"]:
        (target_path / split / "image").mkdir(parents=True, exist_ok=True)
        (target_path / split / "label").mkdir(parents=True, exist_ok=True)

    # Split the dataset
    split_dataset(dataset_path, target_path)
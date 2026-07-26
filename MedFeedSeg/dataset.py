import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

import utils
import config


class QaTa_COV19_v2(Dataset):
    """
    QaTa_COV19_v2 dataset.

    Image shape: [224, 224, 3], pixel value range: [0, n]
    Label shape: [224, 224, 3], pixel value range: {0, 255}
    """
    def __init__(self, dataset_path, mode):
        """
        Initialize the dataset.

        Args:
            dataset_path: Dataset directory containing image and label folders.
            mode: Dataset split ('train', 'val', or 'test').
        """
        super().__init__()
        self.mode = mode

        # Dataset paths
        if mode == 'train':
            self.image_path = os.path.join(dataset_path, 'train_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'train_set', 'label')  # Label directory
        elif mode == 'val':
            self.image_path = os.path.join(dataset_path, 'val_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'val_set', 'label')  # Label directory
        elif mode == 'test':
            self.image_path = os.path.join(dataset_path, 'test_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'test_set', 'label')  # Label directory
        else:
            raise ValueError(f"mode must be 'train', 'val' or 'test', got {mode}")

        self.image_filenames = os.listdir(self.image_path)  # List of image filenames

    def __len__(self):
        """
        Return the size of the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, item):
        """
        Return a sample by index.

        Args:
            item: Sample index.

        Returns:
            sample:
                {
                    'image': image [3, 256, 256],
                    'label': label [1, 256, 256],
                    'sample_filename': filename
                }
        """
        # Process filenames
        image_filename = self.image_filenames[item]  # Image filename corresponding to the given index
        label_filename = 'mask_' + image_filename  # Corresponding label filename

        # Load and preprocess the image
        image = cv2.imread(os.path.join(self.image_path, image_filename))  # Returns a NumPy array with shape [H, W, C]
        image = cv2.resize(image, config.image_size, interpolation=cv2.INTER_LINEAR)  # Resize to 256 × 256
        image = torch.from_numpy(image)  # Convert the NumPy array to a tensor
        image = image / 255  # Normalize pixel values to [0, 1]
        image = image.permute(2, 0, 1)  # Convert to [C, H, W]

        # Load and preprocess the label
        label = cv2.imread(os.path.join(self.label_path, label_filename))  # Returns a NumPy array with shape [H, W, C]
        label = cv2.resize(label, config.image_size, interpolation=cv2.INTER_NEAREST)  # Resize to 256 × 256
        label = torch.from_numpy(label)  # Convert the NumPy array to a tensor
        label = label / 255  # Normalize pixel values to [0, 1]
        label = label.permute(2, 0, 1)  # Convert to [C, H, W]

        # Construct the sample dictionary
        sample = {
            'image': image,
            'label': label[:1, :, :],
            'sample_filename': image_filename
        }

        return sample


class BUSI(Dataset):
    """
    BUSI dataset.

    Image shape: [H, W, 3], pixel value range: [0, n]
    Label shape: [H, W, 3], pixel value range: {0, 255}
    """
    def __init__(self, dataset_path, mode):
        """
        Initialize the dataset.

        Args:
            dataset_path: Dataset directory containing image and label folders.
            mode: Dataset split ('train', 'val', or 'test').
        """
        super().__init__()
        self.mode = mode

        # Dataset paths
        if mode == 'train':
            self.image_path = os.path.join(dataset_path, 'train_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'train_set', 'label')  # Label directory
        elif mode == 'val':
            self.image_path = os.path.join(dataset_path, 'val_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'val_set', 'label')  # Label directory
        elif mode == 'test':
            self.image_path = os.path.join(dataset_path, 'test_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'test_set', 'label')  # Label directory
        else:
            raise ValueError(f"mode must be 'train', 'val' or 'test', got {mode}")

        self.label_filenames = os.listdir(self.label_path)  # List of label filenames

    def __len__(self):
        """
        Return the size of the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.label_filenames)

    def _data_augment(self, image, label):
        """
        Apply data augmentation to the BUSI dataset while ensuring
        that the image and label undergo the same geometric transformations.

        Args:
            image: NumPy array of shape [H, W, C], uint8.
            label: NumPy array of shape [H, W, C], uint8.

        Returns:
            Augmented image and label.
        """
        h, w = image.shape[:2]

        # 1. Random horizontal flip
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        # 2. Random vertical flip
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)

        # 3. Random rotation
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-15, 15)
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            image = cv2.warpAffine(
                image, matrix, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            label = cv2.warpAffine(
                label, matrix, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        # 4. Random brightness and contrast adjustment (image only)
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-20, 20)  # Brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # 5. Random Gaussian noise (image only)
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 8, image.shape).astype(np.float32)
            image = image.astype(np.float32) + noise
            image = np.clip(image, 0, 255).astype(np.uint8)

        # 6. Random Gaussian blur (image only)
        if np.random.rand() < 0.2:
            image = cv2.GaussianBlur(image, (3, 3), 0)

        return image, label

    def __getitem__(self, item):
        """
        Return a sample by index.

        Args:
            item: Sample index.

        Returns:
            sample:
                {
                    'image': image [3, 256, 256],
                    'label': label [1, 256, 256],
                    'sample_filename': filename
                }
        """
        # Process filenames
        label_filename = self.label_filenames[item]  # Label filename corresponding to the given index
        image_filename = label_filename.replace('_mask', '')  # Corresponding image filename

        # Load the image and label
        image = cv2.imread(os.path.join(self.image_path, image_filename))  # Returns a NumPy array with shape [H, W, C]
        label = cv2.imread(os.path.join(self.label_path, label_filename))  # Returns a NumPy array with shape [H, W, C]

        # Apply data augmentation during training
        if self.mode == 'train':
            image, label = self._data_augment(image, label)

        # Preprocess the image by resizing and padding to [256, 256, C]
        image = utils.resize_and_pad(image, target_size=config.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image)
        image = image / 255
        image = image.permute(2, 0, 1)

        # Preprocess the label by resizing and padding to [256, 256, C]
        label = utils.resize_and_pad(label, target_size=config.image_size, interpolation=cv2.INTER_NEAREST)
        label = torch.from_numpy(label)
        label = label / 255
        label = label.permute(2, 0, 1)

        # Construct the sample dictionary
        sample = {
            'image': image,
            'label': label[:1, :, :],
            'sample_filename': image_filename
        }

        return sample


class BraTS_2021(Dataset):
    """
    BraTS_2021 dataset.

    Image shape: [240, 240, 3], pixel value range: [0, n]
    Label shape: [240, 240, 3], pixel value range: {0, 255}
    """
    def __init__(self, dataset_path, mode):
        """
        Initialize the dataset.

        Args:
            dataset_path: Dataset directory containing image and label folders.
            mode: Dataset split ('train', 'val', or 'test').
        """
        super().__init__()
        self.mode = mode

        # Dataset paths
        if mode == 'train':
            self.image_path = os.path.join(dataset_path, 'train_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'train_set', 'label')  # Label directory
        elif mode == 'val':
            self.image_path = os.path.join(dataset_path, 'val_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'val_set', 'label')  # Label directory
        elif mode == 'test':
            self.image_path = os.path.join(dataset_path, 'test_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'test_set', 'label')  # Label directory
        else:
            raise ValueError(f"mode must be 'train', 'val' or 'test', got {mode}")

        self.image_filenames = os.listdir(self.image_path)  # List of image filenames

    def __len__(self):
        """
        Return the size of the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, item):
        """
        Return a sample by index.

        Args:
            item: Sample index.

        Returns:
            sample:
                {
                    'image': image [3, 256, 256],
                    'label': label [1, 256, 256],
                    'sample_filename': filename
                }
        """
        # Process filenames
        image_filename = self.image_filenames[item]  # Image filename corresponding to the given index
        label_filename = image_filename  # Corresponding label filename

        # Load and preprocess the image
        image = cv2.imread(os.path.join(self.image_path, image_filename))  # Returns a NumPy array with shape [H, W, C]
        image = cv2.resize(image, config.image_size, interpolation=cv2.INTER_LINEAR)  # Resize to 256 × 256
        image = torch.from_numpy(image)  # Convert the NumPy array to a tensor
        image = image / 255  # Normalize pixel values to [0, 1]
        image = image.permute(2, 0, 1)  # Convert to [C, H, W]

        # Load and preprocess the label
        label = cv2.imread(os.path.join(self.label_path, label_filename))  # Returns a NumPy array with shape [H, W, C]
        label = cv2.resize(label, config.image_size, interpolation=cv2.INTER_NEAREST)  # Resize to 256 × 256
        label = torch.from_numpy(label)  # Convert the NumPy array to a tensor
        label = label / 255  # Normalize pixel values to [0, 1]
        label = label.permute(2, 0, 1)  # Convert to [C, H, W]

        # Construct the sample dictionary
        sample = {
            'image': image,
            'label': label[:1, :, :],
            'sample_filename': image_filename
        }

        return sample


class LiTS(Dataset):
    """
    LiTS dataset.

    Image shape: [256, 256, 3], pixel value range: [0, n]
    Label shape: [256, 256, 3], pixel value range: {0, 255}
    """
    def __init__(self, dataset_path, mode):
        """
        Initialize the dataset.

        Args:
            dataset_path: Dataset directory containing image and label folders.
            mode: Dataset split ('train', 'val', or 'test').
        """
        super().__init__()
        self.mode = mode

        # Dataset paths
        if mode == 'train':
            self.image_path = os.path.join(dataset_path, 'train_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'train_set', 'label')  # Label directory
        elif mode == 'val':
            self.image_path = os.path.join(dataset_path, 'val_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'val_set', 'label')  # Label directory
        elif mode == 'test':
            self.image_path = os.path.join(dataset_path, 'test_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'test_set', 'label')  # Label directory
        else:
            raise ValueError(f"mode must be 'train', 'val' or 'test', got {mode}")

        self.image_filenames = os.listdir(self.image_path)  # List of image filenames

    def __len__(self):
        """
        Return the size of the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.image_filenames)

    def __getitem__(self, item):
        """
        Return a sample by index.

        Args:
            item: Sample index.

        Returns:
            sample:
                {
                    'image': image [3, 256, 256],
                    'label': label [1, 256, 256],
                    'sample_filename': filename
                }
        """
        # Process filenames
        image_filename = self.image_filenames[item]  # Image filename corresponding to the given index

        # Generate the corresponding label filename
        label_filename = image_filename.replace("volume-", "")
        case_id, slice_id = label_filename.replace(".png", "").split("_")
        label_filename = (
            f"segmentation-{case_id}_livermask_{slice_id}.png"
        )

        # Load and preprocess the image
        image = cv2.imread(os.path.join(self.image_path, image_filename))  # Returns a NumPy array with shape [H, W, C]
        image = cv2.resize(image, config.image_size, interpolation=cv2.INTER_LINEAR)  # Resize to 256 × 256
        image = torch.from_numpy(image)  # Convert the NumPy array to a tensor
        image = image / 255  # Normalize pixel values to [0, 1]
        image = image.permute(2, 0, 1)  # Convert to [C, H, W]

        # Load and preprocess the label
        label = cv2.imread(os.path.join(self.label_path, label_filename))  # Returns a NumPy array with shape [H, W, C]
        label = cv2.resize(label, config.image_size, interpolation=cv2.INTER_NEAREST)  # Resize to 256 × 256
        label = torch.from_numpy(label)  # Convert the NumPy array to a tensor
        label = label / 255  # Normalize pixel values to [0, 1]
        label = label.permute(2, 0, 1)  # Convert to [C, H, W]

        # Construct the sample dictionary
        sample = {
            'image': image,
            'label': label[:1, :, :],
            'sample_filename': image_filename
        }

        return sample


class PolypGen(Dataset):
    """
    PolypGen dataset.

    Image shape: [H, W, 3], pixel value range: [0, n]
    Label shape: [H, W, 3], pixel value ranges: [0, a] and [b, 255]
    """
    def __init__(self, dataset_path, mode):
        """
        Initialize the dataset.

        Args:
            dataset_path: Dataset directory containing image and label folders.
            mode: Dataset split ('train', 'val', or 'test').
        """
        super().__init__()
        self.mode = mode

        # Dataset paths
        if mode == 'train':
            self.image_path = os.path.join(dataset_path, 'train_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'train_set', 'label')  # Label directory
        elif mode == 'val':
            self.image_path = os.path.join(dataset_path, 'val_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'val_set', 'label')  # Label directory
        elif mode == 'test':
            self.image_path = os.path.join(dataset_path, 'test_set', 'image')  # Image directory
            self.label_path = os.path.join(dataset_path, 'test_set', 'label')  # Label directory
        else:
            raise ValueError(f"mode must be 'train', 'val' or 'test', got {mode}")
        self.image_filenames = os.listdir(self.image_path)  # List of image filenames

    def __len__(self):
        """
        Return the size of the dataset.

        Returns:
            Number of samples in the dataset.
        """
        return len(self.image_filenames)  # Dataset size equals the number of images

    def _data_augment(self, image, label):
        """
        Apply data augmentation to the PolypGen dataset while ensuring
        that the image and label undergo the same geometric transformations.

        Args:
            image: NumPy array of shape [H, W, C], uint8.
            label: NumPy array of shape [H, W, C], uint8.

        Returns:
            image, label: Augmented image and label.
        """
        h, w = image.shape[:2]

        # 1. Random horizontal flip
        if np.random.rand() < 0.5:
            image = cv2.flip(image, 1)
            label = cv2.flip(label, 1)

        # 2. Random vertical flip
        if np.random.rand() < 0.3:
            image = cv2.flip(image, 0)
            label = cv2.flip(label, 0)

        # 3. Random rotation
        if np.random.rand() < 0.5:
            angle = np.random.uniform(-15, 15)
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

            image = cv2.warpAffine(
                image, matrix, (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

            label = cv2.warpAffine(
                label, matrix, (w, h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        # 4. Random brightness and contrast adjustment (image only)
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.8, 1.2)  # Contrast
            beta = np.random.uniform(-20, 20)  # Brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

        # 5. Random Gaussian noise (image only)
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 8, image.shape).astype(np.float32)
            image = image.astype(np.float32) + noise
            image = np.clip(image, 0, 255).astype(np.uint8)

        # 6. Random Gaussian blur (image only)
        if np.random.rand() < 0.2:
            image = cv2.GaussianBlur(image, (3, 3), 0)

        return image, label

    def __getitem__(self, item):
        """
        Return a sample by index.

        Args:
            item: Sample index.

        Returns:
            sample: Sample corresponding to the given index:
                {
                    'image': image [3, 256, 256],
                    'label': label [1, 256, 256],
                    'sample_filename': filename
                }
        """
        # Process filenames
        image_filename = self.image_filenames[item]  # Image filename corresponding to the given index
        label_filename = image_filename  # Corresponding label filename

        # Load the image and label
        image = cv2.imread(os.path.join(self.image_path, image_filename))  # Returns a NumPy array with shape [H, W, C]
        label = cv2.imread(os.path.join(self.label_path, label_filename))  # Returns a NumPy array with shape [H, W, C]

        # Apply data augmentation during training
        if self.mode == 'train':
            image, label = self._data_augment(image, label)

        # Preprocess the image by resizing and padding to [256, 256, C]
        image = utils.resize_and_pad(image, target_size=config.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.from_numpy(image)  # Convert the NumPy array to a tensor
        image = image / 255  # Normalize pixel values to [0, 1]
        image = image.permute(2, 0, 1)  # Convert to [C, H, W]

        # Preprocess the label by resizing and padding to [256, 256, C]
        label = utils.resize_and_pad(label, target_size=config.image_size, interpolation=cv2.INTER_NEAREST)
        label = torch.from_numpy(label)  # Convert the NumPy array to a tensor
        label = label / 255  # Normalize pixel values to [0, 1]
        label = (label > 0.5).float()  # Binarize the label
        label = label.permute(2, 0, 1)  # Convert to [C, H, W]

        # Construct the sample dictionary
        sample = {
            'image': image,
            'label': label[:1, :, :],
            'sample_filename': image_filename
        }

        return sample
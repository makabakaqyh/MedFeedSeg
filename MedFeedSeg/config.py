import os
import torch

# ======== Basic Configuration ========
resume_session = ''  # Leave empty to start training from scratch; otherwise, specify the previous session name to resume training.

# Task name (determines the dataset used for training)
# QaTa-COV19-v2 / BraTS_2021 / LiTS / PolypGen
task_name = 'QaTa-COV19-v2'

# Input image size
image_size = (256, 256)

# Model type (determines the model used for training)
# UNet / DeepLabV3plus / SAM / MedFeedSeg
model_type = 'MedFeedSeg'

# Session name (determines the directory for saving checkpoints)
session_name = 'MedFeedSeg'

# GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = '3, 4, 5'
num_device = torch.cuda.device_count()  # Number of available GPUs

# ======== Model Configuration ========
seed = 112161  # Random seed

text_token = 64  # Number of tokens for the text embedding
text_dims = 768  # Dimension of the text embedding

batch_size = 16
train_batch_size = batch_size * num_device  # Training batch size (multi-GPU)
valid_batch_size = batch_size * num_device  # Validation batch size (multi-GPU)
test_batch_size = 2 * batch_size * num_device  # Test batch size (multi-GPU)

learning_rate = 0.0003  # Initial learning rate
min_epochs = 20  # Minimum number of training epochs
max_epochs = 200  # Maximum number of training epochs
early_stopping_patience = 20  # Number of consecutive epochs without improvement before early stopping

# ======== Dataset Configuration ========
dataset_path = os.path.join('..', 'datasets', task_name)  # Dataset directory

# ======== Pretrained Model Configuration ========
text_encoder_path = os.path.join(
    '..', 'pretrained_model', 'BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'
)  # Pretrained text encoder

image_encoder_path = os.path.join(
    '..', 'pretrained_model', 'convnext-tiny-224'
)  # Pretrained image encoder

llm_path = os.path.join(
    '..', 'pretrained_model', 'Qwen3-4B-Instruct-2507'
)  # Large language model

# ======== Saving Configuration ========
save_path = os.path.join('..', 'saved_model', task_name)  # Output directory for the current task
os.makedirs(save_path, exist_ok=True)  # Create the directory if it does not exist

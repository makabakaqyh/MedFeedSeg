import random
import numpy as np
import os
import torch
import torch.nn as nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
from torchmetrics.classification import Accuracy, BinaryJaccardIndex
from torchmetrics.segmentation import DiceScore
from monai.losses import DiceCELoss

import config
import utils
from dataset import QaTa_COV19_v2, BUSI, BraTS_2021, LiTS, PolypGen
from network.UNet import UNet
from network.SAM import build_sam_vit_b
from network.ScribblePromptUNet.network import ScribblePromptUNet
from network.ScribblePromptUNet.scribbleprompt_input import convert_to_grayscale, build_scribbleprompt_input
from network.IMISNet.build_model import build_imisnet, build_imis_prompt
from network.MedFeedSeg import MedFeedSeg


def train_one_epoch(device, epoch, loader, model, criterion, optimizer, lr_scheduler, train_logger, scaler,
                    task_name, tokenizer=None, llm=None):
    """
    Train the model for one epoch over the entire training set.

    Args:
        device: Computation device.
        epoch: Current epoch.
        loader: Data loader.
        model: Neural network model.
        criterion: Loss function.
        optimizer: Optimizer.
        lr_scheduler: Learning rate scheduler.
        train_logger: Training logger.
        scaler: Gradient scaler for mixed-precision training.
        task_name: Task name.
        tokenizer: Tokenizer.
        llm: Large language model used to convert structured prompts into natural language prompts.
    """
    # Initialize evaluation metrics
    accuracy = Accuracy(task='binary').to(device)
    dice = DiceScore(num_classes=2, include_background=False, average="micro", input_format="one-hot").to(device)
    miou = BinaryJaccardIndex().to(device)

    # Iterate over the data loader (starting from index 1)
    for i, batch_sample in enumerate(loader, 1):  # Retrieve the batch index and the current batch
        # Unpack images and labels from the current batch
        images, labels = batch_sample['image'], batch_sample['label']
        # Move all tensors to the target device
        images, labels = images.to(device), labels.to(device)

        # =================================================== Forward Pass ===================================================
        if config.model_type == 'UNet':
            with torch.amp.autocast("cuda"):  # Mixed-precision training
                logits = model(images)
                loss = criterion(logits, labels)

        elif config.model_type == 'SAM':
            # First forward pass
            with torch.amp.autocast("cuda"):  # Mixed-precision training
                logits, low_logits, image_embeddings = model(
                    multimask_output=False,
                    images=images,
                    point_prompts=None,
                    low_logits=None,
                    image_embeddings=None
                )
                loss1 = criterion(logits, labels)

            # Generate prompts
            with torch.no_grad():
                masks = (torch.sigmoid(logits) > 0.5)
                text_prompts, point_prompts = utils.create_prompt(labels, masks, task_name)

            # Second forward pass
            low_logits = low_logits.detach()
            image_embeddings = image_embeddings.detach()
            with torch.amp.autocast("cuda"):  # Mixed-precision training
                logits, low_logits, _ = model(
                    multimask_output=False,
                    images=images,
                    point_prompts=point_prompts,
                    low_logits=low_logits,
                    image_embeddings=image_embeddings
                )
                loss2 = criterion(logits, labels)

            loss = loss1 + loss2

        elif config.model_type == "ScribblePromptUNet":
            # The official ScribblePrompt-UNet uses single-channel input
            sp_images = convert_to_grayscale(images)
            # First forward pass
            first_input = build_scribbleprompt_input(images=sp_images, point_prompts=None, mask_input=None,)
            with torch.amp.autocast("cuda"):
                logits = model(first_input)
                loss1 = criterion(logits, labels)
            # Generate corrective prompts
            with torch.no_grad():
                masks = torch.sigmoid(logits) > 0.5
                _, point_prompts = utils.create_prompt(labels, masks, task_name)
            # Second forward pass
            second_input = build_scribbleprompt_input(images=sp_images, point_prompts=point_prompts,
                                                      mask_input=logits.detach())
            with torch.amp.autocast("cuda"):
                logits = model(second_input)
                loss2 = criterion(logits, labels)

            loss = loss1 + loss2

        elif config.model_type == "IMISNet":
            # Generate the initial prompt, as IMISNet requires an initial target prompt
            # for its first prediction
            with torch.no_grad():
                empty_masks = torch.zeros_like(labels, dtype=torch.bool)
                _, point_prompts = utils.create_prompt(labels, empty_masks, task_name)
                first_prompts = build_imis_prompt(point_prompts=point_prompts, mask_input=None)
            # First interaction
            with torch.amp.autocast("cuda"):
                outputs, image_embeddings = model(images=images, prompts=first_prompts, image_embeddings=None)
                logits = outputs["masks"]
                low_logits = outputs["low_res_masks"]
                loss1 = criterion(logits, labels)
            # Generate corrective prompts
            with torch.no_grad():
                masks = torch.sigmoid(logits) > 0.5
                _, point_prompts = utils.create_prompt(labels, masks, task_name)
                second_prompts = build_imis_prompt(point_prompts=point_prompts, mask_input=low_logits)
            # Second interaction
            with torch.amp.autocast("cuda"):
                outputs, _ = model(images=None, prompts=second_prompts, image_embeddings=image_embeddings.detach())
                logits = outputs["masks"]
                loss2 = criterion(logits, labels)
            loss = loss1 + loss2

        elif config.model_type == 'MedFeedSeg':
            # First forward pass
            with torch.amp.autocast("cuda"):  # Mixed-precision training
                logits, low_logits, image1, image2, image3, image4 = model(images)
                loss1 = criterion(logits, labels)

            # Generate prompts
            with torch.no_grad():
                masks = (torch.sigmoid(logits) > 0.5)
                text_prompts, point_prompts = utils.create_prompt(labels, masks, task_name)
                text_prompts = llm.evaluate_batch(text_prompts)
                text_prompts = tokenizer.tokenize(text_prompts)
                text_prompts = {
                    'input_ids': text_prompts['input_ids'].to(device),
                    'attention_mask': text_prompts['attention_mask'].to(device)
                }

            # Second forward pass
            low_logits = low_logits.detach()
            image1 = image1.detach()
            image2 = image2.detach()
            image3 = image3.detach()
            image4 = image4.detach()

            with torch.amp.autocast("cuda"):  # Mixed-precision training
                logits, low_logits, _, _, _, _ = model(
                    images,
                    text_prompts,
                    point_prompts,
                    low_logits,
                    image1,
                    image2,
                    image3,
                    image4
                )
                loss2 = criterion(logits, labels)

            loss = loss1 + loss2

        # ====================================== Backward Propagation and Optimization (Mixed Precision) ======================================
        optimizer.zero_grad()  # Clear gradients from the previous iteration
        scaler.scale(loss).backward()  # Scale the loss and perform backpropagation
        scaler.unscale_(optimizer)  # Unscale gradients before gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Clip gradients to prevent exploding gradients
        scaler.step(optimizer)  # Update model parameters (invalid updates are skipped automatically)
        scaler.update()  # Update the gradient scaling factor

        # =================================================== Evaluation ===================================================
        preds = torch.sigmoid(logits)  # [B, 1, H, W]
        pred_masks = (preds > 0.5).long()
        label_masks = labels.long()
        pred_onehot = torch.cat([1 - pred_masks, pred_masks], dim=1)  # [B, 2, H, W]
        label_onehot = torch.cat([1 - label_masks, label_masks], dim=1)  # [B, 2, H, W]

        with torch.no_grad():
            accuracy.update(pred_masks, label_masks)
            dice.update(pred_onehot, label_onehot)
            miou.update(pred_masks, label_masks)

    # Compute the average metrics for the current epoch
    with torch.no_grad():
        epoch_accuracy = accuracy.compute().item()
        epoch_dice = dice.compute().item()
        epoch_miou = miou.compute().item()

    # Log the metrics for the current epoch
    train_logger.info(
        f'Epoch: {epoch}  ACC:{epoch_accuracy:.4f}  Dice:{epoch_dice:.4f}  IOU:{epoch_miou:.4f}'
    )

    # Update the learning rate if a scheduler is provided
    if lr_scheduler is not None:
        lr_scheduler.step()


def val_one_epoch(device, epoch, loader, model, val_logger, task_name, tokenizer=None, llm=None):
    """
    Evaluate the model on the validation set after each training epoch.

    Args:
        device: Computation device.
        epoch: Current epoch.
        loader: Data loader.
        model: Neural network model.
        val_logger: Validation logger.
        task_name: Task name.
        tokenizer: Tokenizer.
        llm: Large language model used to convert structured prompts into natural language prompts.

    Returns:
        average_iou: Mean IoU on the validation set.
        average_dice: Mean Dice score on the validation set.
    """
    # Initialize evaluation metrics
    accuracy = Accuracy(task='binary').to(device)
    dice = DiceScore(num_classes=2, include_background=False, average="micro", input_format="one-hot").to(device)
    miou = BinaryJaccardIndex().to(device)

    with torch.no_grad():  # Disable gradient computation
        # Iterate over the data loader (starting from index 1)
        for i, batch_sample in enumerate(loader, 1):  # Retrieve the batch index and the current batch
            # Unpack images and labels from the current batch
            images, labels = batch_sample['image'], batch_sample['label']
            # Move all tensors to the target device
            images, labels = images.to(device), labels.to(device)

            # =================================================== Forward Pass ===================================================
            if config.model_type == 'UNet':
                logits = model(images)

            elif config.model_type == 'SAM':
                # First forward pass
                logits, low_logits, image_embeddings = model(
                    multimask_output=False,
                    images=images,
                    point_prompts=None,
                    low_logits=None,
                    image_embeddings=None
                )

                # Generate prompts
                masks = (torch.sigmoid(logits) > 0.5)
                text_prompts, point_prompts = utils.create_prompt(labels, masks, task_name)

                # Second forward pass
                logits, low_logits, _ = model(
                    multimask_output=False,
                    images=images,
                    point_prompts=point_prompts,
                    low_logits=low_logits,
                    image_embeddings=image_embeddings
                )

            elif config.model_type == "ScribblePromptUNet":
                # The official ScribblePrompt-UNet uses single-channel input
                sp_images = convert_to_grayscale(images)
                # First forward pass
                first_input = build_scribbleprompt_input(images=sp_images, point_prompts=None, mask_input=None, )
                logits = model(first_input)
                # Generate corrective prompts
                masks = torch.sigmoid(logits) > 0.5
                _, point_prompts = utils.create_prompt(labels, masks, task_name)
                # Second forward pass
                second_input = build_scribbleprompt_input(images=sp_images, point_prompts=point_prompts,
                                                          mask_input=logits.detach())
                logits = model(second_input)

            elif config.model_type == "IMISNet":
                # Generate the initial prompt, as IMISNet requires an initial target prompt
                # for its first prediction
                empty_masks = torch.zeros_like(labels, dtype=torch.bool)
                _, point_prompts = utils.create_prompt(labels, empty_masks, task_name)
                first_prompts = build_imis_prompt(point_prompts=point_prompts, mask_input=None)
                # First interaction
                outputs, image_embeddings = model(images=images, prompts=first_prompts, image_embeddings=None)
                logits = outputs["masks"]
                low_logits = outputs["low_res_masks"]
                # Generate corrective prompts
                masks = torch.sigmoid(logits) > 0.5
                _, point_prompts = utils.create_prompt(labels, masks, task_name)
                second_prompts = build_imis_prompt(point_prompts=point_prompts, mask_input=low_logits)
                # Second interaction
                outputs, _ = model(images=None, prompts=second_prompts, image_embeddings=image_embeddings.detach())
                logits = outputs["masks"]

            elif config.model_type == 'MedFeedSeg':
                # First forward pass
                logits, low_logits, image1, image2, image3, image4 = model(images)

                # Generate prompts
                masks = (torch.sigmoid(logits) > 0.5)
                text_prompts, point_prompts = utils.create_prompt(labels, masks, task_name)
                text_prompts = llm.evaluate_batch(text_prompts)
                text_prompts = tokenizer.tokenize(text_prompts)
                text_prompts = {
                    'input_ids': text_prompts['input_ids'].to(device),
                    'attention_mask': text_prompts['attention_mask'].to(device)
                }

                # Second forward pass
                logits, low_logits, _, _, _, _ = model(
                    images,
                    text_prompts,
                    point_prompts,
                    low_logits,
                    image1,
                    image2,
                    image3,
                    image4
                )

            # =================================================== Evaluation ===================================================
            preds = torch.sigmoid(logits)  # [B, 1, H, W]
            pred_masks = (preds > 0.5).long()  # [B, 1, H, W]
            label_masks = labels.long()  # [B, 1, H, W]
            pred_onehot = torch.cat([1 - pred_masks, pred_masks], dim=1)  # [B, 2, H, W]
            label_onehot = torch.cat([1 - label_masks, label_masks], dim=1)  # [B, 2, H, W]

            accuracy.update(pred_masks, label_masks)
            dice.update(pred_onehot, label_onehot)
            miou.update(pred_masks, label_masks)

        # Compute the average metrics for the current epoch
        epoch_accuracy = accuracy.compute().item()
        epoch_dice = dice.compute().item()
        epoch_miou = miou.compute().item()

        # Log the metrics for the current epoch
        val_logger.info(
            f'Epoch: {epoch}  ACC:{epoch_accuracy:.4f}  Dice:{epoch_dice:.4f}  IOU:{epoch_miou:.4f}'
        )

    return epoch_miou, epoch_dice


def save_checkpoint(state, better_model, save_path):
    """
    Save a model checkpoint.

    Args:
        better_model: Whether this checkpoint is the current best model.
        state: Dictionary containing the model state.
        save_path: Directory for saving the checkpoint.
    """
    # Generate the checkpoint filename based on whether it is the best model
    if better_model:  # Save as the current best model
        filename = os.path.join(save_path, 'best_model.pth.tar')
    else:  # Save as a regular checkpoint
        filename = os.path.join(save_path, 'model.pth.tar')

    torch.save(state, filename)  # Save the model checkpoint


def train_model(resume_session: str = None):
    """
    Train the model.

    Args:
        resume_session: Name of the previous training session to resume from.
    """
    # ========== Saving Configuration ==========
    if resume_session:
        # Resume training from the previous checkpoint
        logger_path = os.path.join(config.save_path, resume_session, 'log')  # Directory for training logs of the previous session
        model_path = os.path.join(config.save_path, resume_session, 'model')  # Directory for model checkpoints of the previous session
    else:
        # Train from scratch
        logger_path = os.path.join(config.save_path, config.session_name, 'log')  # Directory for training logs
        model_path = os.path.join(config.save_path, config.session_name, 'model')  # Directory for model checkpoints
        # Create the directories if they do not exist
        os.makedirs(logger_path, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

    # ========== Logger Setup ==========
    train_logger = utils.config_logger(
        logger_name='train_logger',
        to_file=True,
        log_path=os.path.join(logger_path, 'train.log')
    )
    val_logger = utils.config_logger(
        logger_name='val_logger',
        to_stream=True,
        to_file=True,
        log_path=os.path.join(logger_path, 'val.log')
    )

    # ========== Dataset Setup ==========
    if config.task_name == 'QaTa-COV19-v2':
        # Training set
        train_dataset = QaTa_COV19_v2(dataset_path=config.dataset_path, mode='train')
        # Validation set
        val_dataset = QaTa_COV19_v2(dataset_path=config.dataset_path, mode='val')

    elif config.task_name == 'BUSI':
        # Training set
        train_dataset = BUSI(dataset_path=config.dataset_path, mode='train')
        # Validation set
        val_dataset = BUSI(dataset_path=config.dataset_path, mode='val')

    elif config.task_name == 'BraTS_2021':
        # Training set
        train_dataset = BraTS_2021(dataset_path=config.dataset_path, mode='train')
        # Validation set
        val_dataset = BraTS_2021(dataset_path=config.dataset_path, mode='val')

    elif config.task_name == 'LiTS':
        # Training set
        train_dataset = LiTS(dataset_path=config.dataset_path, mode='train')
        # Validation set
        val_dataset = LiTS(dataset_path=config.dataset_path, mode='val')

    elif config.task_name == 'PolypGen':
        # Training set
        train_dataset = PolypGen(dataset_path=config.dataset_path, mode='train')
        # Validation set
        val_dataset = PolypGen(dataset_path=config.dataset_path, mode='val')

    # Data loaders

    # Create the training data loader.
    # It is an iterable object that returns one batch at each iteration.
    train_loader = DataLoader(
        train_dataset,  # Dataset object (must inherit from Dataset and implement __len__ and __getitem__)
        batch_size=config.train_batch_size,  # Batch size
        shuffle=True,  # Shuffle the dataset at the beginning of every epoch
        worker_init_fn=utils.worker_init_fn,  # Initialize each worker to ensure reproducible data loading
        num_workers=8,  # Number of worker processes for parallel data loading
        pin_memory=True  # Pin CPU memory to accelerate data transfer to the GPU
    )

    # Create the validation data loader
    val_loader = DataLoader(
        val_dataset,  # Dataset object (must inherit from Dataset and implement __len__ and __getitem__)
        batch_size=config.valid_batch_size,  # Batch size
        shuffle=False,  # Keep the sample order fixed during validation
        worker_init_fn=utils.worker_init_fn,  # Initialize each worker to ensure reproducible data loading
        num_workers=8,  # Number of worker processes for parallel data loading
        pin_memory=True  # Pin CPU memory to accelerate data transfer to the GPU
    )

    # ========== Model Setup ==========
    # Select the computation device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    if config.model_type == 'UNet':
        model = UNet(logits_channel=1)

    elif config.model_type == 'SAM':
        model = build_sam_vit_b(
            checkpoint=os.path.join('..', 'pretrained_model', 'sam_vit_b_01ec64.pth')
        )

    elif config.model_type == "ScribblePromptUNet":
        model = ScribblePromptUNet(in_channels=5, out_channels=1, features=[192, 192, 192, 192])

    elif config.model_type == "IMISNet":
        model = build_imisnet(
            checkpoint_path=os.path.join("..", "pretrained_model", "IMISNet-B.pth"),
            device=device,
            image_size=config.image_size[0],
            model_type="vit_b",
            multimask_output=False,
            fine_tune=True
        )

    elif config.model_type == 'MedFeedSeg':
        model = MedFeedSeg(
            image_size=config.image_size,
            text_encoder_path=config.text_encoder_path,
            image_encoder_path=config.image_encoder_path
        )

    model.to(device)  # Move the model to the target device

    if config.model_type == 'MedFeedSeg':
        # Initialize the LLM
        llm = utils.LLM(config.llm_path, config.task_name)

        # Initialize the tokenizer
        tokenizer = utils.TextTokenizer(
            config.text_encoder_path,
            config.text_token
        )
    else:
        llm = None
        tokenizer = None

    # ========== Loss Function ==========
    criterion = DiceCELoss(sigmoid=True)

    # ========== Other Settings ==========
    # Learning rate
    lr = config.learning_rate  # Initial learning rate

    # Optimizer (optimize only trainable parameters)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=200,
        eta_min=1e-6
    )

    # Initialize the gradient scaler for mixed-precision training
    scaler = torch.amp.GradScaler("cuda")

    # ========== Start Training ==========
    if resume_session:
        # Resume training from a checkpoint
        checkpoint = torch.load(os.path.join(model_path, 'model.pth.tar'))  # Load the checkpoint
        model.load_state_dict(utils.cleanup_state_dict(checkpoint['model_state_dict']))  # Restore model parameters
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Restore optimizer state
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])  # Restore learning-rate scheduler state
        scaler.load_state_dict(checkpoint['scaler_state_dict'])  # Restore gradient scaler state
        start_epoch = checkpoint['epoch'] + 1

        # Initialize training statistics
        best_checkpoint = torch.load(os.path.join(model_path, 'best_model.pth.tar'))  # Load the best checkpoint
        max_dice = best_checkpoint['val_dice']  # Best Dice score
        max_iou = best_checkpoint['val_iou']  # Best IoU score
        best_epoch = best_checkpoint['epoch']  # Best epoch

    else:
        # Train from scratch
        start_epoch = 1

        # Initialize training statistics
        max_dice = 0.0  # Best Dice score
        max_iou = 0.0  # Best IoU score
        best_epoch = 0  # Best epoch

    # Enable DataParallel when multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # Parallelize training and inference across multiple GPUs

    val_logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")

    # Start the training loop
    for epoch in range(start_epoch, config.max_epochs + 1):
        # Train for one epoch
        train_logger.info(f'========== Epoch [{epoch}/{config.max_epochs}] ==========')
        model.train()  # Set the model to training mode

        train_one_epoch(
            device=device,  # Computation device
            epoch=epoch,  # Current epoch
            loader=train_loader,  # Training data loader
            model=model,  # Model
            criterion=criterion,  # Loss function
            optimizer=optimizer,  # Optimizer
            lr_scheduler=lr_scheduler,  # Learning-rate scheduler
            train_logger=train_logger,  # Training logger
            scaler=scaler,  # Gradient scaler
            task_name=config.task_name,  # Task name
            tokenizer=tokenizer,  # Tokenizer
            llm=llm  # Large language model
        )

        # Evaluate on the validation set
        val_logger.info(f'========== Epoch [{epoch}/{config.max_epochs}] ==========')

        with torch.no_grad():  # Disable gradient computation
            model.eval()  # Set the model to evaluation mode

            # Evaluate the model on the validation set
            val_iou, val_dice = val_one_epoch(
                device=device,  # Computation device
                epoch=epoch,  # Current epoch
                loader=val_loader,  # Validation data loader
                model=model,  # Model
                val_logger=val_logger,  # Validation logger
                task_name=config.task_name,  # Task name
                tokenizer=tokenizer,  # Tokenizer
                llm=llm  # Large language model
            )

        # Save the best model
        if val_dice >= max_dice:  # Save the model if the validation Dice score improves
            val_logger.info(
                f'Saving the best model. Validation Dice improved from {max_dice:.5f} to {val_dice:.5f}.'
            )

            max_iou = val_iou  # Update the best IoU
            max_dice = val_dice  # Update the best Dice score
            best_epoch = epoch  # Update the best epoch

            # Save the checkpoint
            save_checkpoint(
                {
                    'epoch': epoch,  # Current epoch
                    'val_iou': val_iou,  # Validation IoU
                    'val_dice': val_dice,  # Validation Dice score
                    'model_state_dict': model.state_dict(),  # Model parameters
                    'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
                    'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,  # Learning-rate scheduler state
                    'scaler_state_dict': scaler.state_dict()  # Gradient scaler state
                },
                better_model=True,  # Save as the current best model
                save_path=model_path  # Directory for saving checkpoints
            )

        # Save a regular checkpoint every five epochs
        elif epoch % 5 == 0:
            # Save the checkpoint
            save_checkpoint(
                {
                    'epoch': epoch,  # Current epoch
                    'val_iou': val_iou,  # Validation IoU
                    'val_dice': val_dice,  # Validation Dice score
                    'model_state_dict': model.state_dict(),  # Model parameters
                    'optimizer_state_dict': optimizer.state_dict(),  # Optimizer state
                    'scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,  # Learning-rate scheduler state
                    'scaler_state_dict': scaler.state_dict()  # Gradient scaler state
                },
                better_model=False,  # Save as a regular checkpoint
                save_path=model_path  # Directory for saving checkpoints
            )

        # Early stopping
        early_stopping_count = epoch - best_epoch  # Number of consecutive epochs without improvement

        # Stop training if there is no improvement for multiple consecutive epochs
        if (
            early_stopping_count >= config.early_stopping_patience
            and epoch >= config.min_epochs
        ):
            train_logger.info('Early stopping triggered.')
            val_logger.info('Early stopping triggered.')
            break

    # Training completed
    val_logger.info(
        f'Training completed. '
        f'Best epoch: {best_epoch}, '
        f'Best validation Dice: {max_dice:.5f}, '
        f'Best validation IoU: {max_iou:.5f}'
    )


if __name__ == '__main__':
    """
    Execute only when this script is run directly.

    __name__ is a special variable. Its value is '__main__'
    when the module is executed directly, and the module name
    when it is imported by another module.
    """

    # Set the multiprocessing start method to "spawn"
    mp.set_start_method('spawn')

    # Configure deterministic behavior for reproducibility
    deterministic = True
    if not deterministic:
        cudnn.benchmark = True  # Enable cuDNN auto-tuning
        cudnn.deterministic = False  # Disable deterministic algorithms
    else:
        cudnn.benchmark = False  # Disable cuDNN auto-tuning
        cudnn.deterministic = True  # Enable deterministic algorithms

    # Set random seeds for reproducibility
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    # Train the model
    train_model(resume_session=config.resume_session)

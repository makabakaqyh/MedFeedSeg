import os
import torch
import numpy as np
from tqdm import tqdm
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import BinaryJaccardIndex
from scipy.ndimage import binary_erosion, distance_transform_edt

import config
import utils
from dataset import QaTa_COV19_v2, BUSI, BraTS_2021, LiTS, PolypGen
from network.UNet import UNet
from network.SAM import build_sam_vit_b
from network.ScribblePromptUNet.network import ScribblePromptUNet
from network.ScribblePromptUNet.scribbleprompt_input import convert_to_grayscale, build_scribbleprompt_input
from network.IMISNet.build_model import build_imisnet, build_imis_prompt
from network.MedFeedSeg import MedFeedSeg


def hd95(mask, target):
    """
    Compute the 95th percentile Hausdorff Distance (HD95)
    between the predicted mask and the ground-truth mask.

    Args:
        mask: Predicted binary mask.
        target: Ground-truth binary mask.

    Returns:
        hd95: The 95th percentile of the bidirectional surface distances.
    """
    mask = mask.astype(bool)  # 0 -> False, nonzero -> True (foreground=True, background=False)
    target = target.astype(bool)

    if mask.sum() == 0 and target.sum() == 0:  # Both prediction and ground truth contain no foreground
        return 0.0
    if mask.sum() == 0 or target.sum() == 0:  # HD95 is undefined when only one mask contains foreground
        return np.nan

    # Compute the distance transform of the background
    dt_target = distance_transform_edt(~target)  # Distance to the nearest foreground pixel in the ground truth
    dt_pred = distance_transform_edt(~mask)  # Distance to the nearest foreground pixel in the prediction

    # Extract object boundaries using binary erosion
    surface_target = target ^ binary_erosion(target)  # Ground-truth boundary
    surface_pred = mask ^ binary_erosion(mask)  # Prediction boundary

    d1 = dt_target[surface_pred]  # Distances from prediction boundary to ground-truth foreground
    d2 = dt_pred[surface_target]  # Distances from ground-truth boundary to prediction foreground

    all_dist = np.concatenate([d1, d2])  # Combine all surface distances
    hd95 = np.percentile(all_dist, 95)  # Compute the 95th percentile distance

    return hd95


if __name__ == '__main__':
    # ===================================================== Basic Setup =====================================================
    # Configure the logger
    logger_path = os.path.join(config.save_path, config.session_name, 'log', 'test.log')
    test_logger = utils.config_logger(
        logger_name='eval_logger',
        to_stream=True,
        to_file=True,
        log_path=logger_path
    )

    # Select the computation device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ==================================================== Dataset Setup ====================================================
    if config.task_name == 'QaTa-COV19-v2':
        test_dataset = QaTa_COV19_v2(dataset_path=config.dataset_path, mode='test')
    elif config.task_name == 'BUSI':
        test_dataset = BUSI(dataset_path=config.dataset_path, mode='test')
    elif config.task_name == 'BraTS_2021':
        test_dataset = BraTS_2021(dataset_path=config.dataset_path, mode='test')
    elif config.task_name == 'LiTS':
        test_dataset = LiTS(dataset_path=config.dataset_path, mode='test')
    elif config.task_name == 'PolypGen':
        test_dataset = PolypGen(dataset_path=config.dataset_path, mode='test')

    # ===================================================== Model Setup =====================================================
    # Path to the trained model
    model_path = os.path.join(
        config.save_path,
        config.session_name,
        'model',
        'best_model.pth.tar'
    )

    # Initialize the model
    if config.model_type == 'UNet':
        model = UNet(logits_channel=1)

    elif config.model_type == 'SAM':
        model = build_sam_vit_b()

    elif config.model_type == "ScribblePromptUNet":
        model = ScribblePromptUNet(
            in_channels=5,
            out_channels=1,
            features=[192, 192, 192, 192]
        )

    elif config.model_type == "IMISNet":
        model = build_imisnet(
            checkpoint_path=None,
            device=device,
            image_size=config.image_size[0],
            model_type="vit_b",
            multimask_output=False,
            fine_tune=False
        )

    elif config.model_type == 'MedFeedSeg':
        model = MedFeedSeg(
            image_size=config.image_size,
            text_encoder_path=config.text_encoder_path,
            image_encoder_path=config.image_encoder_path
        )

        # Initialize the LLM
        llm = utils.LLM(config.llm_path, config.task_name)

        # Initialize the tokenizer
        tokenizer = utils.TextTokenizer(
            config.text_encoder_path,
            config.text_token
        )

    # Load the model checkpoint
    checkpoint = torch.load(model_path)
    val_dice = checkpoint['val_dice']
    val_iou = checkpoint['val_iou']
    model.load_state_dict(utils.cleanup_state_dict(checkpoint['model_state_dict']))
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # ===================================================== Inference =====================================================
    # Initialize evaluation metrics
    dice = DiceScore(
        num_classes=2,
        include_background=False,
        average="micro",
        input_format="one-hot"
    ).to(device)

    iou = BinaryJaccardIndex().to(device)

    all_dice = []
    all_iou = []
    all_hd95 = []

    with torch.no_grad():  # Disable gradient computation
        for item in tqdm(range(test_dataset.__len__())):
            data = test_dataset.__getitem__(item)
            image = data['image']  # [3, H, W]
            label = data['label']  # [1, H, W]

            # Add a batch dimension and move the tensors to the target device
            image = image.unsqueeze(0).to(device)  # [1, 3, H, W]
            label = label.unsqueeze(0).to(device)  # [1, 1, H, W]

            # ================================================= Forward Pass =================================================
            if config.model_type == 'UNet':
                logit = model(image)

            elif config.model_type == 'SAM':
                # First forward pass
                logit, low_logit, image_embedding = model(
                    multimask_output=False,
                    images=image,
                    point_prompts=None,
                    low_logits=None,
                    image_embeddings=None
                )

                # Generate prompts
                mask = (torch.sigmoid(logit) > 0.5)
                text_prompt, point_prompt = utils.create_prompt(
                    label,
                    mask,
                    config.task_name
                )

                # Second forward pass
                logit, low_logit, _ = model(
                    multimask_output=False,
                    images=image,
                    point_prompts=point_prompt,
                    low_logits=low_logit,
                    image_embeddings=image_embedding
                )

            elif config.model_type == "ScribblePromptUNet":
                # The official ScribblePrompt-UNet uses single-channel input
                sp_image = convert_to_grayscale(image)
                # First forward pass
                first_input = build_scribbleprompt_input(images=sp_image, point_prompts=None, mask_input=None, )
                logit = model(first_input)
                # Generate corrective prompts
                mask = torch.sigmoid(logit) > 0.5
                _, point_prompt = utils.create_prompt(label, mask, config.task_name)
                # Second forward pass
                second_input = build_scribbleprompt_input(images=sp_image, point_prompts=point_prompt,
                                                          mask_input=logit.detach())
                logit = model(second_input)

            elif config.model_type == "IMISNet":
                # Generate the initial prompt, as IMISNet requires an initial target prompt
                # for its first prediction
                empty_mask = torch.zeros_like(label, dtype=torch.bool)
                _, point_prompt = utils.create_prompt(label, empty_mask, config.task_name)
                first_prompt = build_imis_prompt(point_prompts=point_prompt, mask_input=None)
                # First interaction
                output, image_embedding = model(images=image, prompts=first_prompt, image_embeddings=None)
                logit = output["masks"]
                low_logit = output["low_res_masks"]
                # Generate corrective prompts
                mask = torch.sigmoid(logit) > 0.5
                _, point_prompt = utils.create_prompt(label, mask, config.task_name)
                second_prompt = build_imis_prompt(point_prompts=point_prompt, mask_input=low_logit)
                # Second interaction
                output, _ = model(images=None, prompts=second_prompt, image_embeddings=image_embedding.detach())
                logit = output["masks"]

            elif config.model_type == 'MedFeedSeg':
                # First forward pass
                logit, low_logit, image1, image2, image3, image4 = model(image)

                # Generate prompts
                mask = (torch.sigmoid(logit) > 0.5)
                text_prompt, point_prompt = utils.create_prompt(
                    label,
                    mask,
                    config.task_name
                )

                text_prompt = llm.evaluate_batch(text_prompt)
                text_prompt = tokenizer.tokenize(text_prompt)
                text_prompt = {
                    'input_ids': text_prompt['input_ids'].to(device),
                    'attention_mask': text_prompt['attention_mask'].to(device)
                }

                # Second forward pass
                logit, low_logit, _, _, _, _ = model(
                    image,
                    text_prompt,
                    point_prompt,
                    low_logit,
                    image1,
                    image2,
                    image3,
                    image4
                )

            pred = torch.sigmoid(logit)  # [1, 1, H, W]
            label = label.long()  # [1, 1, H, W]

            # ================================================ Compute HD95 ================================================
            # Convert tensors to NumPy arrays
            hd_mask = (pred.cpu().numpy() > 0.5)
            hd_label = label.cpu().numpy()

            sample_hd95 = hd95(hd_mask == 1, hd_label == 1)

            if not np.isnan(sample_hd95):
                all_hd95.append(sample_hd95)

            # ================================================= Evaluation =================================================
            pred_mask = (pred > 0.5).long()
            label_mask = label.long()

            pred_onehot = torch.cat([1 - pred_mask, pred_mask], dim=1)  # [B, 2, H, W]
            label_onehot = torch.cat([1 - label_mask, label_mask], dim=1)  # [B, 2, H, W]

            # Update metric states
            dice.update(pred_onehot, label_onehot)
            iou.update(pred_mask, label_mask)

            # Compute Dice and IoU for the current sample
            if label.sum() == 0:  # No foreground in the ground-truth mask
                if pred_mask.sum() == 0:  # No foreground in the prediction
                    sample_dice = 1.0
                    sample_iou = 1.0
                else:  # False positive prediction
                    sample_dice = 0.0
                    sample_iou = 0.0
            else:  # Compute metrics normally
                sample_dice = dice.compute().item()
                sample_iou = iou.compute().item()

            all_dice.append(sample_dice)
            all_iou.append(sample_iou)

            # Reset metric states
            dice.reset()
            iou.reset()

        # =================================================== Overall Metrics ===================================================
        # Convert to NumPy arrays
        all_iou = np.array(all_iou)
        all_dice = np.array(all_dice)
        all_hd95 = np.array(all_hd95)

        # Compute summary statistics
        mean_iou = all_iou.mean()
        mean_dice = all_dice.mean()
        std_dice = all_dice.std()
        mean_hd95 = all_hd95.mean()
        worst_dice = np.sort(all_dice)[:max(1, int(0.1 * len(all_dice)))].mean()  # Average Dice of the worst-performing 10% samples

    # Log evaluation results
    test_logger.info(
        f'task name:{config.task_name},  model name:{config.model_type}, session name:{config.session_name}'
    )
    test_logger.info(f'Val Dice:{val_dice:.5f}, Val IOU:{val_iou:.5f}')
    test_logger.info(f'Test Dice: {mean_dice:.5f}, Test IOU:{mean_iou:.5f}')
    test_logger.info(f'Std Dice:  {std_dice:.5f}')
    test_logger.info(f'Mean HD95: {mean_hd95:.5f}')
    test_logger.info(f'Worst 10% Dice: {worst_dice:.5f}')
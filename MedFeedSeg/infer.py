import os
import torch
from tqdm import tqdm
from torchmetrics.segmentation import DiceScore
from torchmetrics.classification import Accuracy, BinaryJaccardIndex
from PIL import Image

import config
import utils
from dataset import QaTa_COV19_v2, BUSI, BraTS_2021, LiTS, PolypGen
from network.UNet import UNet
from network.SAM import build_sam_vit_b
from network.ScribblePromptUNet.network import ScribblePromptUNet
from network.ScribblePromptUNet.scribbleprompt_input import convert_to_grayscale, build_scribbleprompt_input
from network.IMISNet.build_model import build_imisnet, build_imis_prompt
from network.MedFeedSeg import MedFeedSeg


if __name__ == '__main__':
    # ==================================================== Output Directory ====================================================
    output_path = os.path.join(config.save_path, config.session_name, 'inference')
    os.makedirs(output_path, exist_ok=True)  # Create the directory if it does not exist

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

    # Define the device (GPU or CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ===================================================== Model Setup =====================================================
    # Path to the trained model
    model_path = os.path.join(config.save_path, config.session_name, 'model', 'best_model.pth.tar')

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
        llm = utils.LLM(config.llm_path, config.task_name)  # Initialize the LLM
        tokenizer = utils.TextTokenizer(config.text_encoder_path, config.text_token)  # Initialize the tokenizer

    # Load the model checkpoint
    checkpoint = torch.load(model_path)
    print(f"val_dice:{checkpoint['val_dice']}")
    print(f"val_iou:{checkpoint['val_iou']}")
    model.load_state_dict(utils.cleanup_state_dict(checkpoint['model_state_dict']))
    model.to(device)
    model.eval()  # Set the model to evaluation mode

    # Initialize evaluation metrics
    acc = Accuracy(task='binary').to(device)
    dice = DiceScore(
        num_classes=2,
        include_background=False,
        average="micro",
        input_format="one-hot"
    ).to(device)
    iou = BinaryJaccardIndex().to(device)

    with torch.no_grad():  # Disable gradient computation
        # for item in tqdm(range(test_dataset.__len__())):
        for item in tqdm(range(100)):
            # Load the sample, add a batch dimension, and move it to the target device
            data = test_dataset.__getitem__(item)
            image = data['image'].unsqueeze(0).to(device)  # [1, 3, H, W]
            label = data['label'].unsqueeze(0).to(device)  # [1, 1, H, W]
            sample_filename = data['sample_filename']

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
                text_prompt, point_prompt = utils.create_prompt(label, mask, config.task_name)

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
                text_prompt, point_prompt = utils.create_prompt(label, mask, config.task_name)
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

            # ================================================= Evaluation =================================================
            pred = torch.sigmoid(logit)  # [1, 1, H, W]
            pred_mask = (pred > 0.5).long()  # [1, 1, H, W]
            label_mask = label.long()  # [1, 1, H, W]
            pred_onehot = torch.cat([1 - pred_mask, pred_mask], dim=1)  # [1, 2, H, W]
            label_onehot = torch.cat([1 - label_mask, label_mask], dim=1)  # [1, 2, H, W]

            # Update metric states
            acc.update(pred_mask, label_mask)
            dice.update(pred_onehot, label_onehot)
            iou.update(pred_mask, label_mask)

            # Compute evaluation metrics
            sample_acc = acc.compute().item()
            sample_dice = dice.compute().item()
            sample_iou = iou.compute().item()

            # Reset metric states
            acc.reset()
            dice.reset()
            iou.reset()

            # ================================================= Save Results =================================================
            mask_np = (pred_mask.cpu().squeeze().numpy() * 255).astype('uint8')
            filepath = os.path.join(output_path, sample_filename)
            Image.fromarray(mask_np).save(filepath)
from types import SimpleNamespace

import torch

from .model import IMISNet
from .segment_anything import sam_model_registry


def build_imisnet(
        checkpoint_path: str | None,
        device: torch.device,
        image_size: int = 256,
        model_type: str = "vit_b",
        multimask_output: bool = False,
        fine_tune: bool = True,
) -> IMISNet:
    """
    Build IMISNet-Net and load the official IMISNet-B checkpoint.

    Args:
        checkpoint_path:
            Path to the official IMISNet-B.pth checkpoint.
        device:
            Training device.
        image_size:
            Model input resolution.
        model_type:
            IMISNet-B uses "vit_b".
        multimask_output:
            False is recommended for binary medical segmentation.
        fine_tune:
            If True, freeze the image and text encoders and fine-tune
            the prompt encoder and mask decoder.
    """
    args = SimpleNamespace(
        image_size=image_size,
        sam_checkpoint=checkpoint_path,
    )

    # build_sam.py loads the official checkpoint internally
    sam = sam_model_registry[model_type](args)
    sam = sam.to(device)

    model = IMISNet(
        sam=sam,
        test_mode=False,
        multimask_output=multimask_output,
        category_weights=None,
        select_mask_num=1,
    ).to(device)

    if fine_tune:
        # Freeze the SAM image encoder
        for parameter in model.image_encoder.parameters():
            parameter.requires_grad = False

        # Freeze the CLIP text encoder
        for parameter in model.text_model.parameters():
            parameter.requires_grad = False

        for parameter in model.text_out_dim.parameters():
            parameter.requires_grad = False

        # Train the prompt encoder
        for parameter in model.prompt_encoder.parameters():
            parameter.requires_grad = True

        # Train the mask decoder
        for parameter in model.mask_decoder.parameters():
            parameter.requires_grad = True

    return model


def build_imis_prompt(point_prompts, mask_input=None):
    point_coords, point_labels = point_prompts

    prompts = {
        "point_coords": point_coords.long(),
        "point_labels": point_labels.long(),
    }

    if mask_input is not None:
        prompts["mask_inputs"] = mask_input.detach()

    return prompts
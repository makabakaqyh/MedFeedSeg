import torch
from functools import partial
from .modeling import ViT, MaskDecoder, PromptEncoder, Sam, TwoWayTransformer
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextConfig
from torch.nn import functional as F

def build_sam_vit_h(args):
    return _build_sam(
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        pretrain_model = 'samvit_huge_patch16'
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(args):
    return _build_sam(
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        pretrain_model = 'samvit_large_patch16'
    )


def build_sam_vit_b(args):
    return _build_sam(
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        image_size=args.image_size,
        checkpoint=args.sam_checkpoint,
        pretrain_model = 'samvit_base_patch16'
    
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    image_size,
    checkpoint,
    pretrain_model
):
    prompt_embed_dim = 768
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder = ViT(
            encoder_embed_dim = encoder_embed_dim,
            pretrain_model= pretrain_model,
            out_chans= prompt_embed_dim,
            depth = encoder_depth,
            freeze_encoder = True,
            pretrained=False,
            ),

        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),

        text_model =  CLIPTextModel(CLIPTextConfig()),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )

    if checkpoint is not None:
        state_dict = torch.load(
            checkpoint,
            map_location="cpu",
        )

        # Support checkpoints wrapped in a dictionary
        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]

        model_keys = set(sam.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())

        model_uses_flat_text_keys = any(
            key.startswith("text_model.embeddings.")
            for key in model_keys
        )

        model_uses_nested_text_keys = any(
            key.startswith("text_model.text_model.embeddings.")
            for key in model_keys
        )

        checkpoint_uses_flat_text_keys = any(
            key.startswith("text_model.embeddings.")
            for key in checkpoint_keys
        )

        checkpoint_uses_nested_text_keys = any(
            key.startswith("text_model.text_model.embeddings.")
            for key in checkpoint_keys
        )

        # Checkpoint: text_model.text_model.xxx
        # Model:      text_model.xxx
        if (
                model_uses_flat_text_keys
                and checkpoint_uses_nested_text_keys
        ):
            converted_state_dict = {}

            for key, value in state_dict.items():
                if key.startswith("text_model.text_model."):
                    key = (
                            "text_model."
                            + key[len("text_model.text_model."):]
                    )

                converted_state_dict[key] = value

            state_dict = converted_state_dict

            print(
                "Converted checkpoint text keys: "
                "text_model.text_model.* -> text_model.*"
            )

        # Checkpoint: text_model.xxx
        # Model:      text_model.text_model.xxx
        elif (
                model_uses_nested_text_keys
                and checkpoint_uses_flat_text_keys
        ):
            converted_state_dict = {}

            for key, value in state_dict.items():
                if key.startswith("text_model."):
                    key = (
                            "text_model.text_model."
                            + key[len("text_model."):]
                    )

                converted_state_dict[key] = value

            state_dict = converted_state_dict

            print(
                "Converted checkpoint text keys: "
                "text_model.* -> text_model.text_model.*"
            )

        # Strict loading ensures that all image/prompt/decoder weights match
        sam.load_state_dict(
            state_dict,
            strict=True,
        )

        print("******Loaded IMISNet parameters")

    return sam


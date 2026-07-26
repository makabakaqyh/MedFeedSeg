import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoModel
from einops import rearrange
from monai.networks.blocks.unetr_block import UnetrUpBlock

import config


class TextEncoder(nn.Module):
    """
    Pretrained text encoder for extracting text features.
    """
    def __init__(self, text_encoder_path):
        """
        Initialize the text encoder.

        Args:
            text_encoder_path: Path to the pretrained text encoder.
        """
        super().__init__()

        # Load the pretrained text encoder from the local directory
        self.model = AutoModel.from_pretrained(text_encoder_path, local_files_only=True)

        # Freeze the parameters of the text encoder
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, text):
        """
        Forward pass.

        Args:
            text: Tokenized text output from the tokenizer,
                  including {'input_ids', 'attention_mask'}.

        Returns:
            The last hidden-state features of the text encoder
            with shape [B, seq_len, 768].
        """
        self.model.eval()

        with torch.no_grad():
            output = self.model(input_ids=text['input_ids'], attention_mask=text['attention_mask'])

        return output.last_hidden_state


class ImageEncoder(nn.Module):
    """
    Pretrained image encoder for extracting image features.
    """
    def __init__(self, image_encoder_path):
        """
        Initialize the image encoder.

        Args:
            image_encoder_path: Path to the pretrained image encoder.
        """
        super().__init__()

        # Load the pretrained image encoder from the local directory
        self.model = AutoModel.from_pretrained(image_encoder_path, local_files_only=True)

    def forward(self, image):
        """
        Forward pass.

        Args:
            image: Input image of shape [B, C, H, W].

        Returns:
            The output features from the four encoder stages:
            [B, 96, H/4, W/4],
            [B, 192, H/8, W/8],
            [B, 384, H/16, W/16],
            [B, 768, H/32, W/32].
        """
        x = self.model.embeddings(pixel_values=image)
        features = []
        for stage in self.model.encoder.stages:
            x = stage(x)
            features.append(x)

        image1, image2, image3, image4 = features

        return image1, image2, image3, image4


class PositionEncoding1D(nn.Module):
    """
    1D positional encoding module.
    """
    def __init__(self, dim: int, seq_len: int = 5000):
        """
        Initialize the positional encoding.

        Args:
            dim: Embedding dimension.
            seq_len: Maximum sequence length.
        """
        super().__init__()

        # Initialize the positional encoding matrix
        position_encode = torch.zeros(seq_len, dim)  # [seq_len, dim]

        # Generate position indices as a column vector
        position = torch.arange(0, seq_len).unsqueeze(1)  # [seq_len, 1]

        # Compute the scaling factor
        factor = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))  # [dim/2]

        # Apply sine to even dimensions
        position_encode[:, 0::2] = torch.sin(position * factor)

        # Apply cosine to odd dimensions
        position_encode[:, 1::2] = torch.cos(position * factor)

        # Add the batch dimension
        position_encode = position_encode.unsqueeze(0)

        # Register as a buffer (saved with the model but not trainable)
        self.register_buffer('position_encode', position_encode)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input feature sequence.

        Returns:
            Feature sequence with positional encoding.
        """
        # Add positional encoding to the input sequence
        x = x + self.position_encode[:, :x.size(1), :]
        return x


class PositionEncoding2D(nn.Module):
    """
    2D positional encoding module.
    """
    def __init__(self, half_dim):
        """
        Initialize the positional encoding.

        Args:
            half_dim: Half of the embedding dimension.
        """
        super().__init__()

        # Register a fixed Gaussian projection matrix as a buffer
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            torch.randn((2, half_dim))
        )

    def _pe_encoding(self, coords):
        """
        Generate positional encoding for normalized coordinates.

        Args:
            coords: Input coordinates of shape [H, W, 2] or [B, N, 2].

        Returns:
            Positional encoding corresponding to the input coordinates.
        """
        # Map coordinates from [0, 1] to [-1, 1]
        coords = 2 * coords - 1

        # Project 2D coordinates into the embedding space
        coords = coords @ self.positional_encoding_gaussian_matrix

        # Map coordinates into the periodic space
        coords = 2 * np.pi * coords

        # Concatenate sine and cosine embeddings
        pe = torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)
        return pe

    def forward(self, image):
        """
        Args:
            image: Image features of shape [B, C, H, W].

        Returns:
            Image features with positional encoding.
        """
        h, w = image.shape[2], image.shape[3]

        # Create a 2D grid
        grid = torch.ones((h, w), device=image.device, dtype=torch.float32)

        # Generate normalized y coordinates
        y_embed = grid.cumsum(dim=0) - 0.5

        # Generate normalized x coordinates
        x_embed = grid.cumsum(dim=1) - 0.5

        y_embed = y_embed / h
        x_embed = x_embed / w

        # Stack x and y coordinates
        coords = torch.stack([x_embed, y_embed], dim=-1)

        # Generate positional encoding for the coordinate grid
        pe = self._pe_encoding(coords)

        # Convert to [1, C, H, W]
        pe = pe.permute(2, 0, 1).unsqueeze(0)

        # Add positional encoding to image features
        output = image + pe
        return output

    def coords_forward(self, coords, image_size):
        """
        Args:
            coords: Point coordinates of shape [B, N, 2].
            image_size: Image size (H, W).

        Returns:
            Positional encoding of the input coordinates.
        """
        # Shift coordinates to the pixel centers
        coords = coords + 0.5

        # Normalize x coordinates
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]

        # Normalize y coordinates
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]

        # Generate positional encoding
        output = self._pe_encoding(coords)
        return output


class PointEncoder(nn.Module):
    """
    Point prompt encoder.
    """
    def __init__(self, embed_dim, image_size):
        """
        Initialize the point encoder.

        Args:
            embed_dim: Embedding dimension.
            image_size: Image size (H, W).
        """
        super().__init__()

        # Positional encoding
        self.position_encoding = PositionEncoding2D(embed_dim // 2)
        self.image_size = image_size  # Image size (H, W)

        # Embedding for positive points
        self.pos_point_embed = nn.Embedding(1, embed_dim)

        # Embedding for negative points
        self.neg_point_embed = nn.Embedding(1, embed_dim)

        # Embedding for invalid points
        self.not_point_embed = nn.Embedding(1, embed_dim)

    def forward(self, point_prompt):
        """
        Forward pass.

        Args:
            point_prompt: Point prompt consisting of
                point coordinates [B, N, 2]
                and point labels [B, N].

        Returns:
            Embedded point prompts of shape [B, N, embed_dim].
        """
        coords, labels = point_prompt

        point_embed = self.position_encoding.coords_forward(
            coords,
            self.image_size
        )

        point_embed[labels == -1] = 0.0
        point_embed[labels == -1] += self.not_point_embed.weight
        point_embed[labels == 0] += self.neg_point_embed.weight  # Negative points
        point_embed[labels == 1] += self.pos_point_embed.weight  # Positive points

        return point_embed


class MLPBlock(nn.Module):
    def __init__(self, embedding_dim, mlp_dim):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.lin1(x)
        x = self.act(x)
        x = self.lin2(x)
        return x


class Decoder(nn.Module):
    """
    Decoder module.
    """
    def __init__(self, dim, skip_first_layer_pe: bool = False):
        """
        Initialize the decoder.

        Args:
            dim: Feature dimension.
            skip_first_layer_pe: Whether to skip positional encoding
                in the first self-attention layer.
        """
        super().__init__()
        self.skip_first_layer_pe = skip_first_layer_pe

        # Positional encoding module
        self.image_pe = PositionEncoding2D(half_dim=dim // 2)

        # Attention modules
        self.self_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.cross_attn_token_to_image = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)
        self.cross_attn_image_to_token = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)

        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.norm4 = nn.LayerNorm(dim)

        # MLP block
        self.mlp = MLPBlock(dim, 2048)

        # UNETR-style upsampling block
        self.skip_connect = UnetrUpBlock(
            2,  # 2D image
            dim,
            dim // 2,
            3,  # Kernel size
            2,  # Upsampling stride
            norm_name='BATCH'  # Batch normalization
        )

    def forward(self, token, token_pe, image, logit, skip):
        """
        Forward pass.

        Args:
            token: Token features of shape [B, L, D].
            token_pe: Positional encoding of the tokens [B, L, D].
            image: Image features [B, C, H, W].
            logit: Low-logit features from the previous iteration [B, C, H, W].
            skip: Skip connection features [B, C/2, 2H, 2W].

        Returns:
            output_image: Decoded image features.
            output_prompt: Updated token features.
        """
        # Record the spatial dimensions
        h, w = image.shape[-2:]
        image = image + logit
        image_with_pe = self.image_pe(image)

        # ================================================ Token Self-Attention ================================================
        if self.skip_first_layer_pe:
            output_token, _ = self.self_attn(query=token, key=token, value=token)
        else:
            qk = token + token_pe
            attn_out, _ = self.self_attn(query=qk, key=qk, value=token)
            output_token = token + attn_out
        output_token = self.norm1(output_token)

        # ======================================== Token-to-Image Cross-Attention =========================================
        q = output_token + token_pe
        k = rearrange(image_with_pe, 'B C H W -> B (H W) C')  # [B,C,H,W] -> [B,P,C]
        v = rearrange(image, 'B C H W -> B (H W) C')  # [B,C,H,W] -> [B,P,C]
        attn_out, _ = self.cross_attn_token_to_image(query=q, key=k, value=v)
        output_token = output_token + attn_out
        output_token = self.norm2(output_token)

        # ==================================================== MLP Block ====================================================
        mlp_out = self.mlp(output_token)
        output_token = output_token + mlp_out
        output_token = self.norm3(output_token)

        # ======================================== Image-to-Token Cross-Attention =========================================
        q = rearrange(image_with_pe, 'B C H W -> B (H W) C')  # [B,C,H,W] -> [B,P,C]
        k = output_token + token_pe
        attn_out, _ = self.cross_attn_image_to_token(query=q, key=k, value=output_token)
        output_image = rearrange(image, 'B C H W -> B (H W) C') + attn_out
        output_image = self.norm4(output_image)

        # Restore the spatial dimensions
        output_image = rearrange(output_image, 'B (H W) C -> B C H W', H=h, W=w)  # [B,P,C] -> [B,C,H,W]

        # Apply the skip connection
        output_image = self.skip_connect(output_image, skip)

        return output_token, output_image


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MedFeedSeg(nn.Module):
    def __init__(self, image_size, text_encoder_path, image_encoder_path, prompt_seq_len=config.text_token):
        """
        Initialize the MedFeedSeg model.

        Args:
            image_size: Original image size (H, W).
            text_encoder_path: Path to the pretrained text encoder.
            image_encoder_path: Path to the pretrained image encoder.
            prompt_seq_len: Length of the text prompt.
        """
        super().__init__()
        self.prompt_seq_len = prompt_seq_len
        feature_dim = [96, 192, 384, 768]  # Number of channels at each feature level

        # =================================================== Encoders ===================================================
        self.image_encoder = ImageEncoder(image_encoder_path)

        # Text prompt encoder
        self.text_prompt_encoder = TextEncoder(text_encoder_path)

        # Embedding used when no text prompt is provided
        self.no_text_prompt_embed = nn.Embedding(1, feature_dim[3])

        # Point prompt encoder
        self.point_prompt_encoder = PointEncoder(feature_dim[3], image_size)

        # Embedding used when no point prompt is provided
        self.no_point_prompt_embed = nn.Embedding(1, feature_dim[3])

        # Logit encoder
        self.logit_downscaling1 = nn.Sequential(
            nn.Conv2d(1, feature_dim[1], kernel_size=2, stride=2),
            LayerNorm2d(feature_dim[1]),
            nn.GELU()
        )
        self.logit_downscaling2 = nn.Sequential(
            nn.Conv2d(feature_dim[1], feature_dim[2], kernel_size=2, stride=2),
            LayerNorm2d(feature_dim[2]),
            nn.GELU()
        )
        self.logit_downscaling3 = nn.Sequential(
            nn.Conv2d(feature_dim[2], feature_dim[3], kernel_size=2, stride=2),
            LayerNorm2d(feature_dim[3]),
            nn.GELU()
        )

        # Embeddings used when no logit is provided
        self.no_logit_embed1 = nn.Embedding(1, feature_dim[1])
        self.no_logit_embed2 = nn.Embedding(1, feature_dim[2])
        self.no_logit_embed3 = nn.Embedding(1, feature_dim[3])

        # =================================================== Decoders ===================================================
        # Learnable mask token
        self.mask_token = nn.Embedding(1, feature_dim[3])

        # Linear projections for token positional encodings
        self.tokens_pe_linear1 = nn.Linear(feature_dim[3], feature_dim[2])
        self.tokens_pe_linear2 = nn.Linear(feature_dim[3], feature_dim[1])

        # Linear projections for adjusting token feature dimensions
        self.tokens_linear1 = nn.Linear(feature_dim[3], feature_dim[2])
        self.tokens_linear2 = nn.Linear(feature_dim[2], feature_dim[1])
        self.tokens_linear3 = nn.Linear(feature_dim[1], feature_dim[0])

        # Initialize the decoders
        self.decoder1 = Decoder(dim=768, skip_first_layer_pe=True)
        self.decoder2 = Decoder(dim=384)
        self.decoder3 = Decoder(dim=192)

        # MLP for generating the dynamic pixel classifier
        self.pixel_classifier_mlps = MLPBlock(feature_dim[0], 2048)

    def forward(self, image, text_prompt=None, point_prompt=None, low_logit=None,
                image1=None, image2=None, image3=None, image4=None):
        """
        Forward pass.

        Args:
            image: Input image of shape [B, C, H, W].
            text_prompt: Tokenized text prompt, including
                {'input_ids', 'attention_mask'}.
            point_prompt: Point prompt consisting of point coordinates
                [B, N, 2] and point labels [B, N].
            low_logit: Low-resolution logits from the previous iteration
                with shape [B, 1, H/4, W/4].

        Returns:
            logits: Segmentation logits of shape [B, 1, H, W].
            low_logit: Low-resolution logits of shape [B, 1, H/4, W/4].
            image1, image2, image3, image4: Reusable image features.
        """
        original_size = (image.shape[-2], image.shape[-1])

        # =================================================== Encoding ===================================================
        # Encode the input image
        if image1 is None or image2 is None or image3 is None or image4 is None:
            image_feature = self.image_encoder(image)  # [B,C,H,W] -> 4 × [B,Ci,Hi,Wi]
            image1 = image_feature[0]  # [B,96,H/4,W/4]
            image2 = image_feature[1]  # [B,192,H/8,W/8]
            image3 = image_feature[2]  # [B,384,H/16,W/16]
            image4 = image_feature[3]  # [B,768,H/32,W/32]

        # Encode the text prompt
        if text_prompt is not None:
            text_prompt = self.text_prompt_encoder(text_prompt)  # {'input_ids','attention_mask'} -> [B,L,768]
        else:
            text_prompt = self.no_text_prompt_embed.weight.reshape(1, 1, -1).expand(
                image.shape[0], self.prompt_seq_len, -1)  # [B,L,768]

        # Encode the point prompt
        if point_prompt is not None:
            point_prompt = self.point_prompt_encoder(point_prompt)  # ([B,N,2],[B,N]) -> [B,N,768]
        else:
            point_prompt = self.no_point_prompt_embed.weight.reshape(1, 1, -1).expand(
                image.shape[0], 1, -1)  # [B,1,768]

        # Encode the low-resolution logits
        if low_logit is not None:
            low_logit1 = self.logit_downscaling1(low_logit)  # [B,1,H/4,W/4] -> [B,192,H/8,W/8]
            low_logit2 = self.logit_downscaling2(low_logit1)  # [B,192,H/8,W/8] -> [B,384,H/16,W/16]
            low_logit3 = self.logit_downscaling3(low_logit2)  # [B,384,H/16,W/16] -> [B,768,H/32,W/32]
        else:
            low_logit1 = self.no_logit_embed1.weight.reshape(1, image2.shape[1], 1, 1).expand(
                image2.shape[0], image2.shape[1], image2.shape[2], image2.shape[3])  # [192] -> [B,192,H/8,W/8]
            low_logit2 = self.no_logit_embed2.weight.reshape(1, image3.shape[1], 1, 1).expand(
                image3.shape[0], image3.shape[1], image3.shape[2], image3.shape[3])  # [384] -> [B,384,H/16,W/16]
            low_logit3 = self.no_logit_embed3.weight.reshape(1, image4.shape[1], 1, 1).expand(
                image4.shape[0], image4.shape[1], image4.shape[2], image4.shape[3])  # [768] -> [B,768,H/32,W/32]

        # =================================================== Decoding ===================================================
        # Concatenate the point prompt after the text prompt
        tokens = torch.cat([text_prompt, point_prompt], dim=1)  # [B,L+N,768]

        # Construct the mask token
        mask_token = self.mask_token.weight.reshape(1, 1, -1).expand(image.shape[0], 1, -1)  # [B,1,768]

        # Append the mask token to the prompt tokens
        tokens = torch.cat([tokens, mask_token], dim=1)  # [B,L+N+1,768]

        # Generate token positional encodings using linear projections
        tokens_pe1 = tokens  # [B,L+N+1,768]
        tokens_pe2 = self.tokens_pe_linear1(tokens)  # [B,L+N+1,768] -> [B,L+N+1,384]
        tokens_pe3 = self.tokens_pe_linear2(tokens)  # [B,L+N+1,768] -> [B,L+N+1,192]

        # Decode image features through multi-scale feature interaction
        f_img = image4

        # [B,L+N+1,768] + [B,768,H/32,W/32] -> [B,L+N+1,768] + [B,384,H/16,W/16]
        tokens, f_img = self.decoder1(token=tokens, token_pe=tokens_pe1,
                                      image=f_img, logit=low_logit3, skip=image3)
        tokens = self.tokens_linear1(tokens)  # [B,L+N+1,768] -> [B,L+N+1,384]

        # [B,L+N+1,384] + [B,384,H/16,W/16] -> [B,L+N+1,384] + [B,192,H/8,W/8]
        tokens, f_img = self.decoder2(token=tokens, token_pe=tokens_pe2,
                                      image=f_img, logit=low_logit2, skip=image2)
        tokens = self.tokens_linear2(tokens)  # [B,L+N+1,384] -> [B,L+N+1,192]

        # [B,L+N+1,192] + [B,192,H/8,W/8] -> [B,L+N+1,192] + [B,96,H/4,W/4]
        tokens, f_img = self.decoder3(token=tokens, token_pe=tokens_pe3,
                                      image=f_img, logit=low_logit1, skip=image1)
        tokens = self.tokens_linear3(tokens)  # [B,L+N+1,192] -> [B,L+N+1,96]

        # Generate the dynamic pixel classifier from the mask token
        pixel_classifier = self.pixel_classifier_mlps(tokens[:, -1:, :])  # [B,1,96] -> [B,1,96]

        # Generate the low-resolution logits using the pixel classifier
        b, c, h, w = f_img.shape
        low_logit = (pixel_classifier @ f_img.flatten(2)).reshape(b, 1, h, w)  # [B,96,H/4,W/4] -> [B,1,H/4,W/4]

        # Upsample the low-resolution logits to the original image size
        logit = F.interpolate(low_logit, original_size,
                              mode="bilinear", align_corners=False)  # [B,1,H/4,W/4] -> [B,1,H,W]

        return logit, low_logit, image1, image2, image3, image4
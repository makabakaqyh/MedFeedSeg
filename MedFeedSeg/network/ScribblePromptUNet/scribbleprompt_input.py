from typing import Optional, Tuple

import torch


def encode_point_prompts(
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        height: int,
        width: int,
        dtype: torch.dtype,
) -> torch.Tensor:
    """
    将点提示编码为正负点提示图。

    Args:
        point_coords: [B, N, 2]，坐标格式为 [x, y]
        point_labels: [B, N]，1=正点，0=负点，-1=无效点
        height: 图像高度
        width: 图像宽度
        dtype: 输出数据类型

    Returns:
        point_maps: [B, 2, H, W]
            第0通道：正点击
            第1通道：负点击
    """
    batch_size, num_points, _ = point_coords.shape
    device = point_coords.device

    point_maps = torch.zeros(
        (batch_size, 2, height, width),
        dtype=dtype,
        device=device,
    )

    # 坐标由 float 转换为可用于张量索引的 long
    coords = point_coords.round().long()

    # 防止坐标越界
    coords_x = coords[..., 0].clamp(0, width - 1)
    coords_y = coords[..., 1].clamp(0, height - 1)

    batch_indices = torch.arange(
        batch_size,
        device=device,
    ).view(-1, 1).expand(-1, num_points)

    # 正点击
    positive = point_labels == 1
    point_maps[
        batch_indices[positive],
        0,
        coords_y[positive],
        coords_x[positive],
    ] = 1.0

    # 负点击
    negative = point_labels == 0
    point_maps[
        batch_indices[negative],
        1,
        coords_y[negative],
        coords_x[negative],
    ] = 1.0

    # point_labels == -1 的位置不会写入提示图
    return point_maps


def convert_to_grayscale(images: torch.Tensor) -> torch.Tensor:
    """
    将输入转换为 ScribblePrompt-UNet 使用的单通道图像。

    Args:
        images: [B, 1, H, W] 或 [B, 3, H, W]

    Returns:
        images: [B, 1, H, W]
    """
    if images.shape[1] == 1:
        return images

    if images.shape[1] == 3:
        return (
            0.299 * images[:, 0:1]
            + 0.587 * images[:, 1:2]
            + 0.114 * images[:, 2:3]
        )

    raise ValueError(
        f"ScribblePrompt-UNet expects 1- or 3-channel images, "
        f"but received shape {tuple(images.shape)}."
    )


def build_scribbleprompt_input(
        images: torch.Tensor,
        point_prompts: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        mask_input: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    构造 ScribblePrompt-UNet 的五通道输入。

    五个通道：
        1. 灰度图像
        2. Bounding box 提示图
        3. 正点提示图
        4. 负点提示图
        5. 上一轮预测 logits

    Args:
        images: [B, 1, H, W]
        point_prompts:
            point_coords: [B, N, 2]
            point_labels: [B, N]
        mask_input: [B, 1, H, W]，上一轮预测 logits

    Returns:
        model_input: [B, 5, H, W]
    """
    if images.ndim != 4 or images.shape[1] != 1:
        raise ValueError(
            f"images must have shape [B, 1, H, W], "
            f"but received {tuple(images.shape)}."
        )

    batch_size, _, height, width = images.shape
    device = images.device
    dtype = images.dtype

    # 当前实验不使用 bounding box
    box_map = torch.zeros(
        (batch_size, 1, height, width),
        dtype=dtype,
        device=device,
    )

    if point_prompts is None:
        # 第一次预测没有点击
        point_maps = torch.zeros(
            (batch_size, 2, height, width),
            dtype=dtype,
            device=device,
        )
    else:
        point_coords, point_labels = point_prompts

        point_coords = point_coords.to(device)
        point_labels = point_labels.to(device)

        point_maps = encode_point_prompts(
            point_coords=point_coords,
            point_labels=point_labels,
            height=height,
            width=width,
            dtype=dtype,
        )

    if mask_input is None:
        # 第一次预测没有上一轮预测
        mask_input = torch.zeros(
            (batch_size, 1, height, width),
            dtype=dtype,
            device=device,
        )
    else:
        if mask_input.shape[-2:] != (height, width):
            mask_input = torch.nn.functional.interpolate(
                mask_input,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        mask_input = mask_input.to(
            device=device,
            dtype=dtype,
        ).detach()

    model_input = torch.cat(
        (
            images,
            box_map,
            point_maps,
            mask_input,
        ),
        dim=1,
    )

    assert model_input.shape == (
        batch_size,
        5,
        height,
        width,
    )

    return model_input
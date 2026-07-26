import random
import numpy as np
import cv2
import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import json

import config


def resize_and_pad(image, target_size, interpolation):
    """
    Resize and pad a NumPy image to the target size.

    Args:
        image: Input image with shape [H, W, C].
        target_size: Target image size.
        interpolation: Interpolation method (different methods are used for images and labels).

    Returns:
        image: Processed image with shape [target_size, target_size, C].
    """
    # Original image size
    h, w, c = image.shape

    # Resize
    scale = target_size[0] / max(h, w)  # Scaling factor
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    image = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    # Padding
    pad_h = max(target_size[0] - new_h, 0)
    pad_w = max(target_size[1] - new_w, 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    image = np.pad(
        image,
        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )

    return image


def config_logger(logger_name: str, to_stream: bool = False, to_file: bool = False, log_path: str = None):
    """
    Configure a logger.

    Args:
        logger_name: Name of the logger.
        to_stream: Whether to output logs to the console.
        to_file: Whether to save logs to a file.
        log_path: Path to the log file.

    Returns:
        logger: Configured logger instance.
    """
    # Create the logger
    logger = logging.getLogger(logger_name)
    # Set the logging level to INFO so that only INFO and above are recorded
    logger.setLevel(level=logging.INFO)
    # Disable propagation to parent loggers
    logger.propagate = False

    # Log formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Logging handlers
    if to_stream:
        # Create a console handler for printing log messages to the terminal
        stream_handler = logging.StreamHandler()
        # Only INFO and above will be displayed
        stream_handler.setLevel(logging.INFO)
        # Attach the handler to the logger
        logger.addHandler(stream_handler)
        # Apply the formatter
        stream_handler.setFormatter(formatter)

    if to_file and log_path:
        # Create a file handler for writing log messages to a file
        file_handler = logging.FileHandler(
            filename=log_path,
            mode='a',
            encoding='UTF-8'
        )
        # Only INFO and above will be written to the file
        file_handler.setLevel(logging.INFO)
        # Attach the handler to the logger
        logger.addHandler(file_handler)
        # Apply the formatter
        file_handler.setFormatter(formatter)

    return logger


def worker_init_fn(worker_id):
    """
    Initialize each DataLoader worker.

    This function is executed when each DataLoader worker process starts.

    Args:
        worker_id: ID of the current worker process.
    """
    random.seed(config.seed + worker_id)  # Set the random seed


def cleanup_state_dict(parameter_dict):
    """
    Remove the ``module.`` prefix automatically added to parameter names
    when a model is trained with DataParallel.

    Args:
        parameter_dict: Dictionary containing model parameters.

    Returns:
        new_dict: State dictionary with the ``module.`` prefix removed.
    """
    new_dict = {}
    for k, v in parameter_dict.items():
        if k.startswith('module.'):  # Remove the "module." prefix
            new_dict[k[7:]] = v
        else:  # Keep the original key
            new_dict[k] = v
    return new_dict


class TextTokenizer:
    def __init__(self, text_encoder_path, text_token):
        self.text_token = text_token
        self.tokenizer = AutoTokenizer.from_pretrained(
            text_encoder_path,
            trust_remote_code=True
        )

    def tokenize(self, text):
        tokens = self.tokenizer(
            text,
            max_length=self.text_token,          # Maximum output sequence length
            padding='max_length',                # Pad to the maximum length
            truncation=True,                     # Truncate overlong sequences
            return_attention_mask=True,          # Return the attention mask
            return_tensors='pt'                  # Return PyTorch tensors
        )

        text = {
            'input_ids': tokens['input_ids'],
            'attention_mask': tokens['attention_mask']
        }
        return text


def analyze_qata_cov19_v2_error(error_mask, error_type, area_threshold):
    """
    Analyze segmentation errors on the QaTa-COV19-v2 dataset and generate
    structured prompts for the LLM to convert into natural-language feedback.

    Args:
        error_mask: Binary mask of the error region.
        error_type: Error type.
        area_threshold: Minimum area threshold.

    Returns:
        attribution: Structured prompt.
    """
    # Perform connected-component analysis:
    # number of components, label map, component statistics, and centroids
    num_component, labels, stats, centroids = cv2.connectedComponentsWithStats(
        error_mask,
        connectivity=8  # Diagonal pixels are also considered connected
    )

    # Analyze the results
    attribution = None

    if num_component > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]  # Areas of all foreground components
        index = np.argmax(areas)  # Index of the largest connected component
        component = (labels == index + 1)  # Largest connected component
        area = areas[index]

        if area >= area_threshold and area >= 2:

            # ============================================== Centroid ===============================================
            cx, cy = centroids[index + 1]

            # ========== Left / Right ==========
            H, W = error_mask.shape
            if cx < W / 2:
                side = "right"  # Left side of the image corresponds to the patient's right lung
            else:
                side = "left"   # Right side of the image corresponds to the patient's left lung

            # ========================================== Vertical Position ==========================================
            vertical_ratio = cy / H

            if area < 1500:
                if vertical_ratio < 0.3:
                    vertical_region = "upper"
                elif vertical_ratio < 0.5:
                    vertical_region = "middle"
                else:
                    vertical_region = "lower"
            else:
                if vertical_ratio < 0.4:
                    vertical_region = "upper-middle"
                else:
                    vertical_region = "middle-lower"

            # ================================================= PCA =================================================
            ys, xs = np.where(component > 0)
            points = np.column_stack([xs, ys]).astype(np.float32)

            mean, eigenvectors, eigenvalues = cv2.PCACompute2(points, mean=None)

            # First principal component (major axis direction)
            major_vec = eigenvectors[0]

            # Eigenvalue of the major axis
            major_value = eigenvalues[0, 0]

            # Eigenvalue of the minor axis
            minor_value = eigenvalues[1, 0]

            # Compute the major-to-minor axis ratio
            pca_axis_ratio = np.sqrt(major_value / max(minor_value, 1e-6))

            # Compute the orientation angle of the major axis
            angle_deg = np.degrees(np.arctan2(major_vec[1], major_vec[0]))

            # Normalize the angle to [-90, 90]
            if angle_deg < -90:
                angle_deg += 180
            elif angle_deg > 90:
                angle_deg -= 180

            # ===================================== Determine the orientation ======================================
            abs_angle = abs(angle_deg)
            if abs_angle <= 20:
                orientation = "horizontal"
            elif abs_angle >= 70:
                orientation = "vertical"
            else:
                orientation = "oblique"

            # ======================================== Determine the shape =========================================
            if pca_axis_ratio >= 3.0:
                shape = "elongated"
            elif pca_axis_ratio >= 1.5:
                shape = "oval"
            else:
                shape = "compact"

            # ======================================== Merge the results ===========================================
            attribution = {
                "error_type": error_type,
                "side": side,
                "vertical_region": vertical_region,
                "shape": shape,
                "orientation": orientation,
            }

    return attribution


def analyze_busi_error(tp, error_mask, error_type, area_threshold):
    """
    Analyze segmentation errors on the BUSI dataset and generate
    structured prompts for the LLM to convert into natural-language feedback.

    Args:
        tp: True positive region mask.
        error_mask: Error region mask.
        error_type: Error type.
        area_threshold: Minimum area threshold.

    Returns:
        attribution: Structured prompt.
    """
    # Perform connected-component analysis:
    # number of components, label map, component statistics, and centroids
    tp_num_component, tp_labels, tp_stats, tp_centroids = cv2.connectedComponentsWithStats(
        tp, connectivity=8)  # Diagonal pixels are also considered connected
    error_num_component, error_labels, error_stats, error_centroids = cv2.connectedComponentsWithStats(
        error_mask, connectivity=8)

    # Analyze the results
    attribution = None

    if error_num_component > 1 and tp_num_component > 1:
        areas = error_stats[1:, cv2.CC_STAT_AREA]  # Areas of all foreground error components
        index = np.argmax(areas)  # Index of the largest connected component
        area = areas[index]

        if area >= area_threshold and area >= 2:
            # ============================================== Centroid ===============================================
            error_cx, error_cy = error_centroids[index + 1]

            tp_areas = tp_stats[1:, cv2.CC_STAT_AREA]  # Areas of all foreground TP components
            tp_index = np.argmax(tp_areas)  # Index of the largest TP component
            tp_cx, tp_cy = tp_centroids[tp_index + 1]

            # =========================== Direction relative to the correctly segmented region ===========================
            dx = error_cx - tp_cx
            dy = error_cy - tp_cy
            angle = np.degrees(np.arctan2(dy, dx))

            if -22.5 <= angle < 22.5:
                boundary_direction = "right"
            elif 22.5 <= angle < 67.5:
                boundary_direction = "lower-right"
            elif 67.5 <= angle < 112.5:
                boundary_direction = "lower"
            elif 112.5 <= angle < 157.5:
                boundary_direction = "lower-left"
            elif angle >= 157.5 or angle < -157.5:
                boundary_direction = "left"
            elif -157.5 <= angle < -112.5:
                boundary_direction = "upper-left"
            elif -112.5 <= angle < -67.5:
                boundary_direction = "upper"
            elif -67.5 <= angle < -22.5:
                boundary_direction = "upper-right"

            # ========================================= Boundary adjustment ==========================================
            if error_type == "under-segmentation":
                boundary_adjustment = "outward"
            elif error_type == "over-segmentation":
                boundary_adjustment = "inward"

            # =========================================== Merge the results ==========================================
            attribution = {
                "error_type": error_type,
                "case_type": "boundary_correction",
                "boundary_direction": boundary_direction,
                "boundary_adjustment": boundary_adjustment
            }

    elif error_num_component > 1 and tp_num_component == 1:
        areas = error_stats[1:, cv2.CC_STAT_AREA]  # Areas of all foreground error components
        index = np.argmax(areas)  # Index of the largest connected component
        area = areas[index]

        if area >= area_threshold and area >= 2:
            # ========================================= Correction scenario =========================================
            if error_type == "under-segmentation":
                case_type = "lesion_completely_missed"
            elif error_type == "over-segmentation":
                case_type = "no_lesion_false_positive"

            # =========================== Direction relative to the correctly segmented region ===========================
            boundary_direction = "unknown"

            # ========================================= Boundary adjustment ==========================================
            boundary_adjustment = "unknown"

            # =========================================== Merge the results ==========================================
            attribution = {
                "error_type": error_type,
                "case_type": case_type,
                "boundary_direction": boundary_direction,
                "boundary_adjustment": boundary_adjustment
            }

    return attribution


def analyze_brats2021_error(tp, error_mask, error_type, area_threshold):
    """
    Analyze segmentation errors on the BraTS_2021 dataset and generate
    structured prompts for the LLM to convert into natural-language feedback.

    Args:
        tp: True positive region mask.
        error_mask: Error region mask.
        error_type: Error type.
        area_threshold: Minimum area threshold.

    Returns:
        attribution: Structured prompt.
    """
    # Perform connected-component analysis:
    # number of components, label map, component statistics, and centroids
    tp_num_component, tp_labels, tp_stats, tp_centroids = cv2.connectedComponentsWithStats(
        tp, connectivity=8)  # Diagonal pixels are also considered connected
    error_num_component, error_labels, error_stats, error_centroids = cv2.connectedComponentsWithStats(
        error_mask, connectivity=8)

    # Analyze the results
    attribution = None

    if error_num_component > 1 and tp_num_component > 1:
        areas = error_stats[1:, cv2.CC_STAT_AREA]  # Areas of all foreground error components
        index = np.argmax(areas)  # Index of the largest connected component
        area = areas[index]

        if area >= area_threshold and area >= 2:
            # ============================================== Centroid ===============================================
            error_cx, error_cy = error_centroids[index + 1]

            # ============================================== Hemisphere =============================================
            H, W = error_mask.shape
            if error_cy < H / 2:
                hemisphere = "right hemisphere"  # Upper half of the image corresponds to the right hemisphere
            else:
                hemisphere = "left hemisphere"  # Lower half of the image corresponds to the left hemisphere

            # ========================================= Boundary adjustment ==========================================
            if error_type == "under-segmentation":
                boundary_adjustment = "outward"
            elif error_type == "over-segmentation":
                boundary_adjustment = "inward"
            else:
                boundary_adjustment = "unknown"

            # =========================================== Merge the results ==========================================
            attribution = {
                "error_type": error_type,
                "hemisphere": hemisphere,
                "case_type": "boundary_correction",
                "boundary_adjustment": boundary_adjustment
            }

    elif error_num_component > 1 and tp_num_component == 1:
        areas = error_stats[1:, cv2.CC_STAT_AREA]  # Areas of all foreground error components
        index = np.argmax(areas)  # Index of the largest connected component
        area = areas[index]

        if area >= area_threshold and area >= 2:
            # ========================================= Correction scenario =========================================
            if error_type == "under-segmentation":
                case_type = "lesion_completely_missed"
            elif error_type == "over-segmentation":
                case_type = "no_lesion_false_positive"

            # ============================================== Centroid ===============================================
            error_cx, error_cy = error_centroids[index + 1]

            # ============================================== Hemisphere =============================================
            H, W = error_mask.shape
            if error_cy < H / 2:
                hemisphere = "right hemisphere"  # Upper half of the image corresponds to the right hemisphere
            else:
                hemisphere = "left hemisphere"  # Lower half of the image corresponds to the left hemisphere

            # ========================================= Boundary adjustment ==========================================
            boundary_adjustment = "unknown"

            # =========================================== Merge the results ==========================================
            attribution = {
                "error_type": error_type,
                "hemisphere": hemisphere,
                "case_type": case_type,
                "boundary_adjustment": boundary_adjustment
            }

    return attribution


def analyze_lits_error(tp, error_mask, error_type, area_threshold):
    """
    Analyze segmentation errors on the LiTS dataset and generate
    structured prompts for the LLM to convert into natural-language feedback.

    Args:
        tp: True positive region mask.
        error_mask: Error region mask.
        error_type: Error type.
        area_threshold: Minimum area threshold.

    Returns:
        attribution: Structured prompt.
    """
    # Perform connected-component analysis:
    # number of components, label map, component statistics, and centroids
    tp_num_component, tp_labels, tp_stats, tp_centroids = cv2.connectedComponentsWithStats(
        tp, connectivity=8)  # Diagonal pixels are also considered connected
    error_num_component, error_labels, error_stats, error_centroids = cv2.connectedComponentsWithStats(
        error_mask, connectivity=8)

    # Analyze the results
    attribution = None

    if error_num_component > 1 and tp_num_component > 1:
        areas = error_stats[1:, cv2.CC_STAT_AREA]  # Areas of all foreground error components
        index = np.argmax(areas)  # Index of the largest connected component
        area = areas[index]

        if area >= area_threshold and area >= 2:
            # ============================================== Centroid ===============================================
            error_cx, error_cy = error_centroids[index + 1]

            tp_areas = tp_stats[1:, cv2.CC_STAT_AREA]  # Areas of all foreground TP components
            tp_index = np.argmax(tp_areas)  # Index of the largest TP component
            tp_cx, tp_cy = tp_centroids[tp_index + 1]

            # =========================== Direction relative to the correctly segmented region ===========================
            dx = error_cx - tp_cx
            dy = error_cy - tp_cy
            angle = np.degrees(np.arctan2(dy, dx))

            if -22.5 <= angle < 22.5:
                boundary_direction = "right"
            elif 22.5 <= angle < 67.5:
                boundary_direction = "lower-right"
            elif 67.5 <= angle < 112.5:
                boundary_direction = "lower"
            elif 112.5 <= angle < 157.5:
                boundary_direction = "lower-left"
            elif angle >= 157.5 or angle < -157.5:
                boundary_direction = "left"
            elif -157.5 <= angle < -112.5:
                boundary_direction = "upper-left"
            elif -112.5 <= angle < -67.5:
                boundary_direction = "upper"
            elif -67.5 <= angle < -22.5:
                boundary_direction = "upper-right"

            # ========================================= Boundary adjustment ==========================================
            if error_type == "under-segmentation":
                boundary_adjustment = "outward"
            elif error_type == "over-segmentation":
                boundary_adjustment = "inward"

            # =========================================== Merge the results ==========================================
            attribution = {
                "error_type": error_type,
                "case_type": "boundary_correction",
                "boundary_direction": boundary_direction,
                "boundary_adjustment": boundary_adjustment
            }

    elif error_num_component > 1 and tp_num_component == 1:
        areas = error_stats[1:, cv2.CC_STAT_AREA]  # Areas of all foreground error components
        index = np.argmax(areas)  # Index of the largest connected component
        area = areas[index]

        if area >= area_threshold and area >= 2:
            # ========================================= Correction scenario =========================================
            if error_type == "under-segmentation":
                case_type = "lesion_completely_missed"
            elif error_type == "over-segmentation":
                case_type = "no_lesion_false_positive"

            # =========================== Direction relative to the correctly segmented region ===========================
            boundary_direction = "unknown"

            # ========================================= Boundary adjustment ==========================================
            boundary_adjustment = "unknown"

            # =========================================== Merge the results ==========================================
            attribution = {
                "error_type": error_type,
                "case_type": case_type,
                "boundary_direction": boundary_direction,
                "boundary_adjustment": boundary_adjustment
            }

    return attribution


def analyze_polypgen_error(tp, error_mask, error_type, area_threshold):
    """
    Analyze segmentation errors on the PolypGen dataset and generate
    structured prompts for the LLM to convert into natural-language feedback.

    Args:
        tp: True positive region mask.
        error_mask: Error region mask.
        error_type: Error type.
        area_threshold: Minimum area threshold.

    Returns:
        attribution: Structured prompt.
    """
    # Perform connected-component analysis:
    # number of components, label map, component statistics, and centroids
    tp_num_component, tp_labels, tp_stats, tp_centroids = cv2.connectedComponentsWithStats(
        tp, connectivity=8)  # Diagonal pixels are also considered connected
    error_num_component, error_labels, error_stats, error_centroids = cv2.connectedComponentsWithStats(
        error_mask, connectivity=8)

    # Analyze the results
    attribution = None

    if error_num_component > 1 and tp_num_component > 1:
        areas = error_stats[1:, cv2.CC_STAT_AREA]  # Areas of all foreground error components
        index = np.argmax(areas)  # Index of the largest connected component
        area = areas[index]

        if area >= area_threshold and area >= 2:
            # ============================================== Centroid ===============================================
            error_cx, error_cy = error_centroids[index + 1]

            tp_areas = tp_stats[1:, cv2.CC_STAT_AREA]  # Areas of all foreground TP components
            tp_index = np.argmax(tp_areas)  # Index of the largest TP component
            tp_cx, tp_cy = tp_centroids[tp_index + 1]

            # =========================== Direction relative to the correctly segmented region ===========================
            dx = error_cx - tp_cx
            dy = error_cy - tp_cy
            angle = np.degrees(np.arctan2(dy, dx))

            if -22.5 <= angle < 22.5:
                boundary_direction = "right"
            elif 22.5 <= angle < 67.5:
                boundary_direction = "lower-right"
            elif 67.5 <= angle < 112.5:
                boundary_direction = "lower"
            elif 112.5 <= angle < 157.5:
                boundary_direction = "lower-left"
            elif angle >= 157.5 or angle < -157.5:
                boundary_direction = "left"
            elif -157.5 <= angle < -112.5:
                boundary_direction = "upper-left"
            elif -112.5 <= angle < -67.5:
                boundary_direction = "upper"
            elif -67.5 <= angle < -22.5:
                boundary_direction = "upper-right"

            # ========================================= Boundary adjustment ==========================================
            if error_type == "under-segmentation":
                boundary_adjustment = "outward"
            elif error_type == "over-segmentation":
                boundary_adjustment = "inward"

            # =========================================== Merge the results ==========================================
            attribution = {
                "error_type": error_type,
                "case_type": "boundary_correction",
                "boundary_direction": boundary_direction,
                "boundary_adjustment": boundary_adjustment
            }

    elif error_num_component > 1 and tp_num_component == 1:
        areas = error_stats[1:, cv2.CC_STAT_AREA]  # Areas of all foreground error components
        index = np.argmax(areas)  # Index of the largest connected component
        area = areas[index]

        if area >= area_threshold and area >= 2:
            # ========================================= Correction scenario =========================================
            if error_type == "under-segmentation":
                case_type = "lesion_completely_missed"
            elif error_type == "over-segmentation":
                case_type = "no_lesion_false_positive"

            # =========================== Direction relative to the correctly segmented region ===========================
            boundary_direction = "unknown"

            # ========================================= Boundary adjustment ==========================================
            boundary_adjustment = "unknown"

            # =========================================== Merge the results ==========================================
            attribution = {
                "error_type": error_type,
                "case_type": case_type,
                "boundary_direction": boundary_direction,
                "boundary_adjustment": boundary_adjustment
            }

    return attribution


def analyze_pred(label, mask, task_name):
    """
    Analyze the segmentation result and generate point prompts
    and structured text prompts.

    Args:
        label: Ground-truth mask [H, W].
        mask: Predicted mask [H, W].
        task_name: Task name.

    Returns:
        text_prompt: Text prompt
            [{"error_type", "side", "vertical_region", "area",
              "shape", "orientation", "centroid",
              "PCA major axis ratio"}, ...]
        (point_coords, point_labels):
            (point coordinates, point labels)
    """
    label_np = (np.array(label) > 0).astype(np.uint8)
    mask_np = (np.array(mask) > 0).astype(np.uint8)

    # Region categories: TP / FP / FN
    tp = ((mask_np == 1) & (label_np == 1)).astype(np.uint8)  # Correctly segmented region
    fp = ((mask_np == 1) & (label_np == 0)).astype(np.uint8)  # Over-segmented region
    fn = ((mask_np == 0) & (label_np == 1)).astype(np.uint8)  # Under-segmented region

    # =============================================== Generate point prompts ===============================================
    point_coords = np.zeros((2, 2), dtype=np.float32)  # [2, 2]
    point_labels = np.full((2,), -1, dtype=np.int64)  # -1 indicates an invalid point [2,]

    ys, xs = np.where(fp > 0)
    if len(xs) > 0:
        idx = np.random.randint(0, len(xs))
        x = int(xs[idx])
        y = int(ys[idx])
        point_coords[0] = [x, y]
        point_labels[0] = 0

    ys, xs = np.where(fn > 0)
    if len(xs) > 0:
        idx = np.random.randint(0, len(xs))
        x = int(xs[idx])
        y = int(ys[idx])
        point_coords[1] = [x, y]
        point_labels[1] = 1

    # Area of the correctly segmented region
    tp_area = int(tp.sum())

    # =============================================== Generate text prompts ===============================================
    text_prompt = []

    if task_name == "QaTa-COV19-v2":
        attribution_fp = analyze_qata_cov19_v2_error(fp, "over-segmentation", tp_area * 0.1)
        if attribution_fp:
            text_prompt.append(attribution_fp)

        attribution_fn = analyze_qata_cov19_v2_error(fn, "under-segmentation", tp_area * 0.1)
        if attribution_fn:
            text_prompt.append(attribution_fn)

    elif task_name == 'BUSI':
        attribution_fp = analyze_busi_error(tp, fp, "over-segmentation", tp_area * 0.1)
        if attribution_fp:
            text_prompt.append(attribution_fp)

        attribution_fn = analyze_busi_error(tp, fn, "under-segmentation", tp_area * 0.1)
        if attribution_fn:
            text_prompt.append(attribution_fn)

    elif task_name == 'BraTS_2021':
        if fn.sum() >= fp.sum() or tp_area == 0:
            attribution_fn = analyze_brats2021_error(tp, fn, "under-segmentation", tp_area * 0.05)
            if attribution_fn:
                text_prompt.append(attribution_fn)

        if fn.sum() < fp.sum() or tp_area == 0:
            attribution_fp = analyze_brats2021_error(tp, fp, "over-segmentation", tp_area * 0.05)
            if attribution_fp:
                text_prompt.append(attribution_fp)

    elif task_name == 'LiTS':
        attribution_fp = analyze_lits_error(tp, fp, "over-segmentation", tp_area * 0.1)
        if attribution_fp:
            text_prompt.append(attribution_fp)

        attribution_fn = analyze_lits_error(tp, fn, "under-segmentation", tp_area * 0.1)
        if attribution_fn:
            text_prompt.append(attribution_fn)

    elif task_name == 'PolypGen':
        attribution_fp = analyze_polypgen_error(tp, fp, "over-segmentation", tp_area * 0.1)
        if attribution_fp:
            text_prompt.append(attribution_fp)

        attribution_fn = analyze_polypgen_error(tp, fn, "under-segmentation", tp_area * 0.1)
        if attribution_fn:
            text_prompt.append(attribution_fn)

    return text_prompt, (point_coords, point_labels)


def create_prompt(labels, masks, task_name):
    """
    Generate prompts for a batch of samples.

    Args:
        labels: Ground-truth masks [B, 1, H, W].
        masks: Predicted masks [B, 1, H, W].
        task_name: Task name.

    Returns:
        text_prompts:
            [[{"error_type", "side", "vertical_region", "area",
               "shape", "orientation", "centroid",
               "PCA major axis ratio"}, ...] * B]

        (point_coords, point_labels):
            (point coordinates [B, N, 2], point labels [B, N])
    """
    text_prompts = []
    point_coords_list = []
    point_labels_list = []

    for b in range(labels.shape[0]):
        label = labels[b, 0].detach().cpu()
        mask = masks[b, 0].detach().cpu()

        text_prompt, point_prompt = analyze_pred(label, mask, task_name)
        point_coords, point_labels = point_prompt  # points: [N, 2], point_labels: [N]

        text_prompts.append(text_prompt)
        point_coords_list.append(point_coords)
        point_labels_list.append(point_labels)

    # Convert B * [N, 2] -> [B, N, 2]
    point_coords = torch.tensor(
        np.stack(point_coords_list, axis=0),
        dtype=torch.float32,
        device=labels.device
    )

    # Convert B * [N] -> [B, N]
    point_labels = torch.tensor(
        np.stack(point_labels_list, axis=0),
        dtype=torch.long,
        device=labels.device
    )

    return text_prompts, (point_coords, point_labels)


def convert_numpy_types(obj):
    """
    Convert NumPy types to native Python types
    so that they can be serialized by json.dumps().
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(x) for x in obj)
    elif isinstance(obj, list):
        return [convert_numpy_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    else:
        return obj


class LLM:
    def __init__(self, llm_path, task_name):
        """
        Initialize the LLM.

        Args:
            llm_path: Path to the locally deployed LLM.
            task_name: Task name.
        """
        self.task_name = task_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_path,
            local_files_only=True,
            padding_side="left"
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_path,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True
        ).eval()

        self.llm.config.pad_token_id = self.tokenizer.pad_token_id

    def build_prompt(self, text_prompt):
        """
        Build the prompt for the LLM.

        Args:
            text_prompt: All structured text prompts for a single sample.

        Returns:
            prompt: Prompt fed to the LLM.
        """
        text_prompt = convert_numpy_types(text_prompt)

        if self.task_name == "QaTa-COV19-v2":
            prompt = f"""
                You are an expert reviewer of chest medical image segmentation.
            
                You will receive structured attributes of segmentation errors computed from an error map.
            
                Your task is to convert these attributes into one concise, human-like segmentation refinement instruction.
            
                Meaning of error types:
                - "over-segmentation" means the prediction includes extra false-positive regions that should be removed.
                - "under-segmentation" means the prediction misses true regions that should be added.
            
                Important laterality rule:
                - The "side" field already refers to patient laterality.
                - "right" means the patient's right lung.
                - "left" means the patient's left lung.
                - Do not reinterpret it as image left or image right.
            
                Location terms:
                - "upper" means upper lung region.
                - "middle" means middle lung region.
                - "lower" means lower lung region.
                - "upper-middle" means the error spans the upper and middle lung regions.
                - "middle-lower" means the error spans the middle and lower lung regions.
            
                Writing rules:
                1. Output only one sentence.
                2. Do not mention JSON.
                3. Do not mention disease diagnosis.
                4. Use natural clinical-style language.
                5. For over-segmentation, say the region should be removed.
                6. For under-segmentation, say the region should be added.
                7. If both over-segmentation and under-segmentation exist, describe both in the same sentence.
                8. If there is no significant error region, output: "No obvious segmentation correction is needed."
                
                Input structured attributes:
                {json.dumps(text_prompt, ensure_ascii=False, indent=2)}
            
                Output:
            """
        elif self.task_name == "BUSI":
            prompt = f"""
                You are an expert reviewer of breast ultrasound lesion segmentation.

                You will receive structured attributes describing segmentation errors.

                Your task is to convert these attributes into one concise, human-like segmentation refinement instruction.

                Meaning of fields:

                - "case_type" indicates the segmentation situation:
                    - "boundary_correction" means a lesion exists and only the lesion boundary needs correction.
                    - "lesion_completely_missed" means a true lesion exists but the segmentation missed it completely.
                    - "no_lesion_false_positive" means no true lesion is present, but a false segmented region exists.

                - "boundary_direction" indicates which lesion boundary should be corrected.
                - "boundary_adjustment" indicates how the boundary should be modified:
                    - "outward" means the boundary should be expanded outward.
                    - "inward" means the boundary should be contracted inward.

                Meaning of error types:

                - "under-segmentation" means part or all of the lesion is missing.
                - "over-segmentation" means extra non-lesion regions are included.

                Boundary directions:

                - "upper"
                - "lower"
                - "left"
                - "right"
                - "upper-left"
                - "upper-right"
                - "lower-left"
                - "lower-right"

                Writing rules:

                1. Output only one sentence.
                2. Use natural clinical-style language.
                3. Do not mention JSON or structured attributes.
                4. Do not mention model predictions.
                5. Do not mention pathology, diagnosis, benignity, malignancy, or disease type.
                6. If case_type is "boundary_correction" and boundary_adjustment is "outward", say the corresponding lesion boundary should be expanded outward.
                7. If case_type is "boundary_correction" and boundary_adjustment is "inward", say the corresponding lesion boundary should be contracted inward.
                8. If case_type is "lesion_completely_missed", say the lesion has been completely missed and should be included in the segmentation.
                9. If case_type is "no_lesion_false_positive", say no lesion is present and the segmented region should be removed.
                10. If multiple error regions are provided, combine all corrections into one coherent sentence.
                11. If there is no significant error region, output exactly:
                    "No obvious segmentation correction is needed."
                
                Example outputs:

                - "Under-segmentation is present in the upper region of the current segmentation, and the boundary should be expanded outward."
                - "Over-segmentation is present in the lower-left region of the current segmentation, and the boundary should be contracted inward."
                - "Over-segmentation is present in the upper-left region of the current segmentation, and the boundary should be contracted inward, while under-segmentation is present in the lower-right region and the boundary should be expanded outward."
                - "The true lesion region has been completely missed and should be included in the segmentation."
                - "The current segmented region is entirely incorrect and should be removed."
                - "The current segmented region is entirely incorrect and should be removed, while the true lesion region has been completely missed and should be included in the segmentation."

                Input structured attributes:
                {json.dumps(text_prompt, ensure_ascii=False, indent=2)}

                Output:
            """
        elif self.task_name == 'BraTS_2021':
            prompt = f"""
                You are an expert reviewer of brain tumor segmentation.

                You will receive structured attributes describing segmentation errors.

                Your task is to convert these attributes into one concise, human-like segmentation refinement instruction.

                Meaning of fields:

                - "hemisphere" indicates the affected brain hemisphere.

                - "case_type" indicates the segmentation situation:
                    - "boundary_correction" means the target region exists and only the boundary needs correction.
                    - "lesion_completely_missed" means a true target region exists but the segmentation missed it completely.
                    - "no_lesion_false_positive" means no true target region is present, but a false segmented region exists.

                - "boundary_adjustment" indicates how the boundary should be modified:
                    - "outward" means the boundary should be expanded outward.
                    - "inward" means the boundary should be contracted inward.

                Meaning of error types:

                - "under-segmentation" means part or all of the target region is missing.
                - "over-segmentation" means extra non-target regions are included.

                Writing rules:

                1. Output only one sentence.
                2. Use natural clinical-style language.
                3. Do not mention JSON or structured attributes.
                4. Do not mention model predictions.
                5. Do not mention tumor diagnosis, pathology, glioma grade, or disease type.
                6. If case_type is "boundary_correction" and boundary_adjustment is "outward", say the corresponding boundary in the affected hemisphere should be expanded outward.
                7. If case_type is "boundary_correction" and boundary_adjustment is "inward", say the corresponding boundary in the affected hemisphere should be contracted inward.
                8. If case_type is "lesion_completely_missed", say the true target region in the affected hemisphere has been completely missed and should be included in the segmentation.
                9. If case_type is "no_lesion_false_positive", say the segmented region in the affected hemisphere is entirely incorrect and should be removed.
                10. If multiple error regions are provided, combine all corrections into one coherent sentence.
                11. If there is no significant error region, output exactly:
                    "No obvious segmentation correction is needed."

                Example outputs:

                - "Under-segmentation is present in the right hemisphere, and the boundary should be expanded outward."
                - "Over-segmentation is present in the left hemisphere, and the boundary should be contracted inward."
                - "The true target region in the right hemisphere has been completely missed and should be included in the segmentation."
                - "The segmented region in the left hemisphere is entirely incorrect and should be removed."
                - "The segmented region in the left hemisphere is entirely incorrect and should be removed, while the true target region in the right hemisphere has been completely missed and should be included in the segmentation."

                Input structured attributes:
                {json.dumps(text_prompt, ensure_ascii=False, indent=2)}

                Output:
            """
        elif self.task_name == "LiTS":
            prompt = f"""
                You are an expert reviewer of CT liver segmentation.

                You will receive structured attributes describing segmentation errors.

                Your task is to convert these attributes into one concise, human-like segmentation refinement instruction.

                Meaning of fields:

                - "case_type" indicates the segmentation situation:
                    - "boundary_correction" means a lesion exists and only the lesion boundary needs correction.
                    - "lesion_completely_missed" means a true lesion exists but the segmentation missed it completely.
                    - "no_lesion_false_positive" means no true lesion is present, but a false segmented region exists.

                - "boundary_direction" indicates which lesion boundary should be corrected.
                - "boundary_adjustment" indicates how the boundary should be modified:
                    - "outward" means the boundary should be expanded outward.
                    - "inward" means the boundary should be contracted inward.

                Meaning of error types:

                - "under-segmentation" means part or all of the lesion is missing.
                - "over-segmentation" means extra non-lesion regions are included.

                Boundary directions:

                - "upper"
                - "lower"
                - "left"
                - "right"
                - "upper-left"
                - "upper-right"
                - "lower-left"
                - "lower-right"

                Writing rules:

                1. Output only one sentence.
                2. Use natural clinical-style language.
                3. Do not mention JSON or structured attributes.
                4. Do not mention model predictions.
                5. Do not mention pathology, diagnosis, benignity, malignancy, or disease type.
                6. If case_type is "boundary_correction" and boundary_adjustment is "outward", say the corresponding lesion boundary should be expanded outward.
                7. If case_type is "boundary_correction" and boundary_adjustment is "inward", say the corresponding lesion boundary should be contracted inward.
                8. If case_type is "lesion_completely_missed", say the lesion has been completely missed and should be included in the segmentation.
                9. If case_type is "no_lesion_false_positive", say no lesion is present and the segmented region should be removed.
                10. If multiple error regions are provided, combine all corrections into one coherent sentence.
                11. If there is no significant error region, output exactly:
                    "No obvious segmentation correction is needed."

                Example outputs:

                - "Under-segmentation is present in the upper region of the current segmentation, and the boundary should be expanded outward."
                - "Over-segmentation is present in the lower-left region of the current segmentation, and the boundary should be contracted inward."
                - "Over-segmentation is present in the upper-left region of the current segmentation, and the boundary should be contracted inward, while under-segmentation is present in the lower-right region and the boundary should be expanded outward."
                - "The true lesion region has been completely missed and should be included in the segmentation."
                - "The current segmented region is entirely incorrect and should be removed."
                - "The current segmented region is entirely incorrect and should be removed, while the true lesion region has been completely missed and should be included in the segmentation."

                Input structured attributes:
                {json.dumps(text_prompt, ensure_ascii=False, indent=2)}

                Output:
            """
        elif self.task_name == "PolypGen":
            prompt = f"""
                You are an expert reviewer of colorectal polyp segmentation in endoscopic images.

                You will receive structured attributes describing segmentation errors.

                Your task is to convert these attributes into one concise, human-like segmentation refinement instruction.

                Meaning of fields:

                - "case_type" indicates the segmentation situation:
                    - "boundary_correction" means a lesion exists and only the lesion boundary needs correction.
                    - "lesion_completely_missed" means a true lesion exists but the segmentation missed it completely.
                    - "no_lesion_false_positive" means no true lesion is present, but a false segmented region exists.

                - "boundary_direction" indicates which lesion boundary should be corrected.
                - "boundary_adjustment" indicates how the boundary should be modified:
                    - "outward" means the boundary should be expanded outward.
                    - "inward" means the boundary should be contracted inward.

                Meaning of error types:

                - "under-segmentation" means part or all of the lesion is missing.
                - "over-segmentation" means extra non-lesion regions are included.

                Boundary directions:

                - "upper"
                - "lower"
                - "left"
                - "right"
                - "upper-left"
                - "upper-right"
                - "lower-left"
                - "lower-right"

                Writing rules:

                1. Output only one sentence.
                2. Use natural clinical-style language.
                3. Do not mention JSON or structured attributes.
                4. Do not mention model predictions.
                5. Do not mention pathology, diagnosis, benignity, malignancy, or disease type.
                6. If case_type is "boundary_correction" and boundary_adjustment is "outward", say the corresponding lesion boundary should be expanded outward.
                7. If case_type is "boundary_correction" and boundary_adjustment is "inward", say the corresponding lesion boundary should be contracted inward.
                8. If case_type is "lesion_completely_missed", say the lesion has been completely missed and should be included in the segmentation.
                9. If case_type is "no_lesion_false_positive", say no lesion is present and the segmented region should be removed.
                10. If multiple error regions are provided, combine all corrections into one coherent sentence.
                11. If there is no significant error region, output exactly:
                    "No obvious segmentation correction is needed."

                Example outputs:

                - "Under-segmentation is present in the upper region of the current segmentation, and the boundary should be expanded outward."
                - "Over-segmentation is present in the lower-left region of the current segmentation, and the boundary should be contracted inward."
                - "Over-segmentation is present in the upper-left region of the current segmentation, and the boundary should be contracted inward, while under-segmentation is present in the lower-right region and the boundary should be expanded outward."
                - "The true lesion region has been completely missed and should be included in the segmentation."
                - "The current segmented region is entirely incorrect and should be removed."
                - "The current segmented region is entirely incorrect and should be removed, while the true lesion region has been completely missed and should be included in the segmentation."

                Input structured attributes:
                {json.dumps(text_prompt, ensure_ascii=False, indent=2)}

                Output:
            """

        return prompt

    def evaluate_batch(self, text_prompts):
        """
        Use the large language model to mimic a physician in providing
        refinement suggestions for the segmentation results.

        Args:
            text_prompts: Structured text prompts generated from the analysis.

        Returns:
            responses: Generated refinement instructions.
        """
        prompts = []

        for text_prompt in text_prompts:  # Process each sample in the current batch
            prompt = self.build_prompt(text_prompt)  # Embed the structured prompt into the complete prompt template
            messages = [{"role": "user", "content": prompt}]  # Wrap the prompt into the chat message format
            # Convert the messages into the chat template expected by the model
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompts.append(text)  # Complete input for the current sample

        # Tokenize the entire batch
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.llm.device)

        input_len = inputs["input_ids"].shape[
            1]  # Record the input prompt length so that only newly generated tokens are kept

        with torch.inference_mode():  # Generate responses for the entire batch
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=False
            )  # Output token IDs

        responses = []
        for output in outputs:  # Process each generated result in the batch
            generated_ids = output[
                            input_len:]  # Remove the input prompt tokens and keep only the newly generated tokens
            # Decode the token IDs into text and remove special tokens and surrounding whitespace
            response = self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True
            ).strip()
            responses.append(response)  # Add the generated response for the current sample

        return responses






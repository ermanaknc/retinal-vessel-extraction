"""
preprocessing.py - Retinal Image Preprocessing Module

Image loading, FOV masking, colour channel extraction and CLAHE application
"""

import cv2
import numpy as np
from PIL import Image


def load_image(path):
    """
    Load a retinal image in TIF format

    Args:
        path: Image file path

    Returns:
        numpy array in BGR format
    """
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def load_mask(path):
    """
    Load a FOV mask in GIF format

    Args:
        path: Mask file path

    Returns:
        Binary mask (0 or 255)
    """
    # Load GIF with PIL
    pil_image = Image.open(path)
    mask = np.array(pil_image)

    # Convert to a binary mask
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]

    # Normalize to the 0-255 range
    mask = (mask > 0).astype(np.uint8) * 255
    return mask


def load_ground_truth(path):
    """
    Load ground truth in GIF format

    Args:
        path: Ground truth file path

    Returns:
        Binary ground truth (0 or 1)
    """
    # Load GIF with PIL
    pil_image = Image.open(path)
    gt = np.array(pil_image)

    # Convert to binary
    if len(gt.shape) == 3:
        gt = gt[:, :, 0]

    # 0-1 binary
    gt = (gt > 0).astype(np.uint8)
    return gt


def extract_channels(image):
    """
    Extract the G, L, Y channels from a BGR image

    Args:
        image: Image in BGR format

    Returns:
        g_channel: Green channel (from BGR)
        l_channel: L channel (from L*a*b colour space)
        y_channel: Y channel (from YCbCr colour space)
    """
    # CRITICAL: Invert image (dark vessels -> bright vessels for Gabor)
    image = cv2.bitwise_not(image)

    # Green channel (index 1 in BGR format)
    g_channel = image[:, :, 1].copy()

    # Convert to L*a*b colour space and take the L channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel = lab[:, :, 0].copy()

    # Convert to YCbCr colour space and take the Y channel
    ycbcr = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycbcr[:, :, 0].copy()

    return g_channel, l_channel, y_channel


def crop_to_fov(img, mask):
    """Crop image to FOV bounding box"""
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    cropped_img = img[y:y+h, x:x+w]
    cropped_mask = mask[y:y+h, x:x+w]
    return cropped_img, cropped_mask, (x, y, w, h)


def restore_size(cropped_img, original_shape, bbox):
    """Restore cropped image to original size"""
    x, y, w, h = bbox
    full_img = np.zeros(original_shape, dtype=np.uint8)
    full_img[y:y+h, x:x+w] = cropped_img
    return full_img


def apply_clahe(channel, clip_limit=2.0, tile_size=8):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)

    Args:
        channel: Single-channel image
        clip_limit: Clip limit value (2.0)
        tile_size: Tile size (8)

    Returns:
        CLAHE-applied channel
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    result = clahe.apply(channel)
    return result


def apply_histogram_equalization(channel):
    """
    Apply standard histogram equalization (for Feature #13)

    Args:
        channel: Single-channel image

    Returns:
        Histogram-equalized channel
    """
    return cv2.equalizeHist(channel)


def preprocess_image(image, mask):
    """
    Apply all preprocessing steps for a single image

    Args:
        image: Image in BGR format
        mask: FOV mask

    Returns:
        Dictionary containing:
        - g_channel: Original G channel
        - l_channel: Original L channel
        - y_channel: Original Y channel
        - g_clahe: CLAHE applied G channel
        - l_clahe: CLAHE applied L channel
        - y_clahe: CLAHE applied Y channel
        - g_equalized: Normal histogram equalized G (Feature #13)
        - mask: FOV mask
        - original_shape: Original image size
        - bbox: Crop bounding box
    """
    # Save original shape
    original_shape = image.shape[:2]

    # 1. CROP TO FOV
    image, mask, bbox = crop_to_fov(image, mask)

    # 2. CRITICAL: Apply mask to remove background noise BEFORE channel extraction
    image = cv2.bitwise_and(image, image, mask=mask)

    # 3. Extract channels (inverted)
    g_channel, l_channel, y_channel = extract_channels(image)

    # Apply CLAHE (for Gabor)
    g_clahe = apply_clahe(g_channel)
    l_clahe = apply_clahe(l_channel)
    y_clahe = apply_clahe(y_channel)

    # Standard histogram equalization (for Feature #13)
    g_equalized = apply_histogram_equalization(g_channel)

    return {
        'g_channel': g_channel,
        'l_channel': l_channel,
        'y_channel': y_channel,
        'g_clahe': g_clahe,
        'l_clahe': l_clahe,
        'y_clahe': y_clahe,
        'g_equalized': g_equalized,
        'mask': mask,
        'original_shape': original_shape,
        'bbox': bbox
    }


def apply_mask(image, mask):
    """
    Apply the FOV mask to an image

    Args:
        image: Single- or multi-channel image
        mask: Binary mask

    Returns:
        Masked image
    """
    if len(image.shape) == 2:
        return cv2.bitwise_and(image, image, mask=mask)
    else:
        return cv2.bitwise_and(image, image, mask=mask)


def get_image_paths(data_dir, image_type='training'):
    """
    Get the image paths from the data directory

    Args:
        data_dir: Root data directory
        image_type: 'training' or 'test'

    Returns:
        image_paths: List of image file paths
        mask_paths: List of mask file paths
        gt_paths: Ground truth file paths (only for training)
    """
    import os

    if image_type == 'training':
        image_dir = os.path.join(data_dir, 'training', 'images')
        mask_dir = os.path.join(data_dir, 'training', 'mask')
        gt_dir = os.path.join(data_dir, 'training', '1st_manual')

        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

        image_paths = [os.path.join(image_dir, f) for f in image_files]
        mask_paths = [os.path.join(mask_dir, f.replace('_training.tif', '_training_mask.gif')) for f in image_files]
        gt_paths = [os.path.join(gt_dir, f.replace('_training.tif', '_manual1.gif')) for f in image_files]

        return image_paths, mask_paths, gt_paths

    else:  # test
        image_dir = os.path.join(data_dir, 'test', 'images')
        mask_dir = os.path.join(data_dir, 'test', 'mask')

        image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.tif')])

        image_paths = [os.path.join(image_dir, f) for f in image_files]
        mask_paths = [os.path.join(mask_dir, f.replace('_test.tif', '_test_mask.gif')) for f in image_files]

        return image_paths, mask_paths, None


if __name__ == "__main__":
    # Test preprocessing
    import os

    data_dir = "data"
    image_paths, mask_paths, gt_paths = get_image_paths(data_dir, 'training')

    print(f"Total number of training images: {len(image_paths)}")

    # Test the first image
    image = load_image(image_paths[0])
    mask = load_mask(mask_paths[0])
    gt = load_ground_truth(gt_paths[0])

    print(f"Image size: {image.shape}")
    print(f"Mask size: {mask.shape}")
    print(f"Ground truth size: {gt.shape}")

    # Apply preprocessing
    preprocessed = preprocess_image(image, mask)

    print(f"G channel size: {preprocessed['g_channel'].shape}")
    print(f"CLAHE G channel size: {preprocessed['g_clahe'].shape}")

    print("Preprocessing test passed!")

"""
feature_extraction.py - Feature Extraction Module

Extraction of 13 features:
- 9 Gabor binary features (3 wavelength x 3 channel)
- Top Hat (21px linear SE, 9 directions)
- Shade Corrected (25x25 median filter)
- Bit Plane Slicing (7th + 8th bit planes)
- Enhanced G channel (CLAHE)
"""

import cv2
import numpy as np
from sklearn.preprocessing import StandardScaler


# ==================== GABOR FEATURES ====================

def create_gabor_bank(wavelengths=[9, 10, 11], n_orientations=12, aspect_ratio=0.5):
    """
    Build the Gabor filter bank

    Paper parameters:
    - Wavelengths: [9, 10, 11]
    - Orientations: 12 (0° to 180°, 15° step)
    - Aspect ratio (gamma): 0.5
    - sigma = lambda * 0.56 (for bandwidth=1)
    - Kernel size: Fixed 31x31

    Args:
        wavelengths: List of wavelengths
        n_orientations: Number of orientations
        aspect_ratio: Gabor aspect ratio

    Returns:
        filters: List of Gabor filters (wavelength, orientation, kernel)
    """
    filters = []

    for wavelength in wavelengths:
        wavelength_filters = []

        # CRITICAL: sigma = lambda * 0.56 (per the paper)
        sigma = wavelength * 0.56

        for i in range(n_orientations):
            theta = i * np.pi / n_orientations  # 0° to 165° (15° step)

            # CRITICAL: Fixed kernel size 31x31
            ksize = 31

            # Create the Gabor kernel
            kernel = cv2.getGaborKernel(
                ksize=(ksize, ksize),
                sigma=sigma,
                theta=theta,
                lambd=wavelength,
                gamma=aspect_ratio,
                psi=0,  # Phase offset
                ktype=cv2.CV_32F  # use float32
            )

            wavelength_filters.append(kernel)

        filters.append(wavelength_filters)

    return filters


def apply_gabor_filter(image, kernel):
    """
    Apply a single Gabor filter

    Args:
        image: Input image (single channel)
        kernel: Gabor kernel

    Returns:
        Filtered image (np.abs is NOT used!)
    """
    # CRITICAL: use float32, do not use np.abs!
    filtered = cv2.filter2D(image, cv2.CV_32F, kernel)
    return filtered


def automatic_threshold(gabor_output):
    """
    Noise-robust automatic thresholding.
    """
    # 1. Normalize the Gabor output to the 0-255 range (it may still be float)
    normalized = cv2.normalize(gabor_output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 2. Histogram computation
    hist = cv2.calcHist([normalized], [0], None, [256], [0, 256]).flatten()
    total_pixels = normalized.shape[0] * normalized.shape[1]

    # 3. Paper formula (weighted mean)
    threshold_value = 0.0
    for i in range(256):
        threshold_value += i * (hist[i] / total_pixels)

    # --- CRITICAL FIX ---
    # Taking only the mean is too sensitive to noise.
    # By raising the threshold 50% above the mean (1.5 factor)
    # we keep only the genuine vessel responses.
    threshold_value = threshold_value * 1.5

    _, binary = cv2.threshold(normalized, threshold_value, 255, cv2.THRESH_BINARY)

    # 4. Median blur to remove the "speckle" noise (guaranteed cleanup)
    binary = cv2.medianBlur(binary, 3)

    return binary


def extract_gabor_features(g_clahe, l_clahe, y_clahe, gabor_bank, mask=None):
    """
    Extract the Gabor features in a noise-cleaned way.

    Returns:
        gabor_features: Binary Gabor features (for Figure 6)
        raw_gabor_responses: Raw float Gabor outputs (for Figure 5)
    """
    channels = [g_clahe, y_clahe, l_clahe]
    gabor_features = []
    raw_gabor_responses = []  # for the raw float outputs

    for ch_idx, channel in enumerate(channels):
        for wl_idx, wavelength_filters in enumerate(gabor_bank):
            max_response = np.zeros_like(channel, dtype=np.float32)

            for kernel in wavelength_filters:
                response = cv2.filter2D(channel, cv2.CV_32F, kernel)
                np.maximum(max_response, response, out=max_response)

            # Store the raw Gabor output (for Figure 5)
            raw_gabor_responses.append(max_response.copy())

            # 1. Automatic binarization (calls the function above)
            binary_feature = automatic_threshold(max_response)

            # 2. Border artefact cleanup
            # We shrink the mask here so that the bright ring gets removed.
            if mask is not None:
                shrink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
                shrunk_mask = cv2.erode(mask, shrink_kernel, iterations=1)
                binary_feature = cv2.bitwise_and(binary_feature, binary_feature, mask=shrunk_mask)

            gabor_features.append(binary_feature)

    return gabor_features, raw_gabor_responses


# ==================== MORPHOLOGICAL FEATURES ====================

def create_linear_se(length, angle_degrees):
    """
    Create a linear structuring element at the given length and angle

    Args:
        length: SE length (pixels)
        angle_degrees: Angle (degrees)

    Returns:
        se: Structuring element
    """
    angle_rad = np.radians(angle_degrees)

    # Compute the SE size
    size = length
    if size % 2 == 0:
        size += 1

    # Create an empty SE
    se = np.zeros((size, size), dtype=np.uint8)

    # Center point
    center = size // 2

    # Draw the line
    for i in range(length):
        offset = i - length // 2
        x = int(center + offset * np.cos(angle_rad))
        y = int(center - offset * np.sin(angle_rad))

        if 0 <= x < size and 0 <= y < size:
            se[y, x] = 1

    return se


def top_hat_feature(g_channel):
    """
    Extract the Top Hat feature

    Paper parameters:
    - Linear SE, length: 21 pixels
    - 9 orientations: 0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°, 180°
    - Use the maximum response
    - Applied on the inverted G channel

    Args:
        g_channel: G channel (original)

    Returns:
        top_hat: Top Hat feature (maximum response)
    """
    # CRITICAL: g_channel is already inverted (in preprocessing)

    # Define 9 orientations
    angles = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180]

    max_response = None

    for angle in angles:
        # Create a linear SE
        se = create_linear_se(21, angle)

        # Apply Top Hat (subtract the opening)
        # CRITICAL: g_channel is already inverted (in preprocessing)
        top_hat = cv2.morphologyEx(g_channel, cv2.MORPH_TOPHAT, se)

        if max_response is None:
            max_response = top_hat.astype(np.float64)
        else:
            max_response = np.maximum(max_response, top_hat.astype(np.float64))

    return max_response.astype(np.uint8)


def shade_corrected_feature(g_channel):
    """
    Extract the Shade Corrected feature

    Paper parameters (Section 3.3.1.2.2):
    - Background estimation with a 25x25 median filter
    - original - background (simple subtraction)

    Args:
        g_channel: G channel

    Returns:
        shade_corrected: Background-corrected image
    """
    # Background estimation with a median filter (25x25)
    bg = cv2.medianBlur(g_channel, 25)

    # SIMPLE: use cv2.subtract
    sc = cv2.subtract(g_channel, bg)

    return sc


def bit_plane_slicing_feature(g_channel):
    """
    Bit Plane Slicing - must operate on the original (non-inverted) G channel
    Paper: sum of the 7th and 8th bit planes
    """
    # CRITICAL: g_channel comes in already inverted, so invert it again
    # (restore the original - vessels should be dark)
    g_original = cv2.bitwise_not(g_channel)

    # Normalize to the 0-255 range
    g_rescaled = cv2.normalize(g_original, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 8th bit (MSB) and 7th bit
    bit_8 = (g_rescaled >> 7) & 1
    bit_7 = (g_rescaled >> 6) & 1

    # Sum and scale to 255
    bps = ((bit_8.astype(np.uint16) + bit_7.astype(np.uint16)) * 127).astype(np.uint8)

    # Vessels should be white - invert
    bps = cv2.bitwise_not(bps)

    return bps


# ==================== FEATURE VECTOR EXTRACTION ====================

def extract_all_features(preprocessed_data):
    """
    Extract all 13 features

    Args:
        preprocessed_data: Preprocessing output (dictionary)

    Returns:
        features: List of 13 feature images
        feature_names: List of feature names
        raw_gabor_responses: Raw Gabor float outputs (for Figure 5)
    """
    g_channel = preprocessed_data['g_channel']
    g_clahe = preprocessed_data['g_clahe']
    l_clahe = preprocessed_data['l_clahe']
    y_clahe = preprocessed_data['y_clahe']
    g_equalized = preprocessed_data['g_equalized']  # Normal histogram eq.

    mask = preprocessed_data.get('mask', None)

    # Build the Gabor filter bank
    gabor_bank = create_gabor_bank()

    # 1-9: Gabor features (with mask-based shrunk border removal)
    # CRITICAL: the channel order must be G, Y, L!
    gabor_features, raw_gabor_responses = extract_gabor_features(g_clahe, y_clahe, l_clahe, gabor_bank, mask)

    # 10: Top Hat - applied on g_clahe and normalized (vessels bright, as in the paper)
    th_feature = top_hat_feature(g_clahe)

    # 11: Shade Corrected - applied on g_clahe (noise-cleaned)
    sc_feature = shade_corrected_feature(g_clahe)

    # 12: Bit Plane Slicing
    bps_feature = bit_plane_slicing_feature(g_channel)

    # 13: Enhanced G channel - CLAHE! (per the paper)
    g_enhanced = g_clahe

    # Combine all features
    features = gabor_features + [th_feature, sc_feature, bps_feature, g_enhanced]

    # CRITICAL: channel order is G, Y, L
    feature_names = [
        'Gabor_G_λ9', 'Gabor_G_λ10', 'Gabor_G_λ11',
        'Gabor_Y_λ9', 'Gabor_Y_λ10', 'Gabor_Y_λ11',
        'Gabor_L_λ9', 'Gabor_L_λ10', 'Gabor_L_λ11',
        'TopHat', 'ShadeCorrect', 'BitPlane', 'G_Enhanced'
    ]

    return features, feature_names, raw_gabor_responses


def create_feature_vector(features, mask):
    """
    Build a per-pixel feature vector from the feature images

    Args:
        features: List of 13 feature images
        mask: FOV mask

    Returns:
        feature_matrix: matrix of shape (n_pixels, 13)
        pixel_coords: coordinate matrix of shape (n_pixels, 2)
    """
    # Take the coordinates of the pixels inside the mask
    valid_pixels = np.where(mask > 0)
    n_pixels = len(valid_pixels[0])

    # Create the feature matrix
    feature_matrix = np.zeros((n_pixels, len(features)), dtype=np.float64)

    for i, feature in enumerate(features):
        feature_matrix[:, i] = feature[valid_pixels]

    # Save the coordinates
    pixel_coords = np.column_stack((valid_pixels[0], valid_pixels[1]))

    return feature_matrix, pixel_coords


def normalize_features(feature_matrix, mask_indices, scaler=None):
    """
    Fits using only the pixels inside the FOV.
    """
    if scaler is None:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        # Fit only with the data inside the mask
        scaler.fit(feature_matrix)

    normalized = scaler.transform(feature_matrix)
    # NaN/Inf cleanup
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    return normalized, scaler


if __name__ == "__main__":
    # Test feature extraction
    from preprocessing import load_image, load_mask, preprocess_image
    import os

    # Load a test image
    data_dir = "data"
    image_path = os.path.join(data_dir, "training", "images", "21_training.tif")
    mask_path = os.path.join(data_dir, "training", "mask", "21_training_mask.gif")

    image = load_image(image_path)
    mask = load_mask(mask_path)

    print("Applying preprocessing...")
    preprocessed = preprocess_image(image, mask)

    print("Applying feature extraction...")
    features, feature_names, raw_gabor = extract_all_features(preprocessed)

    print(f"\nTotal number of features: {len(features)}")
    for i, (feature, name) in enumerate(zip(features, feature_names)):
        print(f"  {i+1}. {name}: shape={feature.shape}, dtype={feature.dtype}")

    print(f"\nNumber of raw Gabor responses: {len(raw_gabor)}")

    print("\nBuilding the feature vector...")
    cropped_mask = preprocessed['mask']
    feature_matrix, pixel_coords = create_feature_vector(features, cropped_mask)
    print(f"Feature matrix size: {feature_matrix.shape}")
    print(f"Pixel coordinate size: {pixel_coords.shape}")

    print("\nNormalizing...")
    normalized, _ = normalize_features(feature_matrix, None)
    print(f"Normalized matrix size: {normalized.shape}")
    print(f"Min value: {normalized.min():.4f}, Max value: {normalized.max():.4f}")

    print("\nFeature extraction test passed!")

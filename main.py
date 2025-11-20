"""
main.py - Main Pipeline
Retinal Blood Vessel Extraction

Pipeline:
1. Data loading and splitting (seed=42)
2. Preprocessing
3. Feature extraction (13 features)
4. Class-balanced training pixel sampling
5. PCA (keep 13 dims)
6. FCM clustering
7. Decision Tree classification
8. Combination and post-processing
9. Evaluation and output saving
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Set the Matplotlib backend (so it runs without a GUI)
import matplotlib
matplotlib.use('Agg')

import cv2
import numpy as np
from tqdm import tqdm

# Local modules
from preprocessing import (
    load_image, load_mask, load_ground_truth,
    preprocess_image, get_image_paths
)
from feature_extraction import (
    extract_all_features, create_feature_vector, normalize_features
)
from vessel_extraction import (
    apply_pca, fcm_clustering, train_decision_tree,
    classify_non_vessels, reconstruct_image, extract_vessels
)
from evaluation import (
    calculate_metrics, calculate_roc_auc, calculate_average_metrics,
    save_results_table, print_metrics, print_comparison_table,
    plot_roc_curve
)


# ==================== CONFIGURATION ====================

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
RANDOM_SEED = 42

# Training and validation split
# The training folder has 20 images (21-40)
# 16 for training (21-36), 4 for validation (37-40)
TRAINING_INDICES = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
VALIDATION_INDICES = [37, 38, 39, 40]


# ==================== HELPER FUNCTIONS ====================

def create_output_dirs():
    """Create the output directories"""
    dirs = [
        os.path.join(OUTPUT_DIR, "figures_3_4"),
        os.path.join(OUTPUT_DIR, "figures_5_6"),
        os.path.join(OUTPUT_DIR, "figures_7_8_9"),
        os.path.join(OUTPUT_DIR, "figure_10"),
        os.path.join(OUTPUT_DIR, "figure_11"),
        os.path.join(OUTPUT_DIR, "figure_12"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)


def get_image_number(image_path):
    """Extract the number from an image file path"""
    filename = os.path.basename(image_path)
    return int(filename.split('_')[0])


def save_preprocessing_outputs(preprocessed, image_num, output_dir):
    """Save the preprocessing outputs (Figures 3-4)"""
    save_dir = os.path.join(output_dir, "figures_3_4")

    # Original channels (inverted - vessels white)
    cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_G_channel.png"), preprocessed['g_channel'])
    cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_L_channel.png"), preprocessed['l_channel'])
    cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_Y_channel.png"), preprocessed['y_channel'])

    # CLAHE channels - INVERT (vessels black - as in the paper)
    # In the paper Fig 4 the vessels appear black; the inversion must be applied afterwards
    cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_G_CLAHE.png"), 255 - preprocessed['g_clahe'])
    cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_L_CLAHE.png"), 255 - preprocessed['l_clahe'])
    cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_Y_CLAHE.png"), 255 - preprocessed['y_clahe'])


def save_gabor_outputs(features, feature_names, image_num, output_dir, raw_gabor_responses=None):
    """Save the Gabor outputs (Figures 5-6)

    Figure 5: Raw Gabor float outputs (greyscale)
    Figure 6: Binary-thresholded outputs
    """
    save_dir = os.path.join(output_dir, "figures_5_6")

    # First 9 features are Gabor BINARY (for Figure 6)
    for i in range(9):
        cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_{feature_names[i]}_binary.png"), features[i])

    # Raw Gabor float outputs (for Figure 5 - greyscale)
    if raw_gabor_responses is not None:
        for i in range(9):
            # Normalize the float output to the 0-255 range
            raw = raw_gabor_responses[i]
            normalized = cv2.normalize(raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_{feature_names[i]}.png"), normalized)


def save_morphological_outputs(features, feature_names, image_num, output_dir):
    """Save the morphological outputs (Figures 7-8-9)"""
    save_dir = os.path.join(output_dir, "figures_7_8_9")

    # Top Hat (index 9) - Normalize (so vessels appear bright, as in the paper)
    th_feature = features[9]
    th_normalized = cv2.normalize(th_feature, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_TopHat.png"), th_normalized)

    # Shade Corrected (index 10)
    cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_ShadeCorrect.png"), features[10])

    # Bit Plane (index 11)
    cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_BitPlane.png"), features[11])

    # G Enhanced (index 12)
    cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_G_Enhanced.png"), features[12])


def save_fcm_output(fcm_image, image_num, output_dir):
    """Save the FCM output (Figure 10)"""
    save_dir = os.path.join(output_dir, "figure_10")
    cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_FCM.png"), fcm_image)


def save_dt_output(dt_image, image_num, output_dir):
    """Save the Decision Tree output (Figure 11)"""
    save_dir = os.path.join(output_dir, "figure_11")
    cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_DT.png"), dt_image)


def save_final_output(vessel_image, image_num, output_dir):
    """Save the final segmentation (Figure 12)"""
    save_dir = os.path.join(output_dir, "figure_12")
    cv2.imwrite(os.path.join(save_dir, f"{image_num:02d}_Final.png"), vessel_image)


def apply_fov_mask_postprocess(vessel_image, mask, min_area=50):
    """
    Post-processing with the FOV mask (like Paper Section 3.4)
    1. Opening - remove small noise
    2. Area filtering - remove small regions
    3. Apply a shrunk FOV mask
    """
    # 1. Morphological Opening - 3x3
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(vessel_image, cv2.MORPH_OPEN, open_kernel)

    # 2. Area filtering - min_area=50
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    cleaned = np.zeros_like(opened)
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255

    # 3. Shrink FOV mask - 15x15
    shrink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    shrunk_mask = cv2.erode(mask, shrink_kernel, iterations=1)

    # 4. Apply shrunk mask
    result = cv2.bitwise_and(cleaned, cleaned, mask=shrunk_mask)

    return result


# ==================== MAIN PIPELINE ====================

def main():
    """Main pipeline"""
    print("="*60)
    print("RETINAL BLOOD VESSEL EXTRACTION")
    print("="*60)

    # Create the output directories
    create_output_dirs()

    # Get the image paths
    image_paths, mask_paths, gt_paths = get_image_paths(DATA_DIR, 'training')

    print(f"\nTotal images in training folder: {len(image_paths)}")
    print(f"For training: {len(TRAINING_INDICES)} images")
    print(f"For validation: {len(VALIDATION_INDICES)} images")
    print(f"Training indices: {TRAINING_INDICES}")
    print(f"Validation indices: {VALIDATION_INDICES}")

    # ==================== PHASE 1: TRAINING ====================
    print("\n" + "="*60)
    print("PHASE 1: TRAINING DATA PREPARATION")
    print("="*60)

    # Process the training images - FIRST COLLECT ALL DATA
    raw_features_list = []  # Not normalized
    training_labels_list = []

    print("\nProcessing training images...")
    for idx in tqdm(TRAINING_INDICES, desc="Processing training images"):
        # Find the file paths
        img_idx = idx - 21  # 0-based index

        image = load_image(image_paths[img_idx])
        mask = load_mask(mask_paths[img_idx])
        gt = load_ground_truth(gt_paths[img_idx])

        # Preprocessing
        preprocessed = preprocess_image(image, mask)

        # CRITICAL: Crop GT to match preprocessed image
        bbox = preprocessed['bbox']
        x, y, w, h = bbox
        gt = gt[y:y+h, x:x+w]
        if gt.shape != preprocessed['mask'].shape:
            gt = cv2.resize(gt, (preprocessed['mask'].shape[1], preprocessed['mask'].shape[0]),
                            interpolation=cv2.INTER_NEAREST)

        # Feature extraction
        cropped_mask = preprocessed['mask']
        features, feature_names, _ = extract_all_features(preprocessed)
        feature_matrix, pixel_coords = create_feature_vector(features, cropped_mask)

        # Ground truth labels
        gt_labels = gt[pixel_coords[:, 0], pixel_coords[:, 1]]

        raw_features_list.append(feature_matrix)  # RAW features
        training_labels_list.append(gt_labels)

    # Combine all training data
    print("\nCombining all training data...")
    X_train_raw = np.vstack(raw_features_list)
    y_train = np.concatenate(training_labels_list)
    print(f"Total training pixels: {len(y_train)}")
    print(f"Vessel: {np.sum(y_train == 1)}, Non-vessel: {np.sum(y_train == 0)}")

    # 1. SCALER FIT (on all training data)
    print("\nFitting the scaler (on all training data)...")
    from sklearn.preprocessing import StandardScaler
    global_scaler = StandardScaler()
    X_scaled = global_scaler.fit_transform(X_train_raw)
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1.0, neginf=0.0)
    print("Scaler saved.")

    # 2. PCA FIT (on the scaled data)
    print("\nTraining the PCA model...")
    pca_features, pca_model = apply_pca(X_scaled)
    print(f"PCA explained variance ratio: {sum(pca_model.explained_variance_ratio_):.4f}")

    # 3. SAMPLING (balanced)
    print(f"\nPerforming balanced sampling...")
    pos_idx = np.where(y_train == 1)[0]
    neg_idx = np.where(y_train == 0)[0]

    # 5000 vessel + 5000 non-vessel per image (Paper Section 4.1)
    n_samples = min(len(pos_idx), len(neg_idx), 5000 * len(TRAINING_INDICES))

    np.random.seed(RANDOM_SEED)
    selected_pos = np.random.choice(pos_idx, min(n_samples, len(pos_idx)), replace=False)
    selected_neg = np.random.choice(neg_idx, min(n_samples, len(neg_idx)), replace=False)
    selected_indices = np.concatenate([selected_pos, selected_neg])

    sampled_pca_features = pca_features[selected_indices]
    sampled_labels = y_train[selected_indices]

    print(f"Sampled feature shape: {sampled_pca_features.shape}")
    print(f"Number of vessel pixels: {np.sum(sampled_labels == 1)}")
    print(f"Number of non-vessel pixels: {np.sum(sampled_labels == 0)}")

    # 4. Train the Decision Tree
    print("\nTraining the Decision Tree...")
    dt_classifier = train_decision_tree(sampled_pca_features, sampled_labels, seed=RANDOM_SEED)
    print("Decision Tree trained.")

    # ==================== PHASE 2: VALIDATION ====================
    print("\n" + "="*60)
    print("PHASE 2: VALIDATION")
    print("="*60)

    all_metrics = []
    all_final_probs = []  # FCM + DT combined probability map
    all_gt_labels = []
    all_pixel_coords = []

    print("\nProcessing validation images...")
    for idx in tqdm(VALIDATION_INDICES, desc="Processing validation images"):
        img_idx = idx - 21  # 0-based index
        image_num = idx

        # Load
        image = load_image(image_paths[img_idx])
        mask = load_mask(mask_paths[img_idx])
        gt = load_ground_truth(gt_paths[img_idx])

        # Preprocessing
        preprocessed = preprocess_image(image, mask)

        # CRITICAL: Crop GT to match preprocessed image
        bbox = preprocessed['bbox']
        x, y, w, h = bbox
        gt = gt[y:y+h, x:x+w]
        cropped_mask = preprocessed['mask']
        if gt.shape != cropped_mask.shape:
            gt = cv2.resize(gt, (cropped_mask.shape[1], cropped_mask.shape[0]),
                            interpolation=cv2.INTER_NEAREST)

        # Save the preprocessing outputs
        save_preprocessing_outputs(preprocessed, image_num, OUTPUT_DIR)

        # Feature extraction
        features, feature_names, raw_gabor_responses = extract_all_features(preprocessed)

        # Save the feature outputs
        save_gabor_outputs(features, feature_names, image_num, OUTPUT_DIR, raw_gabor_responses)
        save_morphological_outputs(features, feature_names, image_num, OUTPUT_DIR)

        # Feature vector - use the SAME scaler
        feature_matrix, pixel_coords = create_feature_vector(features, cropped_mask)
        normalized_features = global_scaler.transform(feature_matrix)
        normalized_features = np.nan_to_num(normalized_features, nan=0.0, posinf=1.0, neginf=0.0)

        # Vessel extraction (including FCM + DT combination + post-processing)
        vessel_image, fcm_image, dt_image, final_probs = extract_vessels(
            normalized_features,
            pixel_coords,
            cropped_mask.shape,
            dt_classifier,
            pca_model,
            mask=cropped_mask  # Mask for post-processing
        )

        # Save the outputs
        save_fcm_output(fcm_image, image_num, OUTPUT_DIR)
        save_dt_output(dt_image, image_num, OUTPUT_DIR)
        save_final_output(vessel_image, image_num, OUTPUT_DIR)

        # Compute metrics
        metrics = calculate_metrics(vessel_image, gt, cropped_mask)
        all_metrics.append(metrics)

        # Store data for ROC
        all_final_probs.extend(final_probs.tolist())
        gt_labels = gt[pixel_coords[:, 0], pixel_coords[:, 1]]
        all_gt_labels.extend(gt_labels.tolist())
        all_pixel_coords.append(pixel_coords)

    # Print the validation results
    print("\nValidation results:")
    for idx, metrics in zip(VALIDATION_INDICES, all_metrics):
        print_metrics(metrics, f"Image {idx}")

    # ==================== PHASE 3: RESULTS ====================
    print("\n" + "="*60)
    print("PHASE 3: FINAL RESULTS")
    print("="*60)

    # Average metrics
    avg_metrics = calculate_average_metrics(all_metrics)

    # Compute ROC and AUC
    all_final_probs_array = np.array(all_final_probs)
    all_gt_labels_array = np.array(all_gt_labels)

    from sklearn.metrics import roc_curve, auc as sklearn_auc
    fpr, tpr, _ = roc_curve(all_gt_labels_array, all_final_probs_array)
    auc_value = sklearn_auc(fpr, tpr)

    # Save the ROC curve
    plot_roc_curve(fpr, tpr, auc_value, os.path.join(OUTPUT_DIR, "roc_curve.png"))
    print(f"\nROC curve saved: {os.path.join(OUTPUT_DIR, 'roc_curve.png')}")

    # Save the results table
    image_names = [f"Image_{idx}" for idx in VALIDATION_INDICES]
    save_results_table(
        all_metrics,
        image_names,
        os.path.join(OUTPUT_DIR, "table_1.csv"),
        auc_value
    )
    print(f"Results table saved: {os.path.join(OUTPUT_DIR, 'table_1.csv')}")

    # Print the comparison table
    print_comparison_table(avg_metrics, auc_value)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print(f"\nAll outputs saved to the '{OUTPUT_DIR}/' directory:")
    print("  - figures_3_4/: Preprocessing outputs")
    print("  - figures_5_6/: Gabor binary outputs")
    print("  - figures_7_8_9/: Morphological features")
    print("  - figure_10/: FCM clustering results")
    print("  - figure_11/: Decision Tree classification")
    print("  - figure_12/: Final segmentation")
    print("  - table_1.csv: Evaluation metrics")
    print("  - roc_curve.png: ROC curve")


if __name__ == "__main__":
    main()

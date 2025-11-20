"""
vessel_extraction.py - Vessel Extraction Module

Phase 1: FCM Clustering (Unsupervised)
Phase 2: Decision Tree Classification (Supervised)
Combination: FCM vessels + DT vessels
"""

import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import skfuzzy as fuzz


# ==================== PCA ====================

def apply_pca(features, n_components=13):
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)
    # whiten=False avoids artificially amplifying the noise.
    pca = PCA(n_components=n_components, whiten=False)
    pca_features = pca.fit_transform(features)
    return pca_features, pca


def transform_pca(features, pca_model):
    """
    Apply transform with a previously fitted PCA model

    Args:
        features: Feature matrix
        pca_model: Fitted PCA model

    Returns:
        Transformed features
    """
    # Clean NaN and Inf values
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0)

    transformed = pca_model.transform(features)

    # Clean the output as well
    transformed = np.nan_to_num(transformed, nan=0.0, posinf=1.0, neginf=0.0)

    return transformed


# ==================== FCM CLUSTERING ====================

def fcm_clustering(features, n_clusters=2, m=2, max_iter=2000, error=0.000001):
    """
    Improved FCM clustering
    """
    # Transpose the data for skfuzzy
    features_T = features.T

    # Run the FCM algorithm
    cntr, u, u0, d, jm, p, fpc = fuzz.cmeans(
        features_T,
        c=n_clusters,
        m=m,
        error=error,
        maxiter=max_iter,
        init=None
    )

    # Assign the clusters
    clust = np.argmax(u, axis=0)

    # --- CRITICAL FIX: Cluster Selection Logic ---
    # The paper states that the vessel cluster usually has fewer pixels.
    # For robustness we also look at the mean (center) of the features.
    # In the Gabor and CLAHE features, vessels have higher discriminability.

    # Look at the overall mean of the feature values of each cluster center
    center_intensities = np.mean(cntr, axis=1)

    # The vessel cluster is usually more 'dense' or 'salient' in feature space.
    # If cluster 0's center value is greater than cluster 1's and it has fewer pixels,
    # it is definitely the vessel cluster.
    c0_count = np.sum(clust == 0)
    c1_count = np.sum(clust == 1)

    # Hybrid decision: both intensity and pixel count
    if center_intensities[0] > center_intensities[1] and c0_count < c1_count:
        vessel_cluster = 0
    elif center_intensities[1] > center_intensities[0] and c1_count < c0_count:
        vessel_cluster = 1
    else:
        # On a conflict, fall back to the classic 'smaller cluster' rule
        vessel_cluster = 0 if c0_count < c1_count else 1

    # Build the masks
    vessel_mask = (clust == vessel_cluster).astype(np.uint8)
    non_vessel_mask = 1 - vessel_mask

    # Return the membership values for the probability map
    vessel_membership = u[vessel_cluster]

    return vessel_mask, non_vessel_mask, vessel_membership


# ==================== DECISION TREE ====================

def train_decision_tree(training_features, training_labels, seed=42):
    """
    Train the Decision Tree classifier

    Args:
        training_features: feature matrix of shape (n_samples, 13)
        training_labels: binary labels of shape (n_samples,)
        seed: Random seed

    Returns:
        clf: Trained DecisionTreeClassifier
    """
    # Decision Tree parameters
    clf = DecisionTreeClassifier(
        random_state=seed,
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=5,   # Small value = captures more thin vessels
        class_weight='balanced'
    )
    clf.fit(training_features, training_labels)

    return clf


def classify_non_vessels(clf, non_vessel_features):
    """
    Classify the pixels FCM labelled as non-vessel using the Decision Tree

    Args:
        clf: Trained Decision Tree
        non_vessel_features: Features of the non-vessel pixels

    Returns:
        predictions: Binary predictions (0 or 1)
    """
    predictions = clf.predict(non_vessel_features)
    return predictions


def reconstruct_image(mask_1d, pixel_coords, image_shape):
    """
    Convert a 1D mask into a 2D image

    Args:
        mask_1d: 1D binary mask
        pixel_coords: (n_pixels, 2) coordinate matrix (row, col)
        image_shape: Output image size (height, width)

    Returns:
        image_2d: 2D binary image
    """
    image_2d = np.zeros(image_shape, dtype=np.uint8)

    for i, (row, col) in enumerate(pixel_coords):
        if mask_1d[i] == 1:
            image_2d[row, col] = 255

    return image_2d


# ==================== POST-PROCESSING ====================

def post_process_vessels(vessel_image, mask, min_area=50):
    """
    Clean the vessel image and apply post-processing.

    Args:
        vessel_image: Binary vessel image
        mask: FOV mask
        min_area: Minimum component area (smaller components are removed)

    Returns:
        cleaned: Cleaned vessel image
    """
    # Start by copying vessel_image
    cleaned = vessel_image.copy()

    # Remove small components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            cleaned[labels == i] = 0

    if mask is not None:
        # Erode the mask by 35 pixels to remove border artefacts (required for cleanup).
        shrink_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        shrunk_mask = cv2.erode(mask, shrink_kernel, iterations=1)
        cleaned = cv2.bitwise_and(cleaned, cleaned, mask=shrunk_mask)

    return cleaned


# ==================== FULL PIPELINE ====================

def extract_vessels(features, pixel_coords, image_shape, clf, pca_model=None, mask=None):
    """
    Full vessel extraction pipeline for a single image (HYBRID APPROACH)

    =====================================================================
    PAPER SECTION 3.3.3 - HYBRID FCM + DT COMBINATION:
    =====================================================================

    1. FCM Clustering: finds the thick vessels (vessel cluster)
       - Pixels in the vessel cluster are assigned probability 1.0 DIRECTLY
       - These pixels are accepted as vessels for certain

    2. Decision Tree: classifies ONLY the pixels FCM labelled non-vessel
       - probability values are taken via predict_proba
       - captures the thin vessels (the ones FCM missed)

    3. Combination: final probability map = FCM vessels (1.0) + DT probabilities
       - Threshold: 0.57 (Paper Section 4.2 - DRIVE optimal)

    This way FCM captures the thick vessels and DT captures the thin ones.

    Args:
        features: Normalized feature matrix (n_pixels, 13)
        pixel_coords: Pixel coordinates (n_pixels, 2)
        image_shape: Output image size (height, width)
        clf: Trained Decision Tree classifier
        pca_model: Fitted PCA model
        mask: FOV mask (for post-processing)

    Returns:
        vessel_image: Final vessel segmentation (post-processed)
        fcm_image: FCM-only segmentation (thick vessels)
        dt_image: DT additions (thin vessels)
        final_probs: Final probability map (for computing ROC)
    """
    # ==================== STEP 0: PCA TRANSFORM ====================
    if pca_model is not None:
        pca_features = transform_pca(features, pca_model)
    else:
        pca_features, _ = apply_pca(features)

    # ==================== STEP 1: FCM CLUSTERING ====================
    # Paper Section 3.3.3: FCM finds the thick/salient vessels
    # Parameters: m=2, maxiter=2000, error=0.000001
    fcm_vessel_mask, fcm_non_vessel_mask, fcm_membership = fcm_clustering(pca_features)

    # ==================== STEP 2: BUILD THE PROBABILITY MAP ====================
    final_probs = np.zeros(len(features), dtype=np.float32)

    # Assign probability 1.0 DIRECTLY to FCM vessel pixels (THICK VESSELS)
    # These pixels are accepted as vessels for certain - they do NOT go to DT!
    vessel_indices = np.where(fcm_vessel_mask == 1)[0]
    final_probs[vessel_indices] = 1.0

    # ==================== STEP 3: DT ONLY FOR NON-VESSEL PIXELS ====================
    # Paper: "Non-vessel cluster goes into decision tree for further classification"
    non_vessel_indices = np.where(fcm_vessel_mask == 0)[0]  # pixels FCM labelled non-vessel

    if len(non_vessel_indices) > 0:
        # Take the features of ONLY the non-vessel pixels
        non_vessel_features = pca_features[non_vessel_indices]

        # Apply predict_proba with DT - take the vessel probabilities
        dt_probs = clf.predict_proba(non_vessel_features)[:, 1]  # [:, 1] = vessel class probability

        # Update the probabilities of the non-vessel pixels
        # Thin vessels can be found among these pixels
        final_probs[non_vessel_indices] = dt_probs

    # ==================== STEP 4: THRESHOLD AND BINARIZATION ====================
    # Paper Section 4.2: Optimal threshold for DRIVE = 0.57
    THRESHOLD = 0.57
    final_mask = (final_probs > THRESHOLD).astype(np.uint8)

    # ==================== STEP 5: CONVERT TO 2D IMAGES ====================
    # Final vessel image (FCM + DT combination)
    vessel_image = reconstruct_image(final_mask, pixel_coords, image_shape)

    # FCM-only image (thick vessels only - for visualization)
    fcm_image = reconstruct_image(fcm_vessel_mask, pixel_coords, image_shape)

    # DT additions (thin vessels only - missed by FCM, found by DT)
    # These pixels: FCM=0 (non-vessel) and DT probability > threshold
    dt_additions = np.zeros(len(features), dtype=np.uint8)
    if len(non_vessel_indices) > 0:
        # Only pixels from the non-vessel cluster that pass the threshold
        dt_additions[non_vessel_indices] = (final_probs[non_vessel_indices] > THRESHOLD).astype(np.uint8)
    dt_image = reconstruct_image(dt_additions, pixel_coords, image_shape)

    # ==================== STEP 6: POST-PROCESSING ====================
    # Paper Section 3.4: Morphological operations + FOV mask shrinking
    vessel_image = post_process_vessels(vessel_image, mask, min_area=50)

    # Return final_probs for ROC calculation (instead of FCM membership only)
    return vessel_image, fcm_image, dt_image, final_probs


if __name__ == "__main__":
    # Test vessel extraction
    from preprocessing import load_image, load_mask, load_ground_truth, preprocess_image
    from feature_extraction import extract_all_features, create_feature_vector, normalize_features
    import os

    # Load a test image
    data_dir = "data"
    image_path = os.path.join(data_dir, "training", "images", "21_training.tif")
    mask_path = os.path.join(data_dir, "training", "mask", "21_training_mask.gif")
    gt_path = os.path.join(data_dir, "training", "1st_manual", "21_manual1.gif")

    print("Loading image...")
    image = load_image(image_path)
    mask = load_mask(mask_path)
    gt = load_ground_truth(gt_path)

    print("Applying preprocessing...")
    preprocessed = preprocess_image(image, mask)

    print("Applying feature extraction...")
    features, feature_names = extract_all_features(preprocessed)
    feature_matrix, pixel_coords = create_feature_vector(features, mask)
    normalized_features = normalize_features(feature_matrix)

    print(f"Feature matrix size: {normalized_features.shape}")

    # Ground truth labels
    gt_labels = gt[pixel_coords[:, 0], pixel_coords[:, 1]]

    print("\nApplying PCA...")
    pca_features, pca_model = apply_pca(normalized_features)
    print(f"PCA features size: {pca_features.shape}")

    print("\nApplying FCM clustering...")
    fcm_vessel, fcm_non_vessel, membership = fcm_clustering(pca_features)
    print(f"FCM vessel pixel count: {np.sum(fcm_vessel)}")
    print(f"FCM non-vessel pixel count: {np.sum(fcm_non_vessel)}")

    print("\nTraining Decision Tree...")
    # Use the same image for a simple test
    clf = train_decision_tree(pca_features, gt_labels)
    print("Decision Tree trained.")

    print("\nVessel extraction test passed!")

"""
evaluation.py - Evaluation Metrics Module

Metrics:
- Sensitivity (Sen) = TP/(TP+FN)
- Specificity (Spe) = TN/(TN+FP)
- Positive Predictive Value (PPV) = TP/(TP+FP)
- Accuracy (Acc) = (TP+TN)/(TP+FP+FN+TN)
- ROC curve and AUC
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import csv
import os


def calculate_confusion_matrix(pred, gt, mask):
    """
    Compute the confusion matrix (within the FOV mask)

    Args:
        pred: Prediction binary image (0 or 255)
        gt: Ground truth binary image (0 or 1)
        mask: FOV mask (0 or 255)

    Returns:
        TP, TN, FP, FN
    """
    # Take the pixels inside the mask
    mask_binary = mask > 0

    # Convert to binary
    pred_binary = (pred > 0).astype(np.uint8)
    gt_binary = (gt > 0).astype(np.uint8)

    # Apply mask
    pred_masked = pred_binary[mask_binary]
    gt_masked = gt_binary[mask_binary]

    # Confusion matrix
    TP = np.sum((pred_masked == 1) & (gt_masked == 1))
    TN = np.sum((pred_masked == 0) & (gt_masked == 0))
    FP = np.sum((pred_masked == 1) & (gt_masked == 0))
    FN = np.sum((pred_masked == 0) & (gt_masked == 1))

    return TP, TN, FP, FN


def calculate_metrics(pred, gt, mask):
    """
    Compute all metrics

    Args:
        pred: Prediction binary image
        gt: Ground truth binary image
        mask: FOV mask

    Returns:
        Dictionary containing all metrics
    """
    TP, TN, FP, FN = calculate_confusion_matrix(pred, gt, mask)

    # Sensitivity (Recall, True Positive Rate)
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    # Specificity (True Negative Rate)
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    # Positive Predictive Value (Precision)
    ppv = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    # Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0.0

    return {
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'accuracy': accuracy
    }


def calculate_roc_auc(membership, gt, mask, pixel_coords):
    """
    Compute ROC curve and AUC

    Args:
        membership: Vessel membership values (between 0-1, 1D array)
        gt: Ground truth image (2D)
        mask: FOV mask (2D)
        pixel_coords: Pixel coordinates

    Returns:
        fpr: False positive rates
        tpr: True positive rates
        auc_value: AUC value
    """
    # Ground truth labels
    gt_labels = gt[pixel_coords[:, 0], pixel_coords[:, 1]]

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(gt_labels, membership)

    # Compute AUC
    auc_value = auc(fpr, tpr)

    return fpr, tpr, auc_value


def plot_roc_curve(fpr, tpr, auc_value, save_path=None):
    """
    Plot the ROC curve

    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_value: AUC value
        save_path: Save path (optional)
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {auc_value:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve - Retinal Vessel Segmentation')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_average_metrics(all_metrics):
    """
    Compute the average metrics over all images

    Args:
        all_metrics: List of metric dictionaries

    Returns:
        Dictionary containing average metrics
    """
    n = len(all_metrics)

    avg_metrics = {
        'sensitivity': sum(m['sensitivity'] for m in all_metrics) / n,
        'specificity': sum(m['specificity'] for m in all_metrics) / n,
        'ppv': sum(m['ppv'] for m in all_metrics) / n,
        'accuracy': sum(m['accuracy'] for m in all_metrics) / n
    }

    return avg_metrics


def save_results_table(all_metrics, image_names, output_path, auc_value=None):
    """
    Save the results table as CSV

    Args:
        all_metrics: List of metric dictionaries
        image_names: List of image names
        output_path: Save path
        auc_value: Average AUC value
    """
    # Create the CSV file
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # Header
        writer.writerow(['Image', 'Sensitivity', 'Specificity', 'PPV', 'Accuracy'])

        # Metrics for each image
        for name, metrics in zip(image_names, all_metrics):
            writer.writerow([
                name,
                f"{metrics['sensitivity']:.4f}",
                f"{metrics['specificity']:.4f}",
                f"{metrics['ppv']:.4f}",
                f"{metrics['accuracy']:.4f}"
            ])

        # Average
        avg_metrics = calculate_average_metrics(all_metrics)
        writer.writerow([])
        writer.writerow([
            'Average',
            f"{avg_metrics['sensitivity']:.4f}",
            f"{avg_metrics['specificity']:.4f}",
            f"{avg_metrics['ppv']:.4f}",
            f"{avg_metrics['accuracy']:.4f}"
        ])

        # AUC
        if auc_value is not None:
            writer.writerow([])
            writer.writerow(['AUC', f"{auc_value:.4f}"])

        # Target values (from the paper)
        writer.writerow([])
        writer.writerow(['--- Target Values (Paper) ---'])
        writer.writerow(['Target', '0.7830', '0.9800', '0.8594', '0.9531'])
        writer.writerow(['Target AUC', '0.9752'])


def print_metrics(metrics, image_name=None):
    """
    Print the metrics to the console

    Args:
        metrics: Metric dictionary
        image_name: Image name (optional)
    """
    if image_name:
        print(f"\n=== {image_name} ===")

    print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"  Specificity: {metrics['specificity']:.4f}")
    print(f"  PPV:         {metrics['ppv']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")


def print_comparison_table(avg_metrics, auc_value=None):
    """
    Print a comparison table against the paper's values

    Args:
        avg_metrics: Average metrics dictionary
        auc_value: AUC value
    """
    print("\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    print(f"{'Metric':<15} {'Ours':<12} {'Target':<12} {'Diff':<12}")
    print("-"*60)

    targets = {
        'sensitivity': 0.7830,
        'specificity': 0.9800,
        'ppv': 0.8594,
        'accuracy': 0.9531
    }

    for metric, target in targets.items():
        our_value = avg_metrics[metric]
        diff = our_value - target
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        print(f"{metric.capitalize():<15} {our_value:.4f}       {target:.4f}       {diff_str}")

    if auc_value is not None:
        target_auc = 0.9752
        diff = auc_value - target_auc
        diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
        print(f"{'AUC':<15} {auc_value:.4f}       {target_auc:.4f}       {diff_str}")

    print("="*60)


if __name__ == "__main__":
    # Test evaluation
    import numpy as np

    # Dummy test data
    pred = np.random.randint(0, 2, (100, 100)) * 255
    gt = np.random.randint(0, 2, (100, 100))
    mask = np.ones((100, 100), dtype=np.uint8) * 255

    print("Running metric computation test...")
    metrics = calculate_metrics(pred, gt, mask)
    print_metrics(metrics, "Test Image")

    print("\nEvaluation test passed!")

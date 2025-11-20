"""
create_figures.py - Script that generates the paper's figures

Figure 3: G, Y, L channels
Figure 4: CLAHE outputs (G, Y, L)
Figure 5: Gabor filter outputs (3x3 grid: G/Y/L rows, λ9/λ10/λ11 columns)
Figure 6: Gabor binary (thresholded) outputs
Figure 7: Top Hat feature (input + TH)
Figure 8: Shade Correction feature (input + SC)
Figure 9: Bit Plane Slicing (8 bit planes)
Figure 10: FCM Clustering output (input + FCM)
Figure 11: Clustering and Classification (input + FCM + combined)
Figure 12: Final post-processing (input + before + after)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

# Matplotlib backend
import matplotlib
matplotlib.use('Agg')

# ==================== CONFIGURATION ====================
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
FIGURES_DIR = "figures"
SAMPLE_IMAGE_NUM = 36  # Sample image number

# ==================== HELPER FUNCTIONS ====================

def load_original_image(image_num):
    """Load the original image"""
    path = os.path.join(DATA_DIR, "training", "images", f"{image_num}_training.tif")
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def load_mask(image_num):
    """Load the FOV mask"""
    path = os.path.join(DATA_DIR, "training", "mask", f"{image_num}_training_mask.gif")
    pil_image = Image.open(path)
    mask = np.array(pil_image)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    return (mask > 0).astype(np.uint8) * 255

def load_output_image(subfolder, filename):
    """Load an output image"""
    path = os.path.join(OUTPUT_DIR, subfolder, filename)
    if os.path.exists(path):
        return cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None

def crop_to_fov(img, mask):
    """Crop to the FOV"""
    coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(coords)
    if len(img.shape) == 3:
        return img[y:y+h, x:x+w]
    return img[y:y+h, x:x+w]

def create_figure_3(image_num, save_path):
    """
    Figure 3: Selected three channels
    (a) G channel, (b) Y channel, (c) L channel
    """
    g_channel = load_output_image("figures_3_4", f"{image_num:02d}_G_channel.png")
    y_channel = load_output_image("figures_3_4", f"{image_num:02d}_Y_channel.png")
    l_channel = load_output_image("figures_3_4", f"{image_num:02d}_L_channel.png")

    if g_channel is None or y_channel is None or l_channel is None:
        print(f"Required files for Figure 3 not found (image {image_num})")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(g_channel, cmap='gray')
    axes[0].set_title('(a) G channel', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(y_channel, cmap='gray')
    axes[1].set_title('(b) Y channel', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(l_channel, cmap='gray')
    axes[2].set_title('(c) L channel', fontsize=12)
    axes[2].axis('off')

    plt.suptitle('Fig. 3. Selected three channels from different color spaces', fontsize=14, y=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure 3 saved: {save_path}")

def create_figure_4(image_num, save_path):
    """
    Figure 4: Outputs of histogram equalization (CLAHE)
    (a) G CLAHE, (b) Y CLAHE, (c) L CLAHE
    """
    g_clahe = load_output_image("figures_3_4", f"{image_num:02d}_G_CLAHE.png")
    y_clahe = load_output_image("figures_3_4", f"{image_num:02d}_Y_CLAHE.png")
    l_clahe = load_output_image("figures_3_4", f"{image_num:02d}_L_CLAHE.png")

    if g_clahe is None or y_clahe is None or l_clahe is None:
        print(f"Required files for Figure 4 not found (image {image_num})")
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(g_clahe, cmap='gray')
    axes[0].set_title('(a) G channel with CLAHE', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(y_clahe, cmap='gray')
    axes[1].set_title('(b) Y channel with CLAHE', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(l_clahe, cmap='gray')
    axes[2].set_title('(c) L channel with CLAHE', fontsize=12)
    axes[2].axis('off')

    plt.suptitle('Fig. 4. Outputs of the histogram equalization operation on (a) G, (b) Y, and (c) L channels',
                 fontsize=12, y=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure 4 saved: {save_path}")

def create_figure_5(image_num, save_path):
    """
    Figure 5: Outputs of Gabor filters (RAW FLOAT OUTPUTS - greyscale)
    3 rows (G, Y, L channels) x 3 columns (λ=9, λ=10, λ=11)

    NOTE: This figure shows the raw (float) Gabor outputs, not binary!
    """
    channels = ['G', 'Y', 'L']
    wavelengths = [9, 10, 11]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for row_idx, channel in enumerate(channels):
        for col_idx, wl in enumerate(wavelengths):
            # Raw float Gabor outputs (not binary)
            filename = f"{image_num:02d}_Gabor_{channel}_λ{wl}.png"
            img = load_output_image("figures_5_6", filename)

            if img is not None:
                axes[row_idx, col_idx].imshow(img, cmap='gray')
            else:
                axes[row_idx, col_idx].text(0.5, 0.5, 'Not Found', ha='center', va='center')

            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f'λ = {wl}', fontsize=12)
            axes[row_idx, col_idx].axis('off')

        # Row labels
        axes[row_idx, 0].text(-0.15, 0.5, f'{channel} channel',
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=12, va='center', rotation=90)

    # Column labels
    axes[2, 0].set_xlabel('(a)', fontsize=12)
    axes[2, 1].set_xlabel('(b)', fontsize=12)
    axes[2, 2].set_xlabel('(c)', fontsize=12)

    plt.suptitle('Fig. 5. Outputs of the Gabor filters on G, Y, and L channels\nwith wavelength of (a) 9, (b) 10, and (c) 11',
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure 5 saved: {save_path}")

def create_figure_6(image_num, save_path):
    """
    Figure 6: Results of automatic thresholding on Gabor outputs (BINARY)
    Same layout as Figure 5 but binary thresholded
    """
    channels = ['G', 'Y', 'L']
    wavelengths = [9, 10, 11]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for row_idx, channel in enumerate(channels):
        for col_idx, wl in enumerate(wavelengths):
            # Binary Gabor outputs (saved with the _binary suffix)
            filename = f"{image_num:02d}_Gabor_{channel}_λ{wl}_binary.png"
            img = load_output_image("figures_5_6", filename)

            if img is not None:
                axes[row_idx, col_idx].imshow(img, cmap='gray')
            else:
                axes[row_idx, col_idx].text(0.5, 0.5, 'Not Found', ha='center', va='center')

            if row_idx == 0:
                axes[row_idx, col_idx].set_title(f'λ = {wl}', fontsize=12)
            axes[row_idx, col_idx].axis('off')

        axes[row_idx, 0].text(-0.15, 0.5, f'{channel} channel',
                             transform=axes[row_idx, 0].transAxes,
                             fontsize=12, va='center', rotation=90)

    axes[2, 0].set_xlabel('(a)', fontsize=12)
    axes[2, 1].set_xlabel('(b)', fontsize=12)
    axes[2, 2].set_xlabel('(c)', fontsize=12)

    plt.suptitle('Fig. 6. Results of the automatic thresholding method on Gabor filter outputs\n(a) λ=9, (b) λ=10, and (c) λ=11',
                 fontsize=14, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure 6 saved: {save_path}")

def create_figure_7(image_num, save_path):
    """
    Figure 7: Top Hat feature
    (a) Input image, (b) extracted TH feature
    """
    original = load_original_image(image_num)
    mask = load_mask(image_num)
    original_cropped = crop_to_fov(original, mask)

    th_feature = load_output_image("figures_7_8_9", f"{image_num:02d}_TopHat.png")

    if th_feature is None:
        print(f"Required files for Figure 7 not found (image {image_num})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_cropped)
    axes[0].set_title('(a) Input image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(th_feature, cmap='gray')
    axes[1].set_title('(b) Extracted TH feature', fontsize=12)
    axes[1].axis('off')

    plt.suptitle('Fig. 7. A sample of TH feature. (a) Input image, and (b) extracted TH feature',
                 fontsize=12, y=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure 7 saved: {save_path}")

def create_figure_8(image_num, save_path):
    """
    Figure 8: Shade Correction feature
    (a) Input image, (b) extracted SC feature
    """
    original = load_original_image(image_num)
    mask = load_mask(image_num)
    original_cropped = crop_to_fov(original, mask)

    sc_feature = load_output_image("figures_7_8_9", f"{image_num:02d}_ShadeCorrect.png")

    if sc_feature is None:
        print(f"Required files for Figure 8 not found (image {image_num})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_cropped)
    axes[0].set_title('(a) Input image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(sc_feature, cmap='gray')
    axes[1].set_title('(b) Extracted SC feature', fontsize=12)
    axes[1].axis('off')

    plt.suptitle('Fig. 8. A sample of SC feature. (a) Input image, and (b) extracted SC feature',
                 fontsize=12, y=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure 8 saved: {save_path}")

def create_figure_9(image_num, save_path):
    """
    Figure 9: Bit Plane Slicing - PAPER-CONSISTENT

    Paper Figure 9:
    - (a) Input image (colour)
    - (b-i) Bit planes 1-8 (from the NORMAL G channel)
    - (j) sum of the 7th + 8th bit planes

    NOTE: the paper uses the NORMAL G channel (not inverted).
    In bit 7 the vessels appear WHITE because the paper uses a different visualization
    """
    original = load_original_image(image_num)
    mask = load_mask(image_num)
    original_cropped = crop_to_fov(original, mask)

    # Load the original BGR image
    path = os.path.join(DATA_DIR, "training", "images", f"{image_num}_training.tif")
    bgr_image = cv2.imread(path)

    # Take the G channel (NORMAL - not inverted)
    g_channel_original = bgr_image[:, :, 1]

    # Crop to the FOV
    g_channel = crop_to_fov(g_channel_original, mask)
    mask_cropped = crop_to_fov(mask, mask)

    # PAPER-CONSISTENT: extract the bit planes from the normal G channel
    # (Vessels dark = low value, background bright = high value)
    bit_planes = []
    for i in range(8):
        bit_plane = ((g_channel >> i) & 1) * 255
        bit_planes.append(bit_plane)

    # Sum of the 7th + 8th bit planes (PAPER FORMULA)
    # Bit 7 = index 6, Bit 8 = index 7 (0-indexed)
    bit_7 = (g_channel >> 6) & 1  # 7th bit plane
    bit_8 = (g_channel >> 7) & 1  # 8th bit plane (MSB)

    # Paper-consistent (j) image:
    # - outside FOV: grey (128)
    # - bit_7=0, bit_8=0: dark (0) - vessels
    # - bit_7=1, bit_8=0 or bit_7=0, bit_8=1: mid grey (127)
    # - bit_7=1, bit_8=1: bright (254) - background
    bps_sum = ((bit_7.astype(np.uint16) + bit_8.astype(np.uint16)) * 127).astype(np.uint8)

    # Make the region outside the FOV grey (as in the paper)
    bps_sum_display = np.full_like(bps_sum, 128, dtype=np.uint8)
    bps_sum_display[mask_cropped > 0] = bps_sum[mask_cropped > 0]

    # Create the figure - matching the paper layout
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    # (a) Input - colour image
    axes[0, 0].imshow(original_cropped)
    axes[0, 0].set_title('(a) Input', fontsize=10)
    axes[0, 0].axis('off')

    # (b-e) Bit 1-4
    for i in range(4):
        axes[0, i+1].imshow(bit_planes[i], cmap='gray')
        axes[0, i+1].set_title(f'({chr(98+i)}) Bit {i+1}', fontsize=10)
        axes[0, i+1].axis('off')

    # (f-i) Bit 5-8
    for i in range(4):
        axes[1, i].imshow(bit_planes[i+4], cmap='gray')
        axes[1, i].set_title(f'({chr(102+i)}) Bit {i+5}', fontsize=10)
        axes[1, i].axis('off')

    # (j) 7th + 8th bits sum
    axes[1, 4].imshow(bps_sum_display, cmap='gray')
    axes[1, 4].set_title('(j) 7th + 8th bits', fontsize=10)
    axes[1, 4].axis('off')

    plt.suptitle('Fig. 9. BPS feature: (a) Input image, (b-i) bit planes 1-8, (j) sum of 7th and 8th bit planes',
                 fontsize=12, y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure 9 saved: {save_path}")

def create_figure_10(image_num, save_path):
    """
    Figure 10: FCM Clustering output
    (a) Input image, (b) initial extraction of vessels by clustering
    """
    original = load_original_image(image_num)
    mask = load_mask(image_num)
    original_cropped = crop_to_fov(original, mask)

    fcm_output = load_output_image("figure_10", f"{image_num:02d}_FCM.png")

    if fcm_output is None:
        print(f"Required files for Figure 10 not found (image {image_num})")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_cropped)
    axes[0].set_title('(a) Input image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(fcm_output, cmap='gray')
    axes[1].set_title('(b) Initial extraction by clustering', fontsize=12)
    axes[1].axis('off')

    plt.suptitle('Fig. 10. An example output of the clustering step. (a) Input image, and\n(b) initial extraction of the vessels by clustering',
                 fontsize=12, y=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure 10 saved: {save_path}")

def create_figure_11(image_num, save_path):
    """
    Figure 11: Outputs of clustering and classification
    (a) Input image, (b) clustering output (FCM), (c) All extracted vessels (FCM + DT combined)
    """
    original = load_original_image(image_num)
    mask = load_mask(image_num)
    original_cropped = crop_to_fov(original, mask)

    fcm_output = load_output_image("figure_10", f"{image_num:02d}_FCM.png")
    dt_output = load_output_image("figure_11", f"{image_num:02d}_DT.png")

    if fcm_output is None:
        print(f"Required files for Figure 11 not found (image {image_num})")
        return

    # (c) FCM + DT combination (all extracted vessels)
    if dt_output is not None:
        combined = cv2.bitwise_or(fcm_output, dt_output)
    else:
        combined = fcm_output

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_cropped)
    axes[0].set_title('(a) Input image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(fcm_output, cmap='gray')
    axes[1].set_title('(b) Clustering output', fontsize=12)
    axes[1].axis('off')

    # FCM + DT combination (all extracted vessels)
    axes[2].imshow(combined, cmap='gray')
    axes[2].set_title('(c) All extracted vessels', fontsize=12)
    axes[2].axis('off')

    plt.suptitle('Fig. 11. Outputs of the clustering and classification steps.\n(a) Input image, (b) clustering output, (c) all extracted vessels by clustering and classification',
                 fontsize=12, y=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure 11 saved: {save_path}")

def create_figure_12(image_num, save_path):
    """
    Figure 12: Post-processing step
    (a) Input image, (b) output of vessel extraction step, (c) final extracted vessels after post-processing
    """
    original = load_original_image(image_num)
    mask = load_mask(image_num)
    original_cropped = crop_to_fov(original, mask)

    fcm_output = load_output_image("figure_10", f"{image_num:02d}_FCM.png")
    dt_output = load_output_image("figure_11", f"{image_num:02d}_DT.png")
    final_output = load_output_image("figure_12", f"{image_num:02d}_Final.png")

    if final_output is None:
        print(f"Required files for Figure 12 not found (image {image_num})")
        return

    # Before post-processing (FCM + DT)
    if fcm_output is not None and dt_output is not None:
        before_postprocess = cv2.bitwise_or(fcm_output, dt_output)
    elif fcm_output is not None:
        before_postprocess = fcm_output
    else:
        before_postprocess = final_output

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original_cropped)
    axes[0].set_title('(a) Input image', fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(before_postprocess, cmap='gray')
    axes[1].set_title('(b) Vessel extraction output', fontsize=12)
    axes[1].axis('off')

    axes[2].imshow(final_output, cmap='gray')
    axes[2].set_title('(c) Final extracted vessels', fontsize=12)
    axes[2].axis('off')

    plt.suptitle('Fig. 12. Output of the post-processing step. (a) Input image, (b) output of the\nvessel extraction step, and (c) final extracted vessels after post-processing',
                 fontsize=12, y=0.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Figure 12 saved: {save_path}")


def create_all_figures():
    """Generate all figures"""
    # Create the figures directory
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("="*60)
    print("GENERATING PAPER FIGURES")
    print("="*60)

    # Sample image number (36)
    image_num = SAMPLE_IMAGE_NUM

    print(f"\nGenerating figures using image #{image_num}...\n")

    # Generate all figures
    create_figure_3(image_num, os.path.join(FIGURES_DIR, "figure_3.png"))
    create_figure_4(image_num, os.path.join(FIGURES_DIR, "figure_4.png"))
    create_figure_5(image_num, os.path.join(FIGURES_DIR, "figure_5.png"))
    create_figure_6(image_num, os.path.join(FIGURES_DIR, "figure_6.png"))
    create_figure_7(image_num, os.path.join(FIGURES_DIR, "figure_7.png"))
    create_figure_8(image_num, os.path.join(FIGURES_DIR, "figure_8.png"))
    create_figure_9(image_num, os.path.join(FIGURES_DIR, "figure_9.png"))
    create_figure_10(image_num, os.path.join(FIGURES_DIR, "figure_10.png"))
    create_figure_11(image_num, os.path.join(FIGURES_DIR, "figure_11.png"))
    create_figure_12(image_num, os.path.join(FIGURES_DIR, "figure_12.png"))

    print("\n" + "="*60)
    print("ALL FIGURES GENERATED!")
    print(f"Outputs: in the {FIGURES_DIR}/ directory")
    print("="*60)


if __name__ == "__main__":
    create_all_figures()

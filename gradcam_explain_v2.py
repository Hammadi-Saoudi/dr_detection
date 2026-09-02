"""
==========================================================
  ETAPE 4 v2 : GRAD-CAM++ MULTI-COUCHES + SCORE-CAM
  Script : gradcam_explain_v2.py

  Ameliorations vs v1 :
  1. MULTI-LAYER : 3 couches EfficientNet combinees
  2. Seuillage OTSU pour isoler les lesions
  3. Contours des zones d'activation traces en blanc
  4. Comparaison Grad-CAM++ vs EigenCAM
  5. Figure 2x3 (6 panneaux separes, sans conflit d'axes)
==========================================================
"""

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

from pytorch_grad_cam import GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================================
# CONFIGURATION
# ==========================================================
DESKTOP_PATH = r'C:\Users\Saoudi\OneDrive\Desktop\aptos2019-blindness-detection'
IMAGE_DIR    = os.path.join(DESKTOP_PATH, 'train_images_enhanced')
CSV_PATH     = os.path.join(DESKTOP_PATH, 'train.csv')
MODEL_PATH   = 'best_dr_model.pth'
OUTPUT_DIR   = 'gradcam_results_v2'
NUM_IMAGES   = 5
RANDOM_SEED  = 42

GRADE_LABELS = {
    0: "Grade 0 : Sain",
    1: "Grade 1 : RD Legere",
    2: "Grade 2 : RD Moderee",
    3: "Grade 3 : RD Severe",
    4: "Grade 4 : RD Proliferative",
}
GRADE_COLORS = {0:'#2ecc71', 1:'#f39c12', 2:'#e67e22', 3:'#e74c3c', 4:'#8e44ad'}

# ==========================================================
# UTILITAIRES
# ==========================================================

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    check = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if check == 0:
        return img
    return np.stack([img[:, :, c][np.ix_(mask.any(1), mask.any(0))] for c in range(3)], axis=-1)


def load_image(img_path, size=256):
    img_cv = cv2.imread(img_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_cv = crop_image_from_gray(img_cv)
    img_cv = cv2.resize(img_cv, (size, size))
    rgb    = img_cv.astype(np.float32) / 255.0
    tf     = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = tf(img_cv).unsqueeze(0)
    return rgb, tensor


def predict(model, tensor, device):
    model.eval()
    with torch.no_grad():
        out   = model(tensor.to(device))
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return int(np.argmax(probs)), probs


def apply_otsu_threshold(cam_map):
    cam_uint8 = (cam_map * 255).astype(np.uint8)
    _, binary = cv2.threshold(cam_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cam_clean = cam_map.copy()
    cam_clean[binary == 0] = 0.0
    return cam_clean, binary


def draw_lesion_contours(rgb_display, binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = (rgb_display * 255).astype(np.uint8).copy()
    big_contours = [c for c in contours if cv2.contourArea(c) > 50]
    cv2.drawContours(overlay, big_contours, -1, (255, 255, 255), 2)
    return overlay.astype(np.float32) / 255.0, len(big_contours)


def multi_layer_cam(model, tensor, target_layers, target_class, device, method='gradcam++'):
    tensor = tensor.to(device)
    maps   = []
    for layer in target_layers:
        if method == 'gradcam++':
            cam_obj = GradCAMPlusPlus(model=model, target_layers=[layer])
        else:
            cam_obj = EigenCAM(model=model, target_layers=[layer])
        targets    = [ClassifierOutputTarget(target_class)]
        grayscale  = cam_obj(input_tensor=tensor, targets=targets)[0]
        if grayscale.max() > 0:
            grayscale = grayscale / grayscale.max()
        maps.append(grayscale)

    weights = np.array([0.20, 0.35, 0.45])[:len(maps)]
    weights /= weights.sum()
    fused = sum(w * m for w, m in zip(weights, maps))
    if fused.max() > 0:
        fused = fused / fused.max()
    return fused


# ==========================================================
# VISUALISATION (6 panneaux, 2x3, axes completement separes)
# ==========================================================
def create_figure(rgb, cam_gradcam, cam_eigen, binary_mask,
                  contour_img, n_lesions, pred_class, true_class,
                  probs, img_name, output_path):

    correct      = pred_class == true_class
    status_str   = "CORRECT" if correct else "INCORRECT"
    status_color = '#2ecc71' if correct else '#e74c3c'
    pred_color   = GRADE_COLORS[pred_class]
    true_color   = GRADE_COLORS[true_class]

    overlay_gradcam = show_cam_on_image(rgb, cam_gradcam, use_rgb=True)
    overlay_clean   = show_cam_on_image(rgb, cam_gradcam * (binary_mask / 255.0), use_rgb=True)
    overlay_eigen   = show_cam_on_image(rgb, cam_eigen,   use_rgb=True)

    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.patch.set_facecolor('#0f0f1a')
    fig.suptitle(
        f"Grad-CAM++ Multi-Couches  |  {img_name}  |  {status_str}  |  {n_lesions} zone(s)",
        fontsize=13, color='white', fontweight='bold', y=1.01
    )

    titles = [
        "1. Image Originale (Retine)",
        "2. Grad-CAM++ Multi-couches (brut)",
        "3. Grad-CAM++ seuillage OTSU",
        "4. EigenCAM (confirmation sans gradient)",
        "5. Contours des Lesions",
        "6. Probabilites par Grade",
    ]

    # Panels image (ligne 1 : ax[0,0] ax[0,1] ax[0,2] | ligne 2 : ax[1,0] ax[1,1])
    img_panels = [axes[0,0], axes[0,1], axes[0,2], axes[1,0], axes[1,1]]
    images     = [rgb, overlay_gradcam, overlay_clean, overlay_eigen, contour_img]
    xlabels    = [
        (f"Vrai : {GRADE_LABELS[true_class]}", true_color),
        ("Rouge=max attention | Bleu=ignore",    '#aaaaaa'),
        ("Zones isolees par OTSU",               '#aaaaaa'),
        ("Plus stable, sans gradient",           '#aaaaaa'),
        (f"Predit : {GRADE_LABELS[pred_class]}", pred_color),
    ]

    for ax, img, (xlabel, xcolor), title in zip(img_panels, images, xlabels, titles):
        ax.set_facecolor('#16213e')
        ax.set_title(title, color='white', fontsize=9, pad=6)
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(xlabel, color=xcolor, fontsize=8, fontweight='bold')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333355')

    # ---- Panel 6 : Bar chart avec positions NUMERIQUES ----
    ax_bar = axes[1, 2]
    ax_bar.set_facecolor('#16213e')
    ax_bar.set_title(titles[5], color='white', fontsize=9, pad=6)
    for sp in ax_bar.spines.values():
        sp.set_edgecolor('#333355')

    y_pos      = np.arange(5)                              # [0,1,2,3,4] numerique
    labels_bar = ['Gr.0 Sain','Gr.1 Leger','Gr.2 Mod.','Gr.3 Sev.','Gr.4 Prolif.']
    bar_colors = [GRADE_COLORS[i] for i in range(5)]

    bars = ax_bar.barh(y_pos, probs * 100,
                       color=bar_colors, edgecolor='white', linewidth=0.5)

    for bar, p in zip(bars, probs):
        ax_bar.text(
            min(p * 100 + 1, 97),
            bar.get_y() + bar.get_height() / 2,
            f'{p*100:.1f}%',
            va='center', ha='left', color='white', fontsize=9
        )

    ax_bar.set_xlim(0, 110)
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(labels_bar, color='white', fontsize=9)
    ax_bar.set_xlabel('Probabilite (%)', color='white', fontsize=9)
    ax_bar.tick_params(axis='x', colors='white')

    bars[pred_class].set_linewidth(2.5)
    bars[pred_class].set_edgecolor('white')

    patch = mpatches.Patch(color=status_color, label=f'Diagnostic {status_str}')
    ax_bar.legend(handles=[patch], loc='lower right',
                  facecolor='#0f0f1a', labelcolor='white', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


# ==========================================================
# SCRIPT PRINCIPAL
# ==========================================================
def main():
    print("=" * 65)
    print("  ETAPE 4 v2 : GRAD-CAM++ MULTI-COUCHES")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil : {device}")

    print("\nChargement du modele EfficientNet-B3...")
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
    model = model.to(device).eval()

    # 3 couches a des profondeurs differentes
    target_layers = [
        model.features[2],   # Details fins (micro-lesions)
        model.features[4],   # Structures vasculaires
        model.features[7],   # Semantique du grade
    ]
    print("Couches : features[2] (fin) + features[4] (mid) + features[7] (semantique)")

    df = pd.read_csv(CSV_PATH)
    _, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['diagnosis'])
    _, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['diagnosis'])
    test_df = test_df.reset_index(drop=True)

    selected = []
    for grade in sorted(test_df['diagnosis'].unique())[:NUM_IMAGES]:
        row = test_df[test_df['diagnosis'] == grade].sample(n=1, random_state=RANDOM_SEED + grade).iloc[0]
        selected.append(row)

    print(f"\n{len(selected)} images (1 par grade) :\n")
    for row in selected:
        print(f"  Grade {int(row['diagnosis'])} : {row['id_code']}")

    results = []
    print("\nGeneration des cartes...\n")

    for i, row in enumerate(selected):
        img_id     = row['id_code']
        true_class = int(row['diagnosis'])
        img_path   = os.path.join(IMAGE_DIR, f"{img_id}.png")

        print(f"[{i+1}/{len(selected)}] {img_id}  (Grade {true_class})")
        if not os.path.exists(img_path):
            print("  ATTENTION : image introuvable.")
            continue

        rgb, tensor = load_image(img_path)
        pred_class, probs = predict(model, tensor, device)
        status = "CORRECT" if pred_class == true_class else "INCORRECT"
        print(f"  Predit : {pred_class} ({GRADE_LABELS[pred_class]}) | {status}")

        print("  Calcul Grad-CAM++ multi-couches...")
        cam_gc = multi_layer_cam(model, tensor, target_layers, pred_class, device, 'gradcam++')

        print("  Calcul EigenCAM multi-couches...")
        cam_eg = multi_layer_cam(model, tensor, target_layers, pred_class, device, 'eigen')

        cam_clean, binary = apply_otsu_threshold(cam_gc)
        contour_img, n_lesions = draw_lesion_contours(rgb, binary)
        print(f"  Zones de lesion detectees : {n_lesions}")

        fname    = f"gradcam_v2_{i+1:02d}_{img_id}_true{true_class}_pred{pred_class}.png"
        out_path = os.path.join(OUTPUT_DIR, fname)

        create_figure(rgb, cam_gc, cam_eg, binary, contour_img,
                      n_lesions, pred_class, true_class,
                      probs, img_id, out_path)
        print(f"  Sauvegarde : {fname}\n")

        results.append({
            'image': img_id, 'grade_reel': true_class,
            'grade_predit': pred_class, 'statut': status,
            'zones': n_lesions, 'confiance': f'{probs[pred_class]:.2%}'
        })

    summary_path = os.path.join(OUTPUT_DIR, 'resume_gradcam_v2.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("ETAPE 4 v2 - GRAD-CAM++ MULTI-COUCHES\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            line = (f"Image : {r['image']}\n"
                    f"  Grade reel    : {r['grade_reel']} - {GRADE_LABELS[r['grade_reel']]}\n"
                    f"  Grade predit  : {r['grade_predit']} - {GRADE_LABELS[r['grade_predit']]}\n"
                    f"  Confiance     : {r['confiance']}\n"
                    f"  Zones detectees : {r['zones']}\n"
                    f"  Statut        : {r['statut']}\n"
                    + "-" * 40 + "\n")
            print(line)
            f.write(line)

    print(f"[OK] Resultats dans '{OUTPUT_DIR}/'")
    print(f"[OK] Resume : '{summary_path}'")


if __name__ == "__main__":
    main()

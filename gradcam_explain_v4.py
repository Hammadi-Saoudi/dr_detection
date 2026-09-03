"""
==========================================================
  ETAPE 4 v4 : GRAD-CAM++ HAUTE PRECISION CLINIQUE
  Script : gradcam_explain_v4.py

  Techniques de precision :
  1. HiResCAM (plus fidele et precis que Grad-CAM++)
  2. ScoreCAM (sans bruit de gradient) en confirmation
  3. Fusion max(HiResCAM, ScoreCAM) pixel-a-pixel
  4. Filtre bilateral (preserve les bords des lesions)
  5. Seuillage adaptatif (adapte a chaque image)
  6. Visualisation clinique rouge pur + legende
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

from pytorch_grad_cam import HiResCAM, ScoreCAM, GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================================
# CONFIGURATION
# ==========================================================
DESKTOP_PATH = r'C:\Users\Saoudi\OneDrive\Desktop\aptos2019-blindness-detection'
IMAGE_DIR    = os.path.join(DESKTOP_PATH, 'train_images_enhanced')
CSV_PATH     = os.path.join(DESKTOP_PATH, 'train.csv')
MODEL_PATH   = 'best_dr_model.pth'
OUTPUT_DIR   = 'gradcam_results_v4'
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
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask  = gray > tol
    check = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if check == 0:
        return img
    return np.stack([img[:, :, c][np.ix_(mask.any(1), mask.any(0))] for c in range(3)], axis=-1)


def load_image(img_path, size=512):
    img_cv = cv2.imread(img_path)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    img_cv = crop_image_from_gray(img_cv)
    img_cv = cv2.resize(img_cv, (size, size))
    rgb    = img_cv.astype(np.float32) / 255.0
    tf = transforms.Compose([
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


def compute_hirescam(model, tensor, target_layers, target_class, device):
    """
    HiResCAM : plus fidele que Grad-CAM++ pour les petites lesions.
    Conserve la resolution spatiale complete (pas d'interpolation grossiere).
    Calcule sur 3 couches, fusionne par MAX pixel-a-pixel.
    """
    tensor = tensor.to(device)
    maps   = []
    for layer in target_layers:
        cam_obj   = HiResCAM(model=model, target_layers=[layer])
        targets   = [ClassifierOutputTarget(target_class)]
        grayscale = cam_obj(input_tensor=tensor, targets=targets)[0]
        if grayscale.max() > 0:
            grayscale = grayscale / grayscale.max()
        maps.append(grayscale.astype(np.float32))
    fused = np.maximum.reduce(maps)
    return fused


def compute_scorecam(model, tensor, target_layer, target_class, device):
    """
    ScoreCAM : sans bruit de gradient, plus stable.
    Seulement sur la derniere couche (couche semantique).
    """
    tensor  = tensor.to(device)
    cam_obj = ScoreCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    grayscale = cam_obj(input_tensor=tensor, targets=targets)[0]
    if grayscale.max() > 0:
        grayscale = grayscale / grayscale.max()
    return grayscale.astype(np.float32)


def fuse_maps(cam_hires, cam_score, alpha=0.65):
    """
    Fusion finale : combinaison lineaire de HiResCAM et ScoreCAM.
    HiResCAM (65%) pour la precision spatiale.
    ScoreCAM (35%) pour la robustesse semantique.
    """
    fused = alpha * cam_hires + (1 - alpha) * cam_score
    # Filtre bilateral : preserve les bords (contours des lesions nets)
    # sans le flou gaussien qui efface les micro-lesions
    fused_uint8 = (fused * 255).astype(np.uint8)
    fused_smooth = cv2.bilateralFilter(fused_uint8, d=9, sigmaColor=75, sigmaSpace=75)
    fused_smooth = fused_smooth.astype(np.float32) / 255.0
    if fused_smooth.max() > 0:
        fused_smooth = fused_smooth / fused_smooth.max()
    return fused_smooth


def adaptive_threshold(cam_map, grade):
    """
    Seuillage adaptatif selon le grade :
    - Grades 0-1 (peu de lesions) : seuil plus bas pour detecter les rares lésions
    - Grades 2-4 (beaucoup de lesions) : seuil plus haut pour isoler les principales
    """
    thresholds = {0: 72, 1: 68, 2: 62, 3: 58, 4: 55}
    percentile = thresholds.get(grade, 62)
    threshold  = np.percentile(cam_map, percentile)
    binary = (cam_map >= threshold).astype(np.uint8) * 255

    # Fermeture morphologique pour combler les petits trous dans les lesions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Ouverture pour supprimer les micro-bruits isoles
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    cam_clean = cam_map.copy()
    cam_clean[binary == 0] = 0.0
    return cam_clean, binary


def make_clinical_overlay(rgb, cam_map, binary_mask):
    """
    Superposition clinique rouge sur fond assombri.
    Seules les zones de lesion apparaissent en rouge.
    Intensite du rouge proportionnelle a la force d'attention.
    """
    h, w = rgb.shape[:2]

    # Fond : zones saines assombries a 40%
    alpha_darken = np.where(binary_mask > 0, 1.0, 0.40)
    result = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        result[:, :, c] = rgb[:, :, c] * alpha_darken

    # Rouge proportionnel a l'intensite du CAM (zones de lesion uniquement)
    intensity = cam_map * (binary_mask / 255.0)
    alpha_red = intensity * 0.80    # Transparence du rouge

    result[:, :, 0] = np.clip(result[:, :, 0] * (1 - alpha_red) + intensity, 0, 1)  # R
    result[:, :, 1] = np.clip(result[:, :, 1] * (1 - alpha_red), 0, 1)              # G
    result[:, :, 2] = np.clip(result[:, :, 2] * (1 - alpha_red), 0, 1)              # B

    # Contours blancs fins autour des zones de lesion
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_uint8 = (result * 255).astype(np.uint8)
    big_contours = [c for c in contours if cv2.contourArea(c) > 80]
    cv2.drawContours(result_uint8, big_contours, -1, (255, 255, 255), 1)

    return np.clip(result_uint8.astype(np.float32) / 255.0, 0, 1), len(big_contours)


# ==========================================================
# CREATION DE LA FIGURE (3 panneaux + legende)
# ==========================================================
def create_figure(rgb, overlay_img, cam_map, n_contours,
                  pred_class, true_class, probs, img_name, output_path):

    correct      = pred_class == true_class
    status_str   = "CORRECT" if correct else "INCORRECT"
    status_color = '#2ecc71' if correct else '#e74c3c'
    pred_color   = GRADE_COLORS[pred_class]
    true_color   = GRADE_COLORS[true_class]

    pct_lesion = 100.0 * (cam_map > 0.05).sum() / cam_map.size

    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        f"Analyse Grad-CAM++ (HiResCAM + ScoreCAM)  |  {img_name}  |  {status_str}",
        fontsize=14, color='white', fontweight='bold', y=1.01
    )

    gs  = fig.add_gridspec(1, 3, wspace=0.06, width_ratios=[1, 1, 0.65])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ---- Panneau 1 : Image originale ----
    ax1.set_facecolor('#0d1117')
    ax1.set_title("Image Originale de la Retine", color='white', fontsize=11, pad=10)
    ax1.imshow(rgb)
    ax1.set_xticks([]); ax1.set_yticks([])
    ax1.set_xlabel(
        f"Vrai : {GRADE_LABELS[true_class]}",
        color=true_color, fontsize=10, fontweight='bold'
    )
    for sp in ax1.spines.values(): sp.set_edgecolor('#333355')

    # ---- Panneau 2 : Overlay clinique + colorbar ----
    ax2.set_facecolor('#0d1117')
    ax2.set_title(
        "Zones d'Attention de l'IA (HiResCAM + ScoreCAM)",
        color='white', fontsize=11, pad=10
    )
    ax2.imshow(overlay_img)
    ax2.set_xticks([]); ax2.set_yticks([])
    ax2.set_xlabel(
        f"Diagnostic IA : {GRADE_LABELS[pred_class]}  |  {n_contours} zone(s) detectee(s)",
        color=pred_color, fontsize=10, fontweight='bold'
    )
    for sp in ax2.spines.values(): sp.set_edgecolor('#333355')

    # Colorbar verticale : gradient rouge avec etiquettes
    ax_cbar = ax2.inset_axes([1.015, 0.0, 0.035, 1.0])
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    ax_cbar.imshow(gradient, aspect='auto', cmap='Reds', origin='lower')
    ax_cbar.set_xticks([])
    ax_cbar.set_yticks([0, 64, 128, 192, 255])
    ax_cbar.set_yticklabels(
        ['Ignore', 'Faible', 'Modere', 'Eleve', 'Maximum'],
        color='white', fontsize=7.5
    )
    ax_cbar.set_facecolor('#0d1117')
    for sp in ax_cbar.spines.values(): sp.set_edgecolor('#444466')

    # ---- Panneau 3 : Bar chart + legende ----
    ax3.set_facecolor('#161b22')
    ax3.set_title("Probabilites par Grade", color='white', fontsize=11, pad=10)
    for sp in ax3.spines.values(): sp.set_edgecolor('#333355')

    y_pos      = np.arange(5)
    labels_bar = ['Gr.0 Sain', 'Gr.1 Leger', 'Gr.2 Mod.', 'Gr.3 Sev.', 'Gr.4 Prolif.']
    bar_colors = [GRADE_COLORS[i] for i in range(5)]

    bars = ax3.barh(y_pos, probs * 100, color=bar_colors,
                    edgecolor='#888', linewidth=0.5, height=0.55)
    for bar, p in zip(bars, probs):
        ax3.text(min(p*100+1.5, 97), bar.get_y()+bar.get_height()/2,
                 f'{p*100:.1f}%', va='center', ha='left',
                 color='white', fontsize=10, fontweight='bold')

    ax3.set_xlim(0, 115)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels_bar, color='white', fontsize=10)
    ax3.set_xlabel('Probabilite (%)', color='white', fontsize=10)
    ax3.tick_params(axis='x', colors='white')
    bars[pred_class].set_linewidth(3.0)
    bars[pred_class].set_edgecolor('white')

    # Bloc d'explication des couleurs
    legende = (
        "LEGENDE DES COULEURS\n"
        "─────────────────────\n"
        " Rouge intense\n"
        "  → Lesion principale\n"
        "    (max attention IA)\n\n"
        " Rouge pale\n"
        "  → Lesion secondaire\n"
        "    (attention partielle)\n\n"
        " Zone sombre\n"
        "  → Tissu sain ignore\n"
        "    par le modele\n\n"
        " Contour blanc\n"
        "  → Bordure de la zone\n"
        "    de decision\n\n"
        f"Couverture : {pct_lesion:.1f}%\n"
        f"de la retine analysee"
    )
    ax3.text(0.02, -0.30, legende,
             transform=ax3.transAxes,
             color='#cccccc', fontsize=8.5,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='#0d1117',
                       edgecolor='#444466', alpha=0.95))

    patch = mpatches.Patch(color=status_color, label=f'Diagnostic {status_str}')
    ax3.legend(handles=[patch], loc='lower right',
               facecolor='#0d1117', labelcolor='white', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


# ==========================================================
# SCRIPT PRINCIPAL
# ==========================================================
def main():
    print("=" * 65)
    print("  ETAPE 4 v4 : HiResCAM + ScoreCAM HAUTE PRECISION")
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

    # Couches pour HiResCAM (3 niveaux de profondeur)
    hires_layers = [
        model.features[2],   # Details fins
        model.features[4],   # Structures intermediaires
        model.features[6],   # Semantique fine
    ]
    # Couche pour ScoreCAM (derniere couche uniquement)
    score_layer = model.features[7]

    print("HiResCAM : features[2]+[4]+[6]")
    print("ScoreCAM : features[7]")

    df = pd.read_csv(CSV_PATH)
    _, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['diagnosis'])
    _, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['diagnosis'])
    test_df = test_df.reset_index(drop=True)

    selected = []
    for grade in sorted(test_df['diagnosis'].unique())[:NUM_IMAGES]:
        row = test_df[test_df['diagnosis'] == grade].sample(
            n=1, random_state=RANDOM_SEED + grade).iloc[0]
        selected.append(row)

    print(f"\n{len(selected)} images (1 par grade) :")
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
        print(f"  Predit : Grade {pred_class} | {status}")

        # HiResCAM multi-couches
        print("  [1/3] Calcul HiResCAM...")
        cam_hires = compute_hirescam(model, tensor, hires_layers, pred_class, device)

        # ScoreCAM (plus lent mais plus stable)
        print("  [2/3] Calcul ScoreCAM...")
        cam_score = compute_scorecam(model, tensor, score_layer, pred_class, device)

        # Fusion + filtre bilateral
        print("  [3/3] Fusion et post-traitement...")
        cam_final = fuse_maps(cam_hires, cam_score, alpha=0.65)

        # Seuillage adaptatif selon le grade
        cam_clean, binary = adaptive_threshold(cam_final, true_class)

        # Overlay clinique
        overlay_img, n_contours = make_clinical_overlay(rgb, cam_clean, binary)

        pct = 100.0 * (cam_clean > 0.05).sum() / cam_clean.size
        print(f"  Zones detectees : {n_contours} | Couverture : {pct:.1f}%")

        fname    = f"gradcam_v4_{i+1:02d}_{img_id}_true{true_class}_pred{pred_class}.png"
        out_path = os.path.join(OUTPUT_DIR, fname)

        create_figure(rgb, overlay_img, cam_clean, n_contours,
                      pred_class, true_class, probs, img_id, out_path)
        print(f"  Sauvegarde : {fname}\n")

        results.append({
            'image': img_id,
            'grade_reel': true_class,
            'grade_predit': pred_class,
            'statut': status,
            'zones': n_contours,
            'couverture': f'{pct:.1f}%',
            'confiance': f'{probs[pred_class]:.2%}'
        })

    # Resume textuel
    summary_path = os.path.join(OUTPUT_DIR, 'resume_gradcam_v4.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("ETAPE 4 v4 - HiResCAM + ScoreCAM HAUTE PRECISION\n")
        f.write("=" * 60 + "\n\n")
        f.write("METHODES UTILISEES :\n")
        f.write("  HiResCAM (65%) : preserves la resolution spatiale fine\n")
        f.write("  ScoreCAM (35%) : stable, sans bruit de gradient\n")
        f.write("  Filtre bilateral : preserve les bords des lesions\n")
        f.write("  Seuillage adaptatif par grade : s'adapte a chaque image\n\n")
        f.write("LEGENDE DES COULEURS :\n")
        f.write("  Rouge intense   = Lesion principale (max attention)\n")
        f.write("  Rouge pale      = Lesion secondaire (attention partielle)\n")
        f.write("  Zone sombre     = Tissu sain ignore\n")
        f.write("  Contour blanc   = Bordure de la zone de decision\n\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(
                f"Image : {r['image']}\n"
                f"  Grade reel    : {r['grade_reel']} - {GRADE_LABELS[r['grade_reel']]}\n"
                f"  Grade predit  : {r['grade_predit']} - {GRADE_LABELS[r['grade_predit']]}\n"
                f"  Confiance     : {r['confiance']}\n"
                f"  Zones         : {r['zones']}\n"
                f"  Couverture    : {r['couverture']}\n"
                f"  Statut        : {r['statut']}\n"
                + "-" * 40 + "\n"
            )
            print(
                f"Image : {r['image']} | Grade {r['grade_reel']} -> {r['grade_predit']} "
                f"| {r['statut']} | Confiance {r['confiance']}"
            )

    print(f"\n[OK] Resultats dans '{OUTPUT_DIR}/'")
    print(f"[OK] Resume : '{summary_path}'")


if __name__ == "__main__":
    main()

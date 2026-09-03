"""
==========================================================
  ETAPE 4 v3 : GRAD-CAM++ SIMPLIFIE ET CLINIQUE
  Script : gradcam_explain_v3.py

  Design epure :
  - 3 panneaux clairs (Original | Heatmap Rouge | Analyse)
  - Superposition rouge UNIQUEMENT sur les zones de lesions
  - Transparent = ignore par l'IA
  - Rouge fonce = lésion principale
  - Legende detaillee de chaque zone
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
import matplotlib.colors as mcolors
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split

from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================================
# CONFIGURATION
# ==========================================================
DESKTOP_PATH = r'C:\Users\Saoudi\OneDrive\Desktop\aptos2019-blindness-detection'
IMAGE_DIR    = os.path.join(DESKTOP_PATH, 'train_images_enhanced')
CSV_PATH     = os.path.join(DESKTOP_PATH, 'train.csv')
MODEL_PATH   = 'best_dr_model.pth'
OUTPUT_DIR   = 'gradcam_results_v3'
NUM_IMAGES   = 5
RANDOM_SEED  = 42

GRADE_LABELS = {
    0: "Grade 0 : Sain (pas de lesion)",
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


def compute_gradcam(model, tensor, target_layers, target_class, device):
    """
    Calcul Grad-CAM++ sur 3 couches, fusionne par MAXIMUM pixel-a-pixel.
    Le maximum capture toutes les zones de lesion, meme les petites.
    """
    tensor = tensor.to(device)
    maps   = []
    for layer in target_layers:
        cam_obj   = GradCAMPlusPlus(model=model, target_layers=[layer])
        targets   = [ClassifierOutputTarget(target_class)]
        grayscale = cam_obj(input_tensor=tensor, targets=targets)[0]
        if grayscale.max() > 0:
            grayscale = grayscale / grayscale.max()
        maps.append(grayscale)

    # Maximum pixel-a-pixel pour capturer toutes les lesions
    fused = np.maximum.reduce(maps)

    # Lissage leger pour reduire le bruit sans effacer les contours
    fused = cv2.GaussianBlur(fused, (9, 9), sigmaX=2.0)
    if fused.max() > 0:
        fused = fused / fused.max()
    return fused


def make_clinical_overlay(rgb, cam_map, threshold_percentile=55):
    """
    Cree une superposition CLINIQUE epuree :
    - Fond original de la retine (image de base)
    - Zones de lesion en ROUGE PUR semi-transparent (pas d'arc-en-ciel)
    - Zones non concernees : image originale inchangee
    
    Plus la zone est rouge/intense, plus l'IA y a focalise son attention.
    """
    h, w = rgb.shape[:2]

    # Seuil : ne garder que les top (100 - threshold_percentile)% d'activations
    threshold = np.percentile(cam_map, threshold_percentile)
    mask      = cam_map >= threshold

    # Colormap personnalisee : transparent -> rouge
    # (transparent = pas de lesion, rouge = lésion maximale)
    red_overlay = np.zeros((h, w, 4), dtype=np.float32)  # RGBA
    intensity   = cam_map.copy()
    intensity[~mask] = 0.0  # Zones sous le seuil : transparentes

    # Canal Rouge
    red_overlay[:, :, 0] = intensity             # R
    red_overlay[:, :, 1] = 0.0                   # G (pas de vert = rouge pur)
    red_overlay[:, :, 2] = 0.0                   # B
    red_overlay[:, :, 3] = intensity * 0.85      # Alpha : proportionnel a l'intensite

    # Image de base avec les zones non-lesion legèrement assombries
    base = rgb.copy()
    alpha_darken = np.where(mask, 1.0, 0.55)
    result_rgb = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        result_rgb[:, :, c] = base[:, :, c] * alpha_darken

    # Superposer le rouge
    alpha_channel = red_overlay[:, :, 3]
    result_rgb[:, :, 0] = np.clip(
        result_rgb[:, :, 0] * (1 - alpha_channel) + red_overlay[:, :, 0], 0, 1
    )
    result_rgb[:, :, 1] = np.clip(
        result_rgb[:, :, 1] * (1 - alpha_channel), 0, 1
    )
    result_rgb[:, :, 2] = np.clip(
        result_rgb[:, :, 2] * (1 - alpha_channel), 0, 1
    )

    return np.clip(result_rgb, 0, 1), mask


def make_intensity_map(cam_map):
    """
    Carte d'intensite simple (niveaux de rouge) avec legende :
    Chaque niveau de rouge correspond a une force d'attention differente.
    """
    # On cree une image RGB avec gradient rouge uniquement
    intensity_img = np.zeros((*cam_map.shape, 3), dtype=np.float32)
    intensity_img[:, :, 0] = cam_map  # Canal R seulement
    return intensity_img


# ==========================================================
# CREATION DE LA FIGURE
# ==========================================================
def create_figure(rgb, cam_map, overlay_img, lesion_mask,
                  pred_class, true_class, probs, img_name, output_path):
    """
    Figure epuree a 3 zones + 1 panneau d'explication :

    [Image Originale] | [Superposition Grad-CAM++ Rouge] | [Barre de probabilites]
                      |         + Legende des couleurs   |
    """
    correct      = pred_class == true_class
    status_str   = "CORRECT" if correct else "INCORRECT"
    status_color = '#2ecc71' if correct else '#e74c3c'
    pred_color   = GRADE_COLORS[pred_class]
    true_color   = GRADE_COLORS[true_class]

    fig = plt.figure(figsize=(22, 9))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        f"Analyse Grad-CAM++  |  {img_name}  |  Diagnostic : {status_str}",
        fontsize=14, color='white', fontweight='bold', y=1.01
    )

    gs = fig.add_gridspec(1, 3, wspace=0.08, width_ratios=[1, 1, 0.7])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    for ax in [ax1, ax2]:
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor('#0d1117')
        for sp in ax.spines.values():
            sp.set_edgecolor('#333355')

    # ---- PANNEAU 1 : Image originale ----
    ax1.set_title("Image Originale de la Retine", color='white', fontsize=11, pad=10)
    ax1.imshow(rgb)
    ax1.set_xlabel(
        f"Vrai diagnostic : {GRADE_LABELS[true_class]}",
        color=true_color, fontsize=10, fontweight='bold'
    )

    # ---- PANNEAU 2 : Superposition Grad-CAM++ clinique ----
    ax2.set_title("Zones d'Attention de l'IA (Grad-CAM++)", color='white', fontsize=11, pad=10)
    ax2.imshow(overlay_img)

    # Annotation de la legende directement sur le panneau
    ax2.set_xlabel(
        f"Diagnostic IA : {GRADE_LABELS[pred_class]}",
        color=pred_color, fontsize=10, fontweight='bold'
    )

    # Colorbar personnalisee a droite du panneau 2
    # On cree manuellement un gradient rouge
    gradient = np.linspace(0, 1, 256).reshape(256, 1)
    ax_cbar = ax2.inset_axes([1.01, 0.0, 0.04, 1.0])
    ax_cbar.imshow(gradient, aspect='auto', cmap='Reds',
                   origin='lower', extent=[0, 1, 0, 1])
    ax_cbar.set_xticks([])
    ax_cbar.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_cbar.set_yticklabels(
        ['Aucune\nattention', 'Faible', 'Modere', 'Eleve', 'Maximale'],
        color='white', fontsize=7
    )
    ax_cbar.set_facecolor('#0d1117')
    for sp in ax_cbar.spines.values():
        sp.set_edgecolor('#333355')

    # ---- PANNEAU 3 : Probabilites par grade ----
    ax3.set_facecolor('#161b22')
    ax3.set_title("Probabilites du Modele", color='white', fontsize=11, pad=10)
    for sp in ax3.spines.values():
        sp.set_edgecolor('#333355')

    y_pos      = np.arange(5)
    labels_bar = ['Gr.0\nSain', 'Gr.1\nLeger', 'Gr.2\nMod.', 'Gr.3\nSev.', 'Gr.4\nProlif.']
    bar_colors = [GRADE_COLORS[i] for i in range(5)]

    bars = ax3.barh(y_pos, probs * 100, color=bar_colors,
                    edgecolor='#aaaaaa', linewidth=0.5, height=0.6)

    for bar, p in zip(bars, probs):
        ax3.text(
            min(p * 100 + 1.5, 97),
            bar.get_y() + bar.get_height() / 2,
            f'{p*100:.1f}%',
            va='center', ha='left', color='white', fontsize=10, fontweight='bold'
        )

    ax3.set_xlim(0, 115)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels_bar, color='white', fontsize=10)
    ax3.set_xlabel('Probabilite (%)', color='white', fontsize=10)
    ax3.tick_params(axis='x', colors='white')

    # Mettre en evidence la barre predite avec un bord blanc epais
    bars[pred_class].set_linewidth(3.0)
    bars[pred_class].set_edgecolor('white')

    # Bloc d'explication sous le bar chart
    explication = (
        "LEGENDE DES COULEURS :\n\n"
        "  Rouge fonce / intense\n"
        "  -> Zone ou l'IA a le plus\n"
        "     focalise son attention\n"
        "     (lesion principale)\n\n"
        "  Rouge clair / pale\n"
        "  -> Zone d'attention secondaire\n"
        "     (lesion moins certaine)\n\n"
        "  Zone sombre (image normale)\n"
        "  -> Zone ignoree par l'IA\n"
        "     (tissu sain)\n\n"
        f"Lesion pct couvert : "
        f"{100*lesion_mask.sum()/(lesion_mask.shape[0]*lesion_mask.shape[1]):.1f}%"
    )
    ax3.text(0, -1.4, explication,
             transform=ax3.get_yaxis_transform(),
             color='#cccccc', fontsize=8.5,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d1117',
                       edgecolor='#333355', alpha=0.9))

    # Badge CORRECT / INCORRECT
    patch = mpatches.Patch(color=status_color, label=f'Diagnostic {status_str}')
    ax3.legend(handles=[patch], loc='lower right',
               facecolor='#0d1117', labelcolor='white', fontsize=10,
               framealpha=0.8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


# ==========================================================
# SCRIPT PRINCIPAL
# ==========================================================
def main():
    print("=" * 65)
    print("  ETAPE 4 v3 : GRAD-CAM++ CLINIQUE ET EPURE")
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

    # 3 couches pour capturer fine + mid + semantique
    target_layers = [
        model.features[2],
        model.features[4],
        model.features[7],
    ]
    print("Couches : features[2] + features[4] + features[7]")

    df = pd.read_csv(CSV_PATH)
    _, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['diagnosis'])
    _, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['diagnosis'])
    test_df = test_df.reset_index(drop=True)

    # 1 image par grade
    selected = []
    for grade in sorted(test_df['diagnosis'].unique())[:NUM_IMAGES]:
        row = test_df[test_df['diagnosis'] == grade].sample(
            n=1, random_state=RANDOM_SEED + grade).iloc[0]
        selected.append(row)

    print(f"\n{len(selected)} images selectionnees (1 par grade) :")
    for row in selected:
        print(f"  Grade {int(row['diagnosis'])} : {row['id_code']}")

    results = []
    print("\nGeneration des cartes Grad-CAM++...\n")

    for i, row in enumerate(selected):
        img_id     = row['id_code']
        true_class = int(row['diagnosis'])
        img_path   = os.path.join(IMAGE_DIR, f"{img_id}.png")

        print(f"[{i+1}/{len(selected)}] {img_id}  (Grade {true_class})")
        if not os.path.exists(img_path):
            print("  ATTENTION : image introuvable.")
            continue

        # Chargement
        rgb, tensor = load_image(img_path)
        pred_class, probs = predict(model, tensor, device)
        status = "CORRECT" if pred_class == true_class else "INCORRECT"
        print(f"  Predit : Grade {pred_class} ({GRADE_LABELS[pred_class]}) | {status}")

        # Grad-CAM++ multi-couches
        print("  Calcul Grad-CAM++...")
        cam_map = compute_gradcam(model, tensor, target_layers, pred_class, device)

        # Superposition clinique rouge
        overlay_img, lesion_mask = make_clinical_overlay(rgb, cam_map, threshold_percentile=55)

        pct_lesion = 100 * lesion_mask.sum() / (lesion_mask.shape[0] * lesion_mask.shape[1])
        print(f"  Zone d'attention : {pct_lesion:.1f}% de la retine")

        # Sauvegarde
        fname    = f"gradcam_v3_{i+1:02d}_{img_id}_true{true_class}_pred{pred_class}.png"
        out_path = os.path.join(OUTPUT_DIR, fname)

        create_figure(rgb, cam_map, overlay_img, lesion_mask,
                      pred_class, true_class, probs, img_id, out_path)

        print(f"  Sauvegarde : {fname}\n")
        results.append({
            'image': img_id,
            'grade_reel': true_class,
            'grade_predit': pred_class,
            'statut': status,
            'pct_attention': f'{pct_lesion:.1f}%',
            'confiance': f'{probs[pred_class]:.2%}'
        })

    # Resume textuel
    summary_path = os.path.join(OUTPUT_DIR, 'resume_gradcam_v3.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("ETAPE 4 v3 - GRAD-CAM++ CLINIQUE\n")
        f.write("=" * 60 + "\n\n")
        f.write("LEGENDE DES COULEURS :\n")
        f.write("  Rouge fonce/intense = Zone d'attention maximale (lesion principale)\n")
        f.write("  Rouge pale          = Attention secondaire (lesion moins certaine)\n")
        f.write("  Zone sombre         = Zone ignoree par l'IA (tissu sain)\n\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            line = (
                f"Image : {r['image']}\n"
                f"  Grade reel    : {r['grade_reel']} - {GRADE_LABELS[r['grade_reel']]}\n"
                f"  Grade predit  : {r['grade_predit']} - {GRADE_LABELS[r['grade_predit']]}\n"
                f"  Confiance     : {r['confiance']}\n"
                f"  Zone attention: {r['pct_attention']} de la retine\n"
                f"  Statut        : {r['statut']}\n"
                + "-" * 40 + "\n"
            )
            print(line)
            f.write(line)

    print(f"[OK] Resultats dans '{OUTPUT_DIR}/'")
    print(f"[OK] Resume : '{summary_path}'")


if __name__ == "__main__":
    main()

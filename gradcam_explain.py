"""
==========================================================
  ÉTAPE 4 : EXPLICABILITÉ PAR GRAD-CAM++
  Script : gradcam_explain.py
  
  Ce script :
  1. Charge votre modèle EfficientNet-B3 entraîné
  2. Sélectionne 5 images aléatoires du Test Set
  3. Applique Grad-CAM++ pour générer des cartes thermiques
  4. Crée de belles visualisations avec l'image originale,
     la heatmap seule, et l'image avec overlay thermique
  5. Sauvegarde tout dans le dossier 'gradcam_results/'
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
matplotlib.use('Agg')  # Backend sans ecran pour Windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ==========================================================
# CONFIGURATION
# ==========================================================
DESKTOP_PATH    = r'C:\Users\Saoudi\OneDrive\Desktop\aptos2019-blindness-detection'
IMAGE_DIR       = os.path.join(DESKTOP_PATH, 'train_images_enhanced')
CSV_PATH        = os.path.join(DESKTOP_PATH, 'train.csv')
MODEL_PATH      = 'best_dr_model.pth'
OUTPUT_DIR      = 'gradcam_results'
NUM_IMAGES      = 5        # Nombre d'images a analyser
RANDOM_SEED     = 42

# Noms lisibles des grades de la retinopathie diabetique
GRADE_LABELS = {
    0: "Grade 0 : Pas de RD",
    1: "Grade 1 : RD Legere",
    2: "Grade 2 : RD Moderee",
    3: "Grade 3 : RD Severe",
    4: "Grade 4 : RD Proliferative",
}

GRADE_COLORS = {
    0: '#2ecc71',   # vert
    1: '#f39c12',   # jaune-orange
    2: '#e67e22',   # orange
    3: '#e74c3c',   # rouge
    4: '#8e44ad',   # violet
}

# ==========================================================
# FONCTIONS UTILITAIRES
# ==========================================================

def crop_image_from_gray(img, tol=7):
    """Auto-crop des bordures noires (Ben Graham method)."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:
            return img
        img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
        img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
        img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
        return np.stack([img1, img2, img3], axis=-1)


def load_and_preprocess(img_path):
    """
    Charge une image, la recadre et la normalise.
    Renvoie:
      - rgb_display : image numpy (H,W,3) float [0,1] pour la visualisation
      - tensor      : tensor PyTorch pret pour le modele
    """
    image_cv = cv2.imread(img_path)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_cv = crop_image_from_gray(image_cv)
    image_cv_resized = cv2.resize(image_cv, (256, 256))
    
    # Version [0,1] pour la visualisation et la superposition heatmap
    rgb_display = image_cv_resized.astype(np.float32) / 255.0
    
    # Version tensor normalise pour le modele
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    tensor = transform(image_cv_resized).unsqueeze(0)
    
    return rgb_display, tensor


def predict(model, tensor, device):
    """Effectue une prediction et renvoie la classe et les probabilites."""
    model.eval()
    with torch.no_grad():
        tensor = tensor.to(device)
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_class = int(np.argmax(probs))
    return pred_class, probs


def generate_gradcam(model, tensor, target_layer, target_class, device):
    """Genere la heatmap Grad-CAM++ pour une image et une classe cible."""
    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(target_class)]
    tensor = tensor.to(device)
    grayscale_cam = cam(input_tensor=tensor, targets=targets)
    return grayscale_cam[0]  # shape (H, W)


def create_beautiful_figure(rgb_display, cam_map, pred_class, true_class,
                             probs, img_name, output_path):
    """
    Cree une figure de publication avec :
      - Colonne 1 : Image originale
      - Colonne 2 : Carte thermique Grad-CAM++ seule
      - Colonne 3 : Overlay (image + heatmap)
      - Colonne 4 : Bar chart des probabilites par classe
    """
    # Superposition heatmap sur l'image originale
    cam_overlay = show_cam_on_image(rgb_display, cam_map, use_rgb=True)
    
    # Couleur de la heatmap seule (colormap JET comme dans les articles medicaux)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam_map), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    pred_color = GRADE_COLORS[pred_class]
    true_color = GRADE_COLORS[true_class]
    correct = pred_class == true_class

    fig = plt.figure(figsize=(20, 6))
    fig.patch.set_facecolor('#1a1a2e')

    # ---- Titre principal ----
    status_str = "CORRECT" if correct else "INCORRECT"
    status_color = '#2ecc71' if correct else '#e74c3c'
    fig.suptitle(
        f"Analyse Grad-CAM++ | Image : {img_name} | Diagnostic : {status_str}",
        fontsize=14, color='white', fontweight='bold', y=1.01
    )

    # ---- Sous-titres de chaque colonne ----
    titles = [
        "Image Originale (Rétine)",
        "Carte Thermique Grad-CAM++\n(Zones d'attention de l'IA)",
        "Superposition (Overlay)",
        "Probabilités par Grade"
    ]

    axes = []
    for i in range(4):
        ax = fig.add_subplot(1, 4, i + 1)
        ax.set_facecolor('#16213e')
        ax.set_title(titles[i], color='white', fontsize=10, pad=8)
        axes.append(ax)

    # Col 1 : Image originale
    axes[0].imshow(rgb_display)
    axes[0].set_xlabel(
        f"Vrai : {GRADE_LABELS[true_class]}",
        color=true_color, fontsize=9, fontweight='bold'
    )

    # Col 2 : Heatmap seule
    axes[1].imshow(heatmap_colored)
    axes[1].set_xlabel(
        "Rouge = Attention max | Bleu = Peu attentionne",
        color='#aaaaaa', fontsize=8
    )

    # Col 3 : Overlay
    axes[2].imshow(cam_overlay)
    axes[2].set_xlabel(
        f"Predit : {GRADE_LABELS[pred_class]}",
        color=pred_color, fontsize=9, fontweight='bold'
    )

    # Supprimer les ticks pour les images
    for ax in axes[:3]:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#444466')

    # Col 4 : Bar chart des probabilites
    ax_bar = axes[3]
    class_names = ['Gr.0\nSain', 'Gr.1\nLeger', 'Gr.2\nModere', 'Gr.3\nSevere', 'Gr.4\nProlif.']
    bar_colors = [GRADE_COLORS[i] for i in range(5)]
    bars = ax_bar.barh(class_names, probs * 100, color=bar_colors, edgecolor='white', linewidth=0.5)
    
    # Valeurs sur les barres
    for bar, prob in zip(bars, probs):
        ax_bar.text(
            min(prob * 100 + 1, 97), bar.get_y() + bar.get_height() / 2,
            f'{prob * 100:.1f}%',
            va='center', ha='left', color='white', fontsize=9, fontweight='bold'
        )

    ax_bar.set_xlim(0, 105)
    ax_bar.set_xlabel('Probabilite (%)', color='white', fontsize=9)
    ax_bar.tick_params(colors='white', labelsize=9)
    ax_bar.set_facecolor('#16213e')
    for spine in ax_bar.spines.values():
        spine.set_edgecolor('#444466')
    
    # Mettre en evidence la barre predite
    bars[pred_class].set_linewidth(2.5)
    bars[pred_class].set_edgecolor('white')

    # Legende : Correct/Incorrect
    patch = mpatches.Patch(color=status_color, label=f'Diagnostic {status_str}')
    ax_bar.legend(handles=[patch], loc='lower right', facecolor='#1a1a2e',
                  labelcolor='white', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close()


# ==========================================================
# SCRIPT PRINCIPAL
# ==========================================================
def main():
    print("=" * 60)
    print("  ETAPE 4 : GRAD-CAM++ (Explicabilite du Modele)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil : {device}")

    # --- Chargement du modele ---
    print("\nChargement du modele EfficientNet-B3...")
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True, map_location=device))
    model = model.to(device)
    model.eval()

    # Couche cible pour Grad-CAM++ (derniere couche convolutive)
    target_layer = model.features[-1]
    print(f"Couche cible : model.features[-1]")

    # --- Chargement du CSV et constitution du test set ---
    df = pd.read_csv(CSV_PATH)
    _, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['diagnosis'])
    _, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['diagnosis'])
    test_df = test_df.reset_index(drop=True)

    # Selection de 1 image par grade (plus representatif qu'aleatoire pur)
    selected = []
    available_grades = sorted(test_df['diagnosis'].unique())
    for grade in available_grades[:NUM_IMAGES]:
        candidates = test_df[test_df['diagnosis'] == grade]
        row = candidates.sample(n=1, random_state=RANDOM_SEED + grade).iloc[0]
        selected.append(row)

    print(f"\n{len(selected)} images selectionnees (1 par grade):")
    for row in selected:
        print(f"  - {row['id_code']} | Grade reel : {row['diagnosis']} ({GRADE_LABELS[row['diagnosis']]})")

    # --- Boucle principale Grad-CAM++ ---
    print(f"\nGeneration des cartes Grad-CAM++...")
    results_summary = []

    for i, row in enumerate(selected):
        img_id    = row['id_code']
        true_class = int(row['diagnosis'])
        img_path  = os.path.join(IMAGE_DIR, f"{img_id}.png")

        print(f"\n  [{i+1}/{len(selected)}] Image : {img_id} (Grade reel : {true_class})")

        if not os.path.exists(img_path):
            print(f"    ATTENTION : Image introuvable, ignoree.")
            continue

        # Chargement
        rgb_display, tensor = load_and_preprocess(img_path)

        # Prediction
        pred_class, probs = predict(model, tensor, device)
        status = "CORRECT" if pred_class == true_class else "INCORRECT"
        print(f"    Prediction : {pred_class} ({GRADE_LABELS[pred_class]}) | {status}")
        print(f"    Probabilites : {[f'{p:.2%}' for p in probs]}")

        # Grad-CAM++ (sur la classe predite ET sur la vraie classe)
        cam_map_pred = generate_gradcam(model, tensor, target_layer, pred_class, device)

        # Construction du nom de fichier de sortie
        output_filename = f"gradcam_{i+1:02d}_img_{img_id}_true{true_class}_pred{pred_class}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        # Creation de la figure
        create_beautiful_figure(
            rgb_display, cam_map_pred, pred_class, true_class,
            probs, img_id, output_path
        )
        print(f"    Sauvegarde : {output_path}")

        results_summary.append({
            'image': img_id,
            'grade_reel': true_class,
            'grade_reel_label': GRADE_LABELS[true_class],
            'grade_predit': pred_class,
            'grade_predit_label': GRADE_LABELS[pred_class],
            'statut': status,
            'confiance': f'{probs[pred_class]:.2%}',
        })

    # --- Résumé final (sauvegardé en texte) ---
    print("\n" + "=" * 60)
    print("  RESUME DES RESULTATS GRAD-CAM++")
    print("=" * 60)
    summary_path = os.path.join(OUTPUT_DIR, 'resume_gradcam.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("RESUME DE L'ETAPE 4 - GRAD-CAM++\n")
        f.write("=" * 60 + "\n\n")
        for r in results_summary:
            line = (
                f"Image : {r['image']}\n"
                f"  Grade reel    : {r['grade_reel']} - {r['grade_reel_label']}\n"
                f"  Grade predit  : {r['grade_predit']} - {r['grade_predit_label']}\n"
                f"  Confiance     : {r['confiance']}\n"
                f"  Statut        : {r['statut']}\n"
                + "-" * 40 + "\n"
            )
            print(line)
            f.write(line)

    print(f"\n[OK] {len(results_summary)} figures sauvegardees dans '{OUTPUT_DIR}/'")
    print(f"[OK] Resume texte sauvegarde dans '{summary_path}'")
    print("\nOuvrez les images PNG pour voir les zones d'attention de l'IA !")


if __name__ == "__main__":
    main()

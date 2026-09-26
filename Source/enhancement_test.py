import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import joblib
import numpy as np
import os
import cv2

# --- 1. QUALITÉ (SVM + DENSENET) ---
def get_densenet_extractor():
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    extractor = torch.nn.Sequential(
        model.features,
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten()
    )
    extractor.eval()
    return extractor

def preprocess_for_qa(image_path):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def assess_quality(image_path, extractor, svm_model):
    input_tensor = preprocess_for_qa(image_path)
    with torch.no_grad():
        features = extractor(input_tensor).numpy()
    prediction = svm_model.predict(features)[0]
    return prediction

# --- 2. FONCTIONS D'AMÉLIORATION ---
def apply_clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)
    merged_lab = cv2.merge((cl, a_channel, b_channel))
    return cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)

def apply_gamma_correction(img, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def apply_unsharp_mask(img):
    gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
    unsharp_image = cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)
    return unsharp_image

# --- 3. DIAGNOSTIC MATHÉMATIQUE ET ROUTAGE ---
def diagnose_and_enhance(img, quality_class):
    """Diagnostique le problème précis et applique le traitement adéquat"""
    if quality_class == 0:
        return img, "Classe 0 (Bonne) -> AUCUN TRAITEMENT"
        
    # Convertir en niveaux de gris pour l'analyse mathématique
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Calcul des métriques
    mean_val = np.mean(gray)
    std_val = np.std(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # 1. Problème d'Illumination (Trop sombre)
    if mean_val < 45:
        # Traitement : On éclaire (Gamma 1.5) puis on sort les détails (CLAHE)
        img_gamma = apply_gamma_correction(img, gamma=1.5)
        img_final = apply_clahe(img_gamma)
        return img_final, f"Sombre (Mean:{mean_val:.0f}) -> GAMMA(1.5) + CLAHE"
        
    # 2. Problème d'Illumination (Trop clair/Reflets)
    elif mean_val > 150:
        img_gamma = apply_gamma_correction(img, gamma=0.7)
        img_final = apply_clahe(img_gamma)
        return img_final, f"Sur-expose (Mean:{mean_val:.0f}) -> GAMMA(0.7) + CLAHE"
        
    # 3. Flou Sévère
    elif laplacian_var < 150:
        img_unsharp = apply_unsharp_mask(img)
        img_final = apply_clahe(img_unsharp)
        return img_final, f"Flou (Laplacian:{laplacian_var:.0f}) -> UNSHARP + CLAHE"
        
    # 4. Manque de Contraste (ou défaut par défaut si Classe 1 ou 2)
    else:
        img_final = apply_clahe(img)
        return img_final, f"Faible Contraste (Std:{std_val:.0f}) -> CLAHE SEUL"

def add_text_to_image(img, text):
    """Ajoute le texte de description sur l'image pour un rendu visuel clair"""
    # Créer une bande noire en bas de l'image
    h, w = img.shape[:2]
    bar_height = 50
    bar = np.zeros((bar_height, w, 3), dtype=np.uint8)
    img_with_bar = np.vstack((img, bar))
    
    # Ajouter le texte
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Ajuster la taille de la police selon la largeur de l'image
    font_scale = w / 1500.0 if w > 500 else 0.5
    thickness = 2
    cv2.putText(img_with_bar, text, (20, h + 35), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)
    
    return img_with_bar

# --- PROGRAMME PRINCIPAL ---
def main():
    print("Chargement des modèles de qualité...")
    svm_path = 'quickqual_dn121_512.pkl'
    svm_model = joblib.load(svm_path)
    extractor = get_densenet_extractor()

    # NOUVEAUX CHEMINS (Sur le Bureau)
    DESKTOP_PATH = r'C:\Users\Saoudi\OneDrive\Desktop\aptos2019-blindness-detection'
    aptos_train_dir = os.path.join(DESKTOP_PATH, 'train_images')
    # Test sur 15 images pour avoir plus de cas différents
    test_images = os.listdir(aptos_train_dir)[:15]
    
    output_dir = 'resultats_hybrides_intelligents'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n--- Début du pipeline adaptatif (Deep Learning + OpenCV) ---")
    for i, img_name in enumerate(test_images):
        img_path = os.path.join(aptos_train_dir, img_name)
        original_img = cv2.imread(img_path)
        
        # 1. ÉVALUATION DE LA QUALITÉ (DEEP LEARNING)
        quality_class = assess_quality(img_path, extractor, svm_model)
        
        # 2. DIAGNOSTIC MATHÉMATIQUE ET TRAITEMENT
        enhanced_img, applied_treatment = diagnose_and_enhance(original_img, quality_class)
        print(f"[{i+1}/15] {img_name} : {applied_treatment}")
        
        # 3. ANNOTATION VISUELLE ET SAUVEGARDE
        # On redimensionne un peu pour que la comparaison côte à côte ne soit pas géante
        original_resized = cv2.resize(original_img, (512, 512))
        enhanced_resized = cv2.resize(enhanced_img, (512, 512))
        
        # Texte sur les images
        original_annotated = add_text_to_image(original_resized, f"ORIGINALE (SVM Classe {quality_class})")
        enhanced_annotated = add_text_to_image(enhanced_resized, applied_treatment)
        
        # Concaténer
        comparison = np.hstack((original_annotated, enhanced_annotated))
        
        # Sauvegarder
        cv2.imwrite(os.path.join(output_dir, f"test_{img_name}"), comparison)

if __name__ == "__main__":
    main()

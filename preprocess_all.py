import os
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import joblib
from tqdm import tqdm

# --- 1. CHARGEMENT DES MODÈLES (QUALITY ASSESSMENT) ---
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

def assess_quality(image_path, extractor, svm_model, device):
    input_tensor = preprocess_for_qa(image_path).to(device)
    with torch.no_grad():
        features = extractor(input_tensor).cpu().numpy()
    prediction = svm_model.predict(features)[0]
    return prediction

# --- 2. FONCTIONS D'AMÉLIORATION OPENCV ---
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
    return cv2.addWeighted(img, 1.5, gaussian, -0.5, 0)

def diagnose_and_enhance(img, quality_class):
    # Classe 0 : On ne touche à rien
    if quality_class == 0:
        return img
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean_val = np.mean(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if mean_val < 45:
        # Trop sombre
        return apply_clahe(apply_gamma_correction(img, gamma=1.5))
    elif mean_val > 150:
        # Trop clair
        return apply_clahe(apply_gamma_correction(img, gamma=0.7))
    elif laplacian_var < 150:
        # Flou
        return apply_clahe(apply_unsharp_mask(img))
    else:
        # Délavé (Contraste faible)
        return apply_clahe(img)

# --- 3. SCRIPT PRINCIPAL DE TRAITEMENT MASSIF ---
def main():
    print("--- Préparation du Traitement de Masse (Étape 1 + 2) ---")
    
    # NOUVEAUX CHEMINS (Sur le Bureau)
    DESKTOP_PATH = r'C:\Users\Saoudi\OneDrive\Desktop\aptos2019-blindness-detection'
    input_dir = os.path.join(DESKTOP_PATH, 'train_images')
    output_dir = os.path.join(DESKTOP_PATH, 'train_images_enhanced')
    svm_path = 'quickqual_dn121_512.pkl'
    
    # Vérification
    if not os.path.exists(input_dir):
        print(f"Erreur: Le dossier source {input_dir} est introuvable.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Utiliser le GPU si possible (accélère le DenseNet)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil de calcul utilisé : {device}")
    
    # Chargement Modèles
    print("Chargement de DenseNet121 et du SVM QuickQual...")
    svm_model = joblib.load(svm_path)
    extractor = get_densenet_extractor().to(device)
    
    # Lister toutes les images
    all_images = [f for f in os.listdir(input_dir) if f.endswith('.png')]
    total_images = len(all_images)
    print(f"{total_images} images trouvées. Début du traitement...")
    
    # Compteurs pour vos statistiques
    stats = {0: 0, 1: 0, 2: 0}
    
    # Boucle de traitement avec barre de progression (tqdm)
    for img_name in tqdm(all_images, desc="Amélioration des images"):
        input_path = os.path.join(input_dir, img_name)
        output_path = os.path.join(output_dir, img_name)
        
        # Si l'image a déjà été traitée (pratique si le script plante et qu'on le relance)
        if os.path.exists(output_path):
            continue
            
        try:
            # 1. Évaluation Qualité
            quality_class = assess_quality(input_path, extractor, svm_model, device)
            stats[quality_class] += 1
            
            # 2. Traitement OpenCV
            img = cv2.imread(input_path)
            enhanced_img = diagnose_and_enhance(img, quality_class)
            
            # 3. Sauvegarde (même nom, même format)
            cv2.imwrite(output_path, enhanced_img)
            
        except Exception as e:
            print(f"\nErreur sur l'image {img_name}: {str(e)}")

    print("\n--- TRAITEMENT TERMINÉ ---")
    print(f"Statistiques des qualités détectées par QuickQual :")
    print(f"Classe 0 (Parfaites, ignorées) : {stats[0]}")
    print(f"Classe 1 (Utilisables, traitées) : {stats[1]}")
    print(f"Classe 2 (Rejet/Mauvaises, traitées) : {stats[2]}")
    print(f"\nToutes les images améliorées sont prêtes dans : {output_dir}")
    print("Vous pouvez maintenant lancer 'train.py' en toute sécurité !")

if __name__ == "__main__":
    main()

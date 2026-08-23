import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from tqdm import tqdm

# --- FOCAL LOSS (Solution Avancée pour le Déséquilibre) ---
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Optionnel: Poids supplémentaires
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 1. Calcul de l'entropie croisée classique
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        # 2. Calcul de "pt" (Probabilité que le modèle avait raison)
        pt = torch.exp(-ce_loss)
        # 3. Formule du Focal Loss : (1 - pt)^gamma * CE_loss
        # Plus l'image est facile (pt proche de 1), plus (1-pt) s'approche de 0 -> on l'ignore.
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = focal_loss * alpha_t

        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

# --- 1. DATASET ---
class APTOSDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = f"{self.dataframe.iloc[idx, 0]}.png"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = int(self.dataframe.iloc[idx, 1])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

# --- 2. ENTRAÎNEMENT ET ÉVALUATION ---
def main():
    print("--- Configuration de l'entraînement avec FOCAL LOSS ---")
    
    # Paramètres
    BATCH_SIZE = 16
    EPOCHS = 15 # On augmente un peu car le Focal Loss apprend plus prudemment
    LEARNING_RATE = 1e-4
    
    # NOUVEAUX CHEMINS (Sur le Bureau, en dehors du dossier de travail)
    DESKTOP_PATH = r'C:\Users\Saoudi\OneDrive\Desktop\aptos2019-blindness-detection'
    IMAGE_DIR = os.path.join(DESKTOP_PATH, 'train_images_enhanced')
    CSV_PATH = os.path.join(DESKTOP_PATH, 'train.csv')
    
    if not os.path.exists(IMAGE_DIR):
        print(f"ERREUR FATALE: Le dossier {IMAGE_DIR} n'existe pas.")
        print("Veuillez d'abord exécuter 'preprocess_all.py' pour créer les images améliorées sur le bureau.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entraînement sur : {device}")
    
    # Lecture des données
    df = pd.read_csv(CSV_PATH)
    
    # SPLIT : Train(70%), Val(15%), Test(15%)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['diagnosis'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['diagnosis'])
    
    print(f"Images d'entraînement : {len(train_df)}")
    print(f"Images de validation : {len(val_df)}")
    print(f"Images de test : {len(test_df)}")

    # Transformations
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # DataLoaders
    train_loader = DataLoader(APTOSDataset(train_df, IMAGE_DIR, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(APTOSDataset(val_df, IMAGE_DIR, val_test_transform), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(APTOSDataset(test_df, IMAGE_DIR, val_test_transform), batch_size=BATCH_SIZE, shuffle=False)

    # Modèle ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 5)
    model = model.to(device)

    # --- CHANGEMENT MAJEUR : FOCAL LOSS ---
    # Remplacement de CrossEntropyLoss par FocalLoss avec gamma=2.0 (le standard optimal)
    criterion = FocalLoss(gamma=2.0)
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- DÉBUT DE LA BOUCLE D'ENTRAÎNEMENT ---")
    best_kappa = -1.0
    
    for epoch in range(EPOCHS):
        print(f"\nÉpoque {epoch+1}/{EPOCHS}")
        
        # --- Mode Entraînement ---
        model.train()
        running_loss = 0.0
        
        for inputs, labels in tqdm(train_loader, desc="Entraînement"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        print(f"Perte (Focal Loss) moyenne : {running_loss/len(train_loader):.4f}")
        
        # --- Mode Validation ---
        model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc="Validation"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                val_preds.extend(predicted.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                
        # Calcul du Kappa
        val_kappa = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
        val_acc = accuracy_score(val_labels, val_preds)
        
        print(f"Validation - Accuracy: {val_acc:.4f} | Kappa Score: {val_kappa:.4f}")
        
        # Sauvegarde
        if val_kappa > best_kappa:
            best_kappa = val_kappa
            torch.save(model.state_dict(), 'best_dr_model.pth')
            print(">>> Nouveau meilleur modèle sauvegardé ! (best_dr_model.pth)")

    # --- 3. ÉVALUATION FINALE SUR LE SET DE TEST ---
    print("\n--- ÉVALUATION FINALE SUR LE TEST SET ---")
    model.load_state_dict(torch.load('best_dr_model.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test Final"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calcul des métriques
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    print("\n================ RÉSULTATS FINAUX (AVEC FOCAL LOSS) ================")
    print(f"Exactitude (Accuracy) globale : {acc:.4f}")
    print(f"Quadratic Weighted Kappa (QWK) : {kappa:.4f}")
    
    print("\nRapport de Classification (Précision, Rappel, F1 par classe) :")
    print(classification_report(all_labels, all_preds, target_names=['0 (Sain)', '1 (Léger)', '2 (Modéré)', '3 (Sévère)', '4 (Prolifératif)']))
    
    print("\nMatrice de Confusion :")
    print(confusion_matrix(all_labels, all_preds))
    print("====================================================================")

if __name__ == "__main__":
    main()

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, cohen_kappa_score
from tqdm import tqdm

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): return img 
        else:
            img1 = img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2 = img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3 = img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

class APTOSDataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
    def __len__(self): return len(self.dataframe)
    def __getitem__(self, idx):
        img_name = f"{self.dataframe.iloc[idx, 0]}.png"
        img_path = os.path.join(self.image_dir, img_name)
        image_cv = cv2.imread(img_path)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        image_cv = crop_image_from_gray(image_cv)
        image = Image.fromarray(image_cv)
        label = int(self.dataframe.iloc[idx, 1])
        if self.transform: image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)

def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.ylabel('Vraie Classe')
    plt.xlabel('Prédiction')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    print("--- GÉNÉRATION DES GRAPHIQUES POUR LE MODÈLE ACTUEL ---")
    BATCH_SIZE = 12
    DESKTOP_PATH = r'C:\Users\Saoudi\OneDrive\Desktop\aptos2019-blindness-detection'
    IMAGE_DIR = os.path.join(DESKTOP_PATH, 'train_images_enhanced')
    CSV_PATH = os.path.join(DESKTOP_PATH, 'train.csv')
    
    os.makedirs('courbes_entrainement', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(CSV_PATH)
    _, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['diagnosis'])
    _, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['diagnosis'])

    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    test_loader = DataLoader(APTOSDataset(test_df, IMAGE_DIR, val_test_transform), batch_size=BATCH_SIZE, shuffle=False)

    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model.load_state_dict(torch.load('best_dr_model.pth', weights_only=True))
    model = model.to(device)
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Test Final avec TTA"):
            inputs, labels = inputs.to(device), labels.to(device)
            out1 = model(inputs)
            out2 = model(torch.flip(inputs, [3]))
            out3 = model(torch.flip(inputs, [2]))
            final_out = (out1 + out2 + out3) / 3.0
            _, predicted = final_out.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
    
    with open('courbes_entrainement/rapport_test_ultime.txt', 'w') as f:
        f.write("================ RÉSULTATS DU CHALLENGE ================\n")
        f.write(f"Exactitude Globale : {acc:.4f}\n")
        f.write(f"QWK : {kappa:.4f}\n\n")
        f.write("Rapport de Classification :\n")
        f.write(classification_report(all_labels, all_preds, target_names=['0', '1', '2', '3', '4']))
        f.write("\n========================================================\n")
        
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, "Matrice de Confusion Ultime", "courbes_entrainement/cm_ultime.png")
    print("\n✅ Matrice de confusion et rapport texte sauvegardés dans 'courbes_entrainement/'")

if __name__ == "__main__":
    main()

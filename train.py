import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
    print("--- CHALLENGE ACCEPTÉ : LE MODÈLE ULTIME (AVEC COURBES) ---")
    BATCH_SIZE = 12 
    EPOCHS = 20
    
    DESKTOP_PATH = r'C:\Users\Saoudi\OneDrive\Desktop\aptos2019-blindness-detection'
    IMAGE_DIR = os.path.join(DESKTOP_PATH, 'train_images_enhanced')
    CSV_PATH = os.path.join(DESKTOP_PATH, 'train.csv')
    
    os.makedirs('courbes_entrainement', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    df = pd.read_csv(CSV_PATH)
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['diagnosis'])
    val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['diagnosis'])

    class_counts = train_df['diagnosis'].value_counts().sort_index().values
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[label] for label in train_df['diagnosis']]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(APTOSDataset(train_df, IMAGE_DIR, train_transform), batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(APTOSDataset(val_df, IMAGE_DIR, val_test_transform), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(APTOSDataset(test_df, IMAGE_DIR, val_test_transform), batch_size=BATCH_SIZE, shuffle=False)

    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 5)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=EPOCHS)

    # --- Historique pour les courbes ---
    history = {'train_loss': [], 'val_acc': [], 'val_kappa': []}
    best_val_acc = -1.0
    
    for epoch in range(EPOCHS):
        print(f"\nÉpoque {epoch+1}/{EPOCHS}")
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc="Entraînement"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            
        epoch_loss = running_loss / len(train_loader)
        history['train_loss'].append(epoch_loss)
            
        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = outputs.max(1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(val_labels, val_preds)
        kappa = cohen_kappa_score(val_labels, val_preds, weights='quadratic')
        history['val_acc'].append(acc)
        history['val_kappa'].append(kappa)
        
        print(f"Perte: {epoch_loss:.4f} | Val Accuracy: {acc:.4f} | Val Kappa: {kappa:.4f}")
        
        if acc > best_val_acc:
            best_val_acc = acc
            torch.save(model.state_dict(), 'best_dr_model.pth')
            print(">>> Nouveau meilleur modèle sauvegardé !")

    # --- SAUVEGARDE DES COURBES D'ENTRAÎNEMENT ---
    print("\nGénération des courbes d'entraînement...")
    epochs_range = range(1, EPOCHS + 1)
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='red', marker='o')
    plt.title('Évolution de la Perte (Label Smoothing)')
    plt.xlabel('Époques')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy', color='blue', marker='x')
    plt.plot(epochs_range, history['val_kappa'], label='Validation QWK', color='green', marker='s')
    plt.title('Évolution des Performances (Validation)')
    plt.xlabel('Époques')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('courbes_entrainement/courbe_ultime.png')

    # --- ÉVALUATION FINALE AVEC TTA ---
    print("\n--- ÉVALUATION AVEC TTA (Test-Time Augmentation) ---")
    model.load_state_dict(torch.load('best_dr_model.pth', weights_only=True))
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
            
    # --- SAUVEGARDE DES RÉSULTATS DE TEST ---
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
    
    print("\n[OK] Les graphiques, la matrice et le rapport texte ont ete sauvegardes dans 'courbes_entrainement/'")

if __name__ == "__main__":
    main()

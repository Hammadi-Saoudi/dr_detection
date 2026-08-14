"""
CNN Classifier pour l'évaluation de qualité des images rétiniennes (dataset EyeQ).
Classes : Good (0), Usable (1), Reject (2)

Usage rapide (test sur un petit échantillon, utile pour vérifier que tout fonctionne) :
    python cnn_classifier.py --sample_size 500 --epochs 3

Entraînement complet :
    python cnn_classifier.py --epochs 20 --batch_size 32
"""

import os
import argparse

import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models

from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score,
    confusion_matrix, classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns

CLASS_NAMES = ["Good", "Usable", "Reject"]
LABEL_MAP = {name: i for i, name in enumerate(CLASS_NAMES)}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class EyeQDataset(Dataset):
    """
    Lit les CSV officiels du repo HzFu/EyeQ (Label_EyeQ_train.csv / Label_EyeQ_test.csv)
    et charge les images correspondantes depuis images_dir.
    """

    def __init__(self, csv_path, images_dir, transform=None):
        self.df = pd.read_csv(csv_path)
        self.images_dir = images_dir
        self.transform = transform
        self._resolve_columns()

    def _resolve_columns(self):
        # Rend le parsing robuste si les noms de colonnes varient légèrement
        cols_lower = {c.lower(): c for c in self.df.columns}
        self.img_col = cols_lower.get("image", self.df.columns[0])
        quality_candidates = [c for c in self.df.columns if "quality" in c.lower()]
        self.quality_col = quality_candidates[0] if quality_candidates else self.df.columns[1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row[self.img_col])
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_name += ".jpeg"
        img_path = os.path.join(self.images_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        label_raw = row[self.quality_col]
        if isinstance(label_raw, str):
            label = LABEL_MAP[label_raw.strip().capitalize()]
        else:
            label = int(label_raw)

        return image, label


def get_transforms():
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


# ---------------------------------------------------------------------------
# Modèle
# ---------------------------------------------------------------------------
def build_model(num_classes=3):
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Boucles d'entraînement / évaluation
# ---------------------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, return_predictions=False):
    model.eval()
    running_loss, all_preds, all_labels = 0.0, [], []
    for images, labels in tqdm(loader, desc="Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    metrics = {
        "loss": running_loss / len(loader.dataset),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_macro": f1_score(all_labels, all_preds, average="macro"),
        # Kappa pondéré quadratique : pertinent car les classes sont ordinales
        # (Good > Usable > Reject) -> pénalise plus une erreur "loin" (Good<->Reject)
        "weighted_kappa": cohen_kappa_score(all_labels, all_preds, weights="quadratic"),
    }

    if return_predictions:
        return metrics, all_labels, all_preds
    return metrics


def plot_confusion_matrix(labels, preds, save_path="confusion_matrix.png"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité terrain")
    plt.title("Matrice de confusion - CNN EyeQ")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Matrice de confusion sauvegardée : {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="EyeQ_repo/data/Label_EyeQ_train.csv")
    parser.add_argument("--test_csv", type=str, default="EyeQ_repo/data/Label_EyeQ_test.csv")
    parser.add_argument("--images_dir", type=str, default="data/eyepacs_images")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--sample_size", type=int, default=None,
        help="Limiter le nombre d'images d'entraînement (utile pour tester rapidement le pipeline)"
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    train_tf, eval_tf = get_transforms()

    full_train_dataset = EyeQDataset(args.train_csv, args.images_dir, transform=train_tf)

    if args.sample_size:
        full_train_dataset.df = full_train_dataset.df.sample(
            n=min(args.sample_size, len(full_train_dataset.df)), random_state=42
        ).reset_index(drop=True)
        print(f"Mode échantillon : {len(full_train_dataset)} images utilisées")

    val_size = int(len(full_train_dataset) * args.val_split)
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    # Le split de validation doit utiliser les transforms d'évaluation (pas d'augmentation)
    val_dataset.dataset.transform = eval_tf

    test_dataset = EyeQDataset(args.test_csv, args.images_dir, transform=eval_tf)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model(num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_kappa = -1
    best_ckpt = os.path.join(args.output_dir, "best_cnn_eyeq.pth")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_metrics["weighted_kappa"])

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} acc: {train_acc:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} acc: {val_metrics['accuracy']:.4f} "
            f"f1: {val_metrics['f1_macro']:.4f} kappa: {val_metrics['weighted_kappa']:.4f}"
        )

        if val_metrics["weighted_kappa"] > best_kappa:
            best_kappa = val_metrics["weighted_kappa"]
            torch.save(model.state_dict(), best_ckpt)
            print(f"  -> Nouveau meilleur modèle sauvegardé (kappa={best_kappa:.4f})")

    print("\n=== Évaluation finale sur le test set ===")
    model.load_state_dict(torch.load(best_ckpt))
    test_metrics, test_labels, test_preds = evaluate(
        model, test_loader, criterion, device, return_predictions=True
    )
    print(test_metrics)
    print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES))
    plot_confusion_matrix(
        test_labels, test_preds,
        save_path=os.path.join(args.output_dir, "confusion_matrix_test.png")
    )


if __name__ == "__main__":
    main()
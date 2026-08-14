"""
Script de préparation des données EyeQ.

Étapes réalisées automatiquement :
1. Clone le repo HzFu/EyeQ (labels de qualité + script de prétraitement officiel)
2. Télécharge EyePACS via l'API Kaggle (pas de téléchargement manuel dans un navigateur)
3. Lance le prétraitement EyeQ sur les images téléchargées

Prérequis MANUELS (une seule fois, ~2 minutes) :
1. Créer un compte Kaggle : https://www.kaggle.com
2. Accepter les règles de la compétition (obligatoire pour débloquer le download API) :
   https://www.kaggle.com/c/diabetic-retinopathy-detection/rules
3. Récupérer un token API : Kaggle > Account > "Create New API Token"
   -> télécharge un fichier kaggle.json
4. Placer ce fichier dans ~/.kaggle/kaggle.json (Colab : uploader puis
   `mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`)

Usage :
    python download_data.py                     # téléchargement complet (~80 Go)
    python download_data.py --sample_only        # skip le download EyePACS (si déjà présent)
"""

import os
import subprocess
import argparse


def run(cmd):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def clone_eyeq_repo(dest="EyeQ_repo"):
    if not os.path.exists(dest):
        run(f"git clone https://github.com/HzFu/EyeQ.git {dest}")
    else:
        print(f"{dest} existe déjà, clone ignoré")


def check_kaggle_credentials():
    kaggle_json = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json):
        raise FileNotFoundError(
            "kaggle.json introuvable dans ~/.kaggle/. "
            "Suis les prérequis manuels décrits en haut de ce fichier."
        )


def download_eyepacs(dest="data/eyepacs_raw"):
    check_kaggle_credentials()
    os.makedirs(dest, exist_ok=True)
    run(f"kaggle competitions download -c diabetic-retinopathy-detection -p {dest}")
    for f in os.listdir(dest):
        if f.endswith(".zip"):
            print(f"Décompression de {f} (peut prendre du temps, ~80 Go décompressés)...")
            run(f"unzip -q {os.path.join(dest, f)} -d {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Ne pas télécharger EyePACS (si les images sont déjà présentes localement)"
    )
    args = parser.parse_args()

    clone_eyeq_repo()

    if not args.skip_download:
        download_eyepacs()
    else:
        print("Téléchargement EyePACS ignoré (--skip_download)")

    print("\n=== Étape suivante : prétraitement des images ===")
    print("cd EyeQ_repo/EyeQ_preprocess")
    print("python EyeQ_process_main.py")
    print("\nLes labels de qualité sont déjà disponibles dans :")
    print("  EyeQ_repo/data/Label_EyeQ_train.csv")
    print("  EyeQ_repo/data/Label_EyeQ_test.csv")


if __name__ == "__main__":
    main()
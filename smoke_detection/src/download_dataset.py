"""
download_dataset.py
--------------------
Télécharge et organise automatiquement le dataset D-Fire (fire + smoke, format YOLO)
dans l'arborescence attendue par ce projet (data/images/{train,val,test},
data/labels/{train,val,test}).

D-Fire est distribué officiellement via OneDrive (non scriptable sans authentification
manuelle), mais une copie "prête à l'emploi" au même format YOLO est disponible sur
Kaggle : https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo

Ce script utilise donc l'API Kaggle par défaut.

Prérequis (une seule fois) :
    pip install kaggle
    -> Créer un token API sur https://www.kaggle.com/settings (bouton "Create New Token")
    -> Placer le fichier kaggle.json téléchargé dans ~/.kaggle/kaggle.json
       (chmod 600 ~/.kaggle/kaggle.json sous Linux/Mac)

Usage :
    python src/download_dataset.py --dest data --split-ratio 0.8 0.1 0.1

Si tu préfères la source officielle GAIA (images non pré-splittées, mais dataset
"canonique" cité dans le papier), les liens OneDrive sont affichés en fin de script :
    - Dataset complet  : https://1drv.ms/u/c/c0bd25b6b048b01d/EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw
    - Train/Val/Test    : https://1drv.ms/f/c/c0bd25b6b048b01d/Ema8FFze8mFIlM1Hn81BUUgBE3vnnmK4SQxybS-nHRt2pA
    (téléchargement manuel requis, OneDrive ne fournissant pas d'API publique simple)
"""

import argparse
import random
import shutil
from pathlib import Path

KAGGLE_DATASET = "sayedgamal99/smoke-fire-detection-yolo"

ONEDRIVE_LINKS = {
    "dataset (images + labels)": "https://1drv.ms/u/c/c0bd25b6b048b01d/EbLgD7bES4FDvUN37Grxn8QBF5gIBBc7YV2qklF08GCiBw",
    "train/val/test pré-splittés": "https://1drv.ms/f/c/c0bd25b6b048b01d/Ema8FFze8mFIlM1Hn81BUUgBE3vnnmK4SQxybS-nHRt2pA",
    "vidéos de surveillance": "https://1drv.ms/f/c/c0bd25b6b048b01d/EhT2Jy6L-YlGvZv-gXH2SnYBENQsnUW96LpZtv_6PngjYQ",
}

IMG_EXTS = {".jpg", ".jpeg", ".png"}


def parse_args():
    parser = argparse.ArgumentParser(description="Téléchargement du dataset D-Fire (via Kaggle)")
    parser.add_argument("--dest", type=str, default="data",
                         help="Dossier de destination final (structure images/labels)")
    parser.add_argument("--raw-dir", type=str, default="data/_raw_kaggle",
                         help="Dossier temporaire de téléchargement brut")
    parser.add_argument("--split-ratio", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                         metavar=("TRAIN", "VAL", "TEST"),
                         help="Ratios train/val/test si le dataset téléchargé n'est pas déjà splitté")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true",
                         help="Réutiliser un dossier --raw-dir déjà téléchargé (sans re-télécharger)")
    return parser.parse_args()


def download_from_kaggle(raw_dir: Path):
    try:
        import kagglehub
    except ImportError:
        print("Le package 'kagglehub' n'est pas installé. Installation...")
        import subprocess
        subprocess.run(["pip", "install", "--break-system-packages", "-q", "kagglehub"], check=True)
        import kagglehub

    print(f"⬇Téléchargement du dataset Kaggle '{KAGGLE_DATASET}'...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print(f"Téléchargé dans : {path}")

    raw_dir.mkdir(parents=True, exist_ok=True)
    for item in Path(path).iterdir():
        target = raw_dir / item.name
        if target.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)
    return raw_dir


def find_image_label_pairs(raw_dir: Path):
    """Cherche récursivement toutes les paires (image, label .txt) dans le dossier brut."""
    pairs = []
    all_images = [p for p in raw_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    for img_path in all_images:
        # Le label est censé porter le même nom, quelque part sous un dossier 'labels'
        candidates = [
            img_path.parent / f"{img_path.stem}.txt",
            img_path.parent.parent / "labels" / img_path.parent.name / f"{img_path.stem}.txt",
        ]
        label_path = next((c for c in candidates if c.exists()), None)
        if label_path is None:
            # Recherche large en dernier recours
            matches = list(raw_dir.rglob(f"{img_path.stem}.txt"))
            label_path = matches[0] if matches else None
        if label_path is not None:
            pairs.append((img_path, label_path))
    return pairs


def already_split(raw_dir: Path):
    """Détecte si le dataset téléchargé contient déjà train/val/test."""
    return any((raw_dir / "images" / s).exists() for s in ["train", "val", "test"])


def organize_pre_split(raw_dir: Path, dest: Path):
    for split in ["train", "val", "test"]:
        src_img = raw_dir / "images" / split
        src_lbl = raw_dir / "labels" / split
        if not src_img.exists():
            continue
        dst_img = dest / "images" / split
        dst_lbl = dest / "labels" / split
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for img in src_img.glob("*"):
            if img.suffix.lower() in IMG_EXTS:
                shutil.copy2(img, dst_img / img.name)
                lbl = src_lbl / f"{img.stem}.txt"
                if lbl.exists():
                    shutil.copy2(lbl, dst_lbl / lbl.name)
        print(f"  -> {split}: {len(list(dst_img.glob('*')))} images copiées")


def organize_and_split(pairs, dest: Path, ratios, seed):
    random.seed(seed)
    random.shuffle(pairs)

    n = len(pairs)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    splits = {
        "train": pairs[:n_train],
        "val": pairs[n_train:n_train + n_val],
        "test": pairs[n_train + n_val:],
    }

    for split, split_pairs in splits.items():
        img_dir = dest / "images" / split
        lbl_dir = dest / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for img_path, label_path in split_pairs:
            shutil.copy2(img_path, img_dir / img_path.name)
            shutil.copy2(label_path, lbl_dir / f"{img_path.stem}.txt")
        print(f"  -> {split}: {len(split_pairs)} images copiées")


def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir)
    dest = Path(args.dest)

    if not args.skip_download:
        download_from_kaggle(raw_dir)
    else:
        print(f"Téléchargement ignoré, réutilisation de : {raw_dir}")

    print("\nOrganisation du dataset...")
    if already_split(raw_dir):
        print("Dataset déjà splitté en train/val/test détecté.")
        organize_pre_split(raw_dir, dest)
    else:
        print("Pas de split détecté, recherche des paires image/label et split manuel...")
        pairs = find_image_label_pairs(raw_dir)
        print(f"{len(pairs)} paires image/label trouvées.")
        if not pairs:
            print("Aucune paire trouvée. Vérifie la structure du dossier téléchargé "
                  f"({raw_dir}) et adapte find_image_label_pairs() si nécessaire.")
            return
        organize_and_split(pairs, dest, args.split_ratio, args.seed)

    print(f"\nDataset prêt dans '{dest}/images' et '{dest}/labels'.")
    print("Vérifie l'intégrité avec :")
    print(f"    python src/check_dataset.py --data {dest}")

    print("\nSource officielle GAIA (dataset canonique du papier de référence), "
          "téléchargement manuel via OneDrive si besoin :")
    for name, url in ONEDRIVE_LINKS.items():
        print(f"   - {name}: {url}")


if __name__ == "__main__":
    main()

"""
check_dataset.py
-----------------
Vérifie l'intégrité du dataset avant l'entraînement :
    - chaque image a bien un fichier de labels correspondant
    - les labels respectent le format YOLO (5 valeurs, coordonnées 0-1)
    - statistiques de répartition des classes par split

Usage :
    python src/check_dataset.py --data data
"""

import argparse
from collections import Counter
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Vérification du dataset YOLO")
    parser.add_argument("--data", type=str, default="data", help="Racine du dataset")
    return parser.parse_args()


def check_split(images_dir: Path, labels_dir: Path, split_name: str):
    errors = []
    class_counts = Counter()
    n_images = 0
    n_boxes = 0

    if not images_dir.exists():
        print(f"Dossier images introuvable pour '{split_name}': {images_dir}")
        return

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        n_images += 1
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            errors.append(f"Label manquant pour {img_path.name}")
            continue

        with open(label_path, "r") as f:
            lines = [l.strip() for l in f if l.strip()]

        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{label_path.name}: ligne mal formée -> '{line}'")
                continue
            cls_id, x, y, w, h = parts
            try:
                x, y, w, h = float(x), float(y), float(w), float(h)
            except ValueError:
                errors.append(f"{label_path.name}: valeurs non numériques -> '{line}'")
                continue
            if not all(0.0 <= v <= 1.0 for v in (x, y, w, h)):
                errors.append(f"{label_path.name}: coordonnées hors [0,1] -> '{line}'")
            class_counts[cls_id] += 1
            n_boxes += 1

    print(f"\n--- Split: {split_name} ---")
    print(f"Images                : {n_images}")
    print(f"Boîtes annotées       : {n_boxes}")
    print(f"Répartition par classe: {dict(class_counts)}")
    if errors:
        print(f"{len(errors)} erreur(s) détectée(s) :")
        for e in errors[:20]:
            print(f"   - {e}")
        if len(errors) > 20:
            print(f"   ... et {len(errors) - 20} autres")
    else:
        print("Aucun problème détecté.")


def main():
    args = parse_args()
    root = Path(args.data)

    for split in ["train", "val", "test"]:
        check_split(root / "images" / split, root / "labels" / split, split)


if __name__ == "__main__":
    main()

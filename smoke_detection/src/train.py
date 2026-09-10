"""
train.py
--------
Entraîne un modèle YOLO (Ultralytics) pour la détection de fumée / feu.

Usage :
    python src/train.py --data data/smoke.yaml --model yolov8n.pt --epochs 100

Le modèle de base peut être :
    yolov8n.pt / yolov8s.pt / yolov8m.pt   (YOLOv8)
    yolo11n.pt / yolo11s.pt                (YOLO11, plus récent)
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement YOLO - Détection de fumée")
    parser.add_argument("--data", type=str, default="data/smoke.yaml",
                         help="Chemin vers le fichier de config du dataset (.yaml)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="Poids de départ (pré-entraînés COCO) ou .yaml pour partir de zéro")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="0",
                         help="'0' pour GPU 0, 'cpu' pour CPU, '0,1' pour multi-GPU")
    parser.add_argument("--patience", type=int, default=30,
                         help="Early stopping : nb d'époques sans amélioration avant arrêt")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="smoke_detector")
    parser.add_argument("--resume", action="store_true",
                         help="Reprendre le dernier entraînement interrompu")
    return parser.parse_args()


def main():
    args = parse_args()

    assert Path(args.data).exists(), f"Fichier de config introuvable : {args.data}"

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # Augmentations utiles pour la fumée (texture diffuse, faible contraste)
        hsv_h=0.015,
        hsv_s=0.5,       # saturation : la fumée a peu de couleur, on limite l'agressivité
        hsv_v=0.4,       # variation de luminosité (fumée diurne / nocturne)
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,  # utile car la fumée est souvent un petit objet dans l'image
        cos_lr=True,
        optimizer="auto",
        val=True,
        plots=True,
        seed=42,
    )

    print("\nEntraînement terminé.")
    print(f"Meilleurs poids : {args.project}/{args.name}/weights/best.pt")
    print(f"Derniers poids  : {args.project}/{args.name}/weights/last.pt")

    # Export automatique en ONNX pour faciliter le déploiement
    best_weights = Path(args.project) / args.name / "weights" / "best.pt"
    if best_weights.exists():
        print("\nExport ONNX...")
        trained_model = YOLO(str(best_weights))
        trained_model.export(format="onnx", opset=12, simplify=True)
        print("Export terminé.")


if __name__ == "__main__":
    main()

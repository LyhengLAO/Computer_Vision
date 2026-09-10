"""
evaluate.py
-----------
Évalue un modèle YOLO entraîné sur le split de validation/test
et produit :
    - mAP50, mAP50-95, precision, recall, F1 par classe
    - matrice de confusion
    - courbes PR / F1
    - un rapport CSV + JSON exploitable pour le README / le suivi de projet

Usage :
    python src/evaluate.py --weights runs/train/smoke_detector/weights/best.pt \
                            --data data/smoke.yaml --split val
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Évaluation du modèle YOLO - Détection de fumée")
    parser.add_argument("--weights", type=str, required=True,
                         help="Chemin vers les poids .pt entraînés")
    parser.add_argument("--data", type=str, default="data/smoke.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25,
                         help="Seuil de confiance pour compter une détection")
    parser.add_argument("--iou", type=float, default=0.5,
                         help="Seuil IoU pour le NMS et le matching des bounding box")
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--out", type=str, default="outputs/evaluation")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        plots=True,          # génère matrice de confusion, courbes PR/F1/P/R dans runs/val/
        save_json=True,
        project=str(out_dir),
        name="run",
    )

    # ---- Résumé global ----
    summary = {
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision_mean": float(metrics.box.mp),
        "recall_mean": float(metrics.box.mr),
    }

    # ---- Détail par classe ----
    class_names = metrics.names
    per_class_rows = []
    for i, cls_idx in enumerate(metrics.ap_class_index):
        p = metrics.box.p[i]
        r = metrics.box.r[i]
        ap50 = metrics.box.ap50[i]
        ap = metrics.box.ap[i]
        f1 = 2 * p * r / (p + r + 1e-9)
        per_class_rows.append({
            "classe": class_names[int(cls_idx)],
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1": round(float(f1), 4),
            "mAP50": round(float(ap50), 4),
            "mAP50-95": round(float(ap), 4),
        })

    df = pd.DataFrame(per_class_rows)
    df.to_csv(out_dir / "metrics_per_class.csv", index=False)

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # ---- Affichage console ----
    print("\n================ RÉSULTATS D'ÉVALUATION ================")
    print(f"Split           : {args.split}")
    print(f"mAP50           : {summary['mAP50']:.4f}")
    print(f"mAP50-95        : {summary['mAP50-95']:.4f}")
    print(f"Precision (moy) : {summary['precision_mean']:.4f}")
    print(f"Recall (moy)    : {summary['recall_mean']:.4f}")
    print("\nDétail par classe :")
    print(df.to_string(index=False))
    print(f"\nMatrice de confusion + courbes PR/F1 : {out_dir}/run/")
    print(f"CSV détaillé                         : {out_dir}/metrics_per_class.csv")
    print(f"Résumé JSON                          : {out_dir}/summary.json")
    print("==========================================================")


if __name__ == "__main__":
    main()

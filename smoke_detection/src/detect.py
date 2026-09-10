"""
detect.py
---------
Inférence YOLO avec affichage/dessin des bounding boxes sur :
    - une image
    - une vidéo (fichier)
    - un flux webcam / caméra en temps réel

Usage :
    # Image
    python src/detect.py --weights best.pt --source photo.jpg

    # Vidéo
    python src/detect.py --weights best.pt --source video.mp4 --save

    # Webcam locale/USB (flux temps réel, index 0 = caméra par défaut)
    python src/detect.py --weights best.pt --source 0

    # Caméra IP de vidéosurveillance (flux RTSP en temps réel)
    python src/detect.py --weights best.pt --source rtsp://user:pass@192.168.1.10:554/stream1

    # Caméra IP HTTP/MJPEG
    python src/detect.py --weights best.pt --source http://192.168.1.10:8080/video
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Couleurs distinctes par classe (BGR)
COLORS = {
    "smoke": (180, 180, 180),   # gris
    "fire": (0, 69, 255),       # orange/rouge
}
DEFAULT_COLOR = (0, 255, 0)


def parse_args():
    parser = argparse.ArgumentParser(description="Inférence YOLO - Détection de fumée")
    parser.add_argument("--weights", type=str, required=True, help="Poids .pt entraînés")
    parser.add_argument("--source", type=str, required=True,
                         help="Chemin image/vidéo, ou index caméra (ex: 0), ou URL de flux RTSP/HTTP")
    parser.add_argument("--conf", type=float, default=0.35, help="Seuil de confiance")
    parser.add_argument("--iou", type=float, default=0.45, help="Seuil IoU pour le NMS")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--save", action="store_true", help="Sauvegarder la sortie annotée")
    parser.add_argument("--out", type=str, default="outputs/predictions")
    parser.add_argument("--show", action="store_true", default=True,
                         help="Afficher la fenêtre en direct (désactiver sur serveur headless)")
    parser.add_argument("--alert-conf", type=float, default=0.6,
                         help="Seuil de confiance déclenchant une alerte visuelle/console")
    return parser.parse_args()


def draw_detections(frame, result, alert_conf=0.6):
    """Dessine les bounding boxes + labels sur une frame et renvoie (frame, alerte_déclenchée)."""
    alert = False
    names = result.names
    if result.boxes is None:
        return frame, alert

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label_name = names[cls_id]
        color = COLORS.get(label_name, DEFAULT_COLOR)

        if conf >= alert_conf:
            alert = True
            thickness = 3
        else:
            thickness = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        label = f"{label_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

    if alert:
        cv2.putText(frame, "!!! ALERTE FUMEE/FEU DETECTEE !!!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

    return frame, alert


def is_image(path_str: str) -> bool:
    return Path(path_str).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def is_camera_index(source: str) -> bool:
    return source.isdigit()


def run_on_image(model, args):
    frame = cv2.imread(args.source)
    if frame is None:
        raise FileNotFoundError(f"Image introuvable : {args.source}")

    result = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                            device=args.device, verbose=False)[0]
    frame, alert = draw_detections(frame, result, args.alert_conf)

    if args.save:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pred_{Path(args.source).name}"
        cv2.imwrite(str(out_path), frame)
        print(f"Résultat sauvegardé : {out_path}")

    if args.show:
        cv2.imshow("Détection fumée/feu", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_on_video_or_stream(model, args):
    source = int(args.source) if is_camera_index(args.source) else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source vidéo : {args.source}")

    writer = None
    if args.save:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = out_dir / "pred_output.mp4"
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    prev_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                                device=args.device, verbose=False)[0]
        frame, alert = draw_detections(frame, result, args.alert_conf)

        # FPS en temps réel
        frame_count += 1
        now = time.time()
        fps_display = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if alert:
            print(f"[ALERTE] Frame {frame_count} - fumée/feu détecté avec forte confiance")

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("Détection fumée/feu - flux temps réel", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Vidéo annotée sauvegardée : {args.out}/pred_output.mp4")
    cv2.destroyAllWindows()


def main():
    args = parse_args()
    model = YOLO(args.weights)

    if is_image(args.source):
        run_on_image(model, args)
    else:
        run_on_video_or_stream(model, args)


if __name__ == "__main__":
    main()

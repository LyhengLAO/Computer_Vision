"""
vehicle_counter.py
-------------------
Comptage de véhicules en franchissement de ligne, sur vidéo ou flux caméra live
(webcam, RTSP, HTTP). PAS conçu pour des images fixes — le comptage nécessite un
suivi temporel (tracking) d'un flux continu.

Fonctionnement :
    1. Charge la configuration des lignes IN/OUT (créée avec line_setup.py)
    2. Lance le tracking YOLO (ByteTrack intégré à Ultralytics) sur le flux
    3. Pour chaque véhicule suivi, détecte le franchissement des lignes IN/OUT
    4. Distingue le type de véhicule (voiture, camion, moto, bus, vélo)
    5. Affiche en direct : boîtes, IDs, lignes, tableau des compteurs
    6. Met à jour en continu un CSV résumé + un log détaillé des événements

Usage :
    # Vidéo déjà enregistrée
    python src/vehicle_counter.py --source video.mp4 --lines config/lines_config.json --save

    # Caméra live (webcam locale, index 0)
    python src/vehicle_counter.py --source 0 --lines config/lines_config.json

    # Caméra IP de vidéosurveillance (RTSP)
    python src/vehicle_counter.py --source rtsp://user:pass@192.168.1.10:554/stream1 \\
                                   --lines config/lines_config.json
"""

import argparse
import json
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from counter_core import LineCounter, VEHICLE_CLASSES, draw_line_with_arrow, draw_counts_table


def parse_args():
    parser = argparse.ArgumentParser(description="Comptage de véhicules IN/OUT par franchissement de ligne")
    parser.add_argument("--source", type=str, required=True,
                         help="Fichier vidéo, index caméra (0, 1...), ou URL RTSP/HTTP")
    parser.add_argument("--lines", type=str, default="config/lines_config.json",
                         help="Fichier de config des lignes (généré par line_setup.py)")
    parser.add_argument("--weights", type=str, default="yolov8n.pt",
                         help="Poids YOLO pré-entraînés COCO (contient déjà les classes véhicules)")
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                         help="bytetrack.yaml ou botsort.yaml (trackers fournis par Ultralytics)")
    parser.add_argument("--csv-out", type=str, default="outputs/counts_summary.csv")
    parser.add_argument("--events-out", type=str, default="outputs/counts_events.csv")
    parser.add_argument("--save-video", action="store_true", help="Sauvegarder la vidéo annotée")
    parser.add_argument("--save-video-path", type=str, default="outputs/counted_output.mp4")
    parser.add_argument("--csv-update-every", type=int, default=15,
                         help="Fréquence (en frames) de mise à jour du CSV pendant le traitement")
    parser.add_argument("--show", action="store_true", default=True,
                         help="Afficher la fenêtre en direct (désactiver sur serveur headless)")
    return parser.parse_args()


def load_lines_config(path):
    with open(path, "r") as f:
        raw = json.load(f)
    # JSON stocke des listes, on reconvertit en tuples pour la logique géométrique
    def to_tuples(d):
        return {k: tuple(v) for k, v in d.items()}
    return to_tuples(raw["in_line"]), to_tuples(raw["out_line"])


def get_centroid(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def main():
    args = parse_args()

    lines_path = Path(args.lines)
    if not lines_path.exists():
        raise FileNotFoundError(
            f"Config des lignes introuvable : {lines_path}\n"
            f"Lance d'abord : python src/line_setup.py --source {args.source} --out {args.lines}"
        )
    in_line, out_line = load_lines_config(lines_path)
    counter = LineCounter(in_line, out_line)

    model = YOLO(args.weights)
    source = int(args.source) if str(args.source).isdigit() else args.source

    writer = None
    if args.save_video:
        cap_probe = cv2.VideoCapture(source)
        fps = cap_probe.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_probe.release()
        out_path = Path(args.save_video_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    frame_idx = 0
    prev_time = time.time()

    # model.track() en mode stream=True renvoie un générateur de résultats, frame par frame,
    # avec des track_id persistants (ByteTrack) — indispensable pour le comptage de franchissement.
    stream = model.track(
        source=source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        classes=list(VEHICLE_CLASSES.keys()),
        tracker=args.tracker,
        persist=True,
        stream=True,
        verbose=False,
    )

    for result in stream:
        frame = result.orig_img
        frame_idx += 1

        if result.boxes is not None and result.boxes.id is not None:
            for box, track_id, cls_id in zip(result.boxes.xyxy, result.boxes.id, result.boxes.cls):
                track_id = int(track_id)
                cls_id = int(cls_id)
                class_name = VEHICLE_CLASSES.get(cls_id, "autre")
                centroid = get_centroid(box.tolist())

                counter.update(track_id, class_name, centroid)

                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(frame, f"{class_name} #{track_id}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 3, (0, 0, 255), -1)

        draw_line_with_arrow(frame, in_line, (0, 255, 0), "IN", cv2)
        draw_line_with_arrow(frame, out_line, (0, 0, 255), "OUT", cv2)
        draw_counts_table(frame, counter, cv2)

        now = time.time()
        fps_display = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now
        cv2.putText(frame, f"FPS: {fps_display:.1f}", (15, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)

        if args.show:
            cv2.imshow("Comptage de vehicules IN/OUT", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if frame_idx % args.csv_update_every == 0:
            counter.export_csv(args.csv_out)
            counter.export_events_log(args.events_out)

    # Écriture finale garantie même si la boucle s'arrête pile entre deux mises à jour
    counter.export_csv(args.csv_out)
    counter.export_events_log(args.events_out)

    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    print("\n================ RÉSUMÉ DU COMPTAGE ================")
    for row in counter.as_rows():
        print(f"{row['type_vehicule']:10s} | entrant: {row['nombre_entrant']:4d} | "
              f"sortant: {row['nombre_sortant']:4d} | présents: {row['vehicules_presents']:4d}")
    print(f"\nCSV résumé : {args.csv_out}")
    print(f"Log détaillé des franchissements : {args.events_out}")
    if args.save_video:
        print(f"Vidéo annotée : {args.save_video_path}")
    print("======================================================")


if __name__ == "__main__":
    main()

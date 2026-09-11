"""
line_setup.py
-------------
Outil interactif pour définir les lignes de comptage IN et OUT AVANT de lancer
le comptage, en cliquant directement sur la première frame de la vidéo/caméra.

Pour chaque ligne, 3 clics sont demandés :
    1. Premier point de la ligne
    2. Second point de la ligne
    3. Point "direction" : un clic du côté où un véhicule doit se trouver APRÈS
       avoir franchi la ligne pour que ce soit compté (ex: du côté intérieur du
       site pour la ligne IN). Ça sert uniquement à définir le sens compté, ce
       point n'est pas affiché comme un objet à part sur la vidéo finale.

La configuration est sauvegardée dans un fichier JSON réutilisable (utile pour une
caméra fixe : on ne redéfinit les lignes qu'une seule fois par angle de caméra).

Usage :
    python src/line_setup.py --source video.mp4 --out config/lines_config.json
    python src/line_setup.py --source 0 --out config/lines_config.json   # webcam
"""

import argparse
import json
from pathlib import Path

import cv2

INSTRUCTIONS = [
    "Ligne IN - point 1 (cliquez)",
    "Ligne IN - point 2 (cliquez)",
    "Ligne IN - point direction : cliquez du cote 'entrant' (cote compte comme IN)",
    "Ligne OUT - point 1 (cliquez)",
    "Ligne OUT - point 2 (cliquez)",
    "Ligne OUT - point direction : cliquez du cote 'sortant' (cote compte comme OUT)",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Définition interactive des lignes de comptage IN/OUT")
    parser.add_argument("--source", type=str, required=True,
                         help="Vidéo, index webcam (0), ou URL RTSP/HTTP")
    parser.add_argument("--out", type=str, default="config/lines_config.json")
    return parser.parse_args()


def get_first_frame(source):
    src = int(source) if str(source).isdigit() else source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Impossible d'ouvrir la source : {source}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Impossible de lire une frame depuis la source.")
    return frame


def run_interactive_setup(frame):
    clicks = []
    display = frame.copy()

    def redraw():
        nonlocal display
        display = frame.copy()
        cv2.putText(display, INSTRUCTIONS[min(len(clicks), 5)], (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "'r' = recommencer | 'q' = annuler", (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

        # Ligne IN (3 premiers clics)
        if len(clicks) >= 2:
            cv2.line(display, clicks[0], clicks[1], (0, 255, 0), 2)
        if len(clicks) >= 3:
            mid = ((clicks[0][0] + clicks[1][0]) // 2, (clicks[0][1] + clicks[1][1]) // 2)
            cv2.arrowedLine(display, mid, clicks[2], (0, 255, 0), 2, tipLength=0.3)
        # Ligne OUT (3 clics suivants)
        if len(clicks) >= 5:
            cv2.line(display, clicks[3], clicks[4], (0, 0, 255), 2)
        if len(clicks) >= 6:
            mid = ((clicks[3][0] + clicks[4][0]) // 2, (clicks[3][1] + clicks[4][1]) // 2)
            cv2.arrowedLine(display, mid, clicks[5], (0, 0, 255), 2, tipLength=0.3)

        for pt in clicks:
            cv2.circle(display, pt, 5, (255, 255, 0), -1)

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 6:
            clicks.append((x, y))
            redraw()

    redraw()
    window_name = "Definition des lignes IN / OUT"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, on_click)

    while True:
        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("r"):
            clicks.clear()
            redraw()
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return None
        if len(clicks) == 6:
            cv2.putText(display, "Termine ! Appuyez sur une touche pour valider.", (15, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow(window_name, display)
            cv2.waitKey(0)
            break

    cv2.destroyAllWindows()
    return {
        "in_line": {"p1": clicks[0], "p2": clicks[1], "direction_point": clicks[2]},
        "out_line": {"p1": clicks[3], "p2": clicks[4], "direction_point": clicks[5]},
    }


def main():
    args = parse_args()
    frame = get_first_frame(args.source)
    config = run_interactive_setup(frame)

    if config is None:
        print(" Configuration annulée.")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f" Lignes sauvegardées dans : {out_path}")
    print("Relance-le uniquement si l'angle de la caméra change.")


if __name__ == "__main__":
    main()

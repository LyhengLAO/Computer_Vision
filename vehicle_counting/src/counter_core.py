"""
counter_core.py
----------------
Logique centrale, réutilisable, du comptage de véhicules par franchissement de ligne.
Utilisée à la fois par le script CLI (vehicle_counter.py) et l'application
Streamlit (app.py) pour éviter toute duplication de la logique métier.

Principe du franchissement de ligne :
    Chaque ligne (IN ou OUT) est définie par 2 points (p1, p2) qui tracent la ligne,
    plus 1 point de référence (direction_point) situé du côté "compté" de la ligne.
    À chaque frame, on calcule de quel côté de la ligne se trouve le centroïde de
    chaque véhicule suivi (tracké). Quand ce côté change de signe d'une frame à l'autre
    ET que le nouveau signe correspond au côté du direction_point, on compte un
    franchissement dans le bon sens (on ignore les franchissements dans l'autre sens,
    typique d'une route à double sens où on ne veut compter qu'un flux à la fois).
"""

import csv
import time
from collections import defaultdict
from pathlib import Path

# Classes COCO pertinentes pour le trafic routier (indices natifs YOLO/COCO)
VEHICLE_CLASSES = {
    1: "velo",
    2: "voiture",
    3: "moto",
    5: "bus",
    7: "camion",
}


def side_of_line(point, p1, p2):
    """Retourne un signe (+1/-1/0) indiquant de quel côté de la droite (p1,p2) se
    trouve `point`. Utilise le produit vectoriel 2D (cross product)."""
    px, py = point
    x1, y1 = p1
    x2, y2 = p2
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


class LineCounter:
    """Encapsule les 2 lignes (IN/OUT), le suivi des positions précédentes par
    track_id, les compteurs par classe, et l'export CSV."""

    def __init__(self, in_line: dict, out_line: dict):
        """
        in_line / out_line : dict avec les clés 'p1', 'p2', 'direction_point'
            (chacune un tuple (x, y) en pixels).
        """
        self.lines = {"in": in_line, "out": out_line}
        # Côté "compté" de chaque ligne, déterminé une fois pour toutes
        self.counted_side = {
            name: side_of_line(cfg["direction_point"], cfg["p1"], cfg["p2"])
            for name, cfg in self.lines.items()
        }
        # Dernier signe connu (par ligne) pour chaque track_id
        self._last_side = {"in": {}, "out": {}}
        # Compteurs : {classe: {"in": n, "out": n}}
        self.counts = defaultdict(lambda: {"in": 0, "out": 0})
        # IDs déjà comptés pour une ligne donnée, pour ne jamais compter 2x le même
        # objet sur la même ligne (même s'il oscille pile sur la ligne plusieurs frames)
        self._already_counted = {"in": set(), "out": set()}
        self.events = []  # historique des franchissements pour le log CSV détaillé

    def update(self, track_id: int, class_name: str, centroid: tuple) -> list:
        """À appeler à chaque frame pour un objet tracké donné.
        Retourne la liste des événements déclenchés à cette frame :
        [("in", class_name), ...] ou [] si rien ne se passe."""
        triggered = []
        for line_name, cfg in self.lines.items():
            current_side = side_of_line(centroid, cfg["p1"], cfg["p2"])
            previous_side = self._last_side[line_name].get(track_id)

            if (
                previous_side is not None
                and previous_side != current_side
                and current_side == self.counted_side[line_name]
                and track_id not in self._already_counted[line_name]
            ):
                self.counts[class_name][line_name] += 1
                self._already_counted[line_name].add(track_id)
                triggered.append((line_name, class_name))
                self.events.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "track_id": track_id,
                    "classe": class_name,
                    "direction": line_name,
                })

            self._last_side[line_name][track_id] = current_side

        return triggered

    def total(self, line_name: str) -> int:
        return sum(v[line_name] for v in self.counts.values())

    def as_rows(self):
        """Retourne les compteurs sous forme de lignes (type_vehicule, in, out, présents),
        y compris les classes jamais vues (comptées à 0) pour un CSV toujours complet.
        'vehicules_presents' = nombre_entrant - nombre_sortant : occupation courante estimée
        par type (peut être négatif si un véhicule était déjà présent avant le début du
        comptage et ressort ensuite — à interpréter avec prudence en début de session)."""
        rows = []
        seen = set(self.counts.keys())
        for cls in list(VEHICLE_CLASSES.values()):
            c = self.counts.get(cls, {"in": 0, "out": 0})
            rows.append({
                "type_vehicule": cls,
                "nombre_entrant": c["in"],
                "nombre_sortant": c["out"],
                "vehicules_presents": c["in"] - c["out"],
            })
            seen.discard(cls)
        # Sécurité : si une classe hors mapping standard apparaît, on l'ajoute aussi
        for cls in seen:
            c = self.counts[cls]
            rows.append({
                "type_vehicule": cls,
                "nombre_entrant": c["in"],
                "nombre_sortant": c["out"],
                "vehicules_presents": c["in"] - c["out"],
            })
        # Ligne de total tous types confondus, en dernier
        total_in = sum(r["nombre_entrant"] for r in rows)
        total_out = sum(r["nombre_sortant"] for r in rows)
        rows.append({
            "type_vehicule": "TOTAL",
            "nombre_entrant": total_in,
            "nombre_sortant": total_out,
            "vehicules_presents": total_in - total_out,
        })
        return rows

    def export_csv(self, path):
        """Écrit (écrase) le CSV résumé — appelé en continu pendant le traitement
        pour un suivi 'en live' du fichier."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["type_vehicule", "nombre_entrant", "nombre_sortant", "vehicules_presents"]
            )
            writer.writeheader()
            for row in self.as_rows():
                writer.writerow(row)

    def export_events_log(self, path):
        """Écrit le log détaillé (un franchissement = une ligne), utile pour
        l'audit ou pour recalculer des statistiques a posteriori (par tranche horaire, etc.)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "track_id", "classe", "direction"])
            writer.writeheader()
            for event in self.events:
                writer.writerow(event)


def draw_line_with_arrow(frame, cfg, color, label, cv2_module):
    """Dessine une ligne de comptage + une flèche indiquant le sens compté."""
    cv2 = cv2_module
    p1, p2 = tuple(map(int, cfg["p1"])), tuple(map(int, cfg["p2"]))
    dp = tuple(map(int, cfg["direction_point"]))
    cv2.line(frame, p1, p2, color, 3)
    mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
    cv2.arrowedLine(frame, mid, dp, color, 2, tipLength=0.3)
    cv2.putText(frame, label, (p1[0], p1[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return frame


def draw_counts_table(frame, counter: "LineCounter", cv2_module, origin=(15, 30)):
    """Dessine un petit tableau live des compteurs directement sur la frame."""
    cv2 = cv2_module
    x0, y0 = origin
    line_height = 24
    cv2.rectangle(frame, (x0 - 10, y0 - 25), (x0 + 260, y0 + line_height * (len(VEHICLE_CLASSES) + 1) + 5),
                  (0, 0, 0), -1)
    cv2.putText(frame, f"{'Type':10s} {'IN':>5s} {'OUT':>5s}", (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for i, cls in enumerate(VEHICLE_CLASSES.values(), start=1):
        c = counter.counts.get(cls, {"in": 0, "out": 0})
        y = y0 + i * line_height
        cv2.putText(frame, f"{cls:10s} {c['in']:>5d} {c['out']:>5d}", (x0, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame

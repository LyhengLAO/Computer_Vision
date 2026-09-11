"""
app.py
------
Déploiement du comptage de véhicules IN/OUT — uniquement vidéo/caméra (pas d'image
fixe, le comptage nécessite un flux temporel).

Deux modes :
    1. Upload d'une vidéo -> définition des lignes sur la première frame -> traitement complet
    2. Caméra IP / RTSP / USB en direct -> définition des lignes sur un instantané -> comptage live

Lancement :
    streamlit run src/app.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

from counter_core import LineCounter, VEHICLE_CLASSES, draw_line_with_arrow, draw_counts_table

st.set_page_config(page_title="Comptage de véhicules - YOLO", page_icon="🚗", layout="wide")

DEFAULT_WEIGHTS = "yolov8n.pt"  # modèle COCO pré-entraîné : connaît déjà voiture/camion/moto/bus/vélo


@st.cache_resource
def load_model(weights_path):
    return YOLO(weights_path)


def get_centroid(xyxy):
    x1, y1, x2, y2 = xyxy
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def line_definition_ui(frame, key_prefix):
    """Interface de définition des lignes IN/OUT par saisie numérique de coordonnées,
    avec aperçu visuel immédiat superposé sur la frame. Alternative robuste au clic
    souris (non natif dans Streamlit sans dépendance supplémentaire)."""
    h, w = frame.shape[:2]
    st.caption(f"Dimensions de l'image : {w}×{h} px — utilise-les comme repère pour placer les lignes.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("** Ligne IN (entrée)**")
        in_x1 = st.slider("IN — x1", 0, w, int(w * 0.1), key=f"{key_prefix}_in_x1")
        in_y1 = st.slider("IN — y1", 0, h, int(h * 0.5), key=f"{key_prefix}_in_y1")
        in_x2 = st.slider("IN — x2", 0, w, int(w * 0.45), key=f"{key_prefix}_in_x2")
        in_y2 = st.slider("IN — y2", 0, h, int(h * 0.5), key=f"{key_prefix}_in_y2")
        in_dir_side = st.radio("Côté compté comme 'entrant'", ["Au-dessus de la ligne", "En-dessous de la ligne"],
                                key=f"{key_prefix}_in_dir")
    with col2:
        st.markdown("** Ligne OUT (sortie)**")
        out_x1 = st.slider("OUT — x1", 0, w, int(w * 0.55), key=f"{key_prefix}_out_x1")
        out_y1 = st.slider("OUT — y1", 0, h, int(h * 0.5), key=f"{key_prefix}_out_y1")
        out_x2 = st.slider("OUT — x2", 0, w, int(w * 0.9), key=f"{key_prefix}_out_x2")
        out_y2 = st.slider("OUT — y2", 0, h, int(h * 0.5), key=f"{key_prefix}_out_y2")
        out_dir_side = st.radio("Côté compté comme 'sortant'", ["Au-dessus de la ligne", "En-dessous de la ligne"],
                                 key=f"{key_prefix}_out_dir")

    def direction_point(x1, y1, x2, y2, side_choice):
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        offset = -60 if side_choice == "Au-dessus de la ligne" else 60
        return (mid_x, mid_y + offset)

    in_line = {"p1": (in_x1, in_y1), "p2": (in_x2, in_y2),
               "direction_point": direction_point(in_x1, in_y1, in_x2, in_y2, in_dir_side)}
    out_line = {"p1": (out_x1, out_y1), "p2": (out_x2, out_y2),
                "direction_point": direction_point(out_x1, out_y1, out_x2, out_y2, out_dir_side)}

    preview = frame.copy()
    draw_line_with_arrow(preview, in_line, (0, 255, 0), "IN", cv2)
    draw_line_with_arrow(preview, out_line, (0, 0, 255), "OUT", cv2)
    st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="Aperçu des lignes (flèche = sens compté)")

    return in_line, out_line


st.sidebar.title(" Paramètres")
weights_input = st.sidebar.text_input("Poids du modèle", value=DEFAULT_WEIGHTS,
                                       help="yolov8n.pt (COCO) suffit : il connaît déjà voiture/camion/moto/bus/vélo")
conf_threshold = st.sidebar.slider("Seuil de confiance", 0.1, 0.9, 0.4, 0.05)
tracker_choice = st.sidebar.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])

mode = st.sidebar.radio("Mode d'entrée", ["📤 Upload vidéo", "🌐 Caméra IP / RTSP / USB en direct"])

st.title("🚗 Comptage de véhicules IN/OUT — YOLO + tracking")
st.caption(
    "Trace 2 lignes (IN et OUT) sur un flux vidéo ou caméra, compte les véhicules qui les "
    "franchissent dans le bon sens, distingue voiture/camion/moto/bus/vélo, et exporte un CSV en direct."
)

model = load_model(weights_input)

# ------------------------------------------------------------------
# MODE 1 : Upload vidéo
# ------------------------------------------------------------------
if mode == " Upload vidéo":
    uploaded_file = st.file_uploader("Importer une vidéo", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file is not None:
        tmp_path = Path("outputs") / f"upload_{int(time.time())}.mp4"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        cap = cv2.VideoCapture(str(tmp_path))
        ret, first_frame = cap.read()
        cap.release()

        if not ret:
            st.error(" Impossible de lire la vidéo importée.")
            st.stop()

        st.subheader("1 Définir les lignes IN / OUT")
        in_line, out_line = line_definition_ui(first_frame, key_prefix="upload")

        if st.button("▶ Lancer le comptage sur toute la vidéo"):
            counter = LineCounter(in_line, out_line)
            cap = cv2.VideoCapture(str(tmp_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_video_path = Path("outputs") / f"counted_{tmp_path.stem}.mp4"
            writer = cv2.VideoWriter(str(out_video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_placeholder = st.empty()
            counts_placeholder = st.empty()

            stream = model.track(
                source=str(tmp_path), conf=conf_threshold, classes=list(VEHICLE_CLASSES.keys()),
                tracker=tracker_choice, persist=True, stream=True, verbose=False,
            )

            frame_idx = 0
            for result in stream:
                frame = result.orig_img
                frame_idx += 1

                if result.boxes is not None and result.boxes.id is not None:
                    for box, track_id, cls_id in zip(result.boxes.xyxy, result.boxes.id, result.boxes.cls):
                        track_id, cls_id = int(track_id), int(cls_id)
                        class_name = VEHICLE_CLASSES.get(cls_id, "autre")
                        centroid = get_centroid(box.tolist())
                        counter.update(track_id, class_name, centroid)

                        x1, y1, x2, y2 = map(int, box.tolist())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                        cv2.putText(frame, f"{class_name} #{track_id}", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2, cv2.LINE_AA)

                draw_line_with_arrow(frame, in_line, (0, 255, 0), "IN", cv2)
                draw_line_with_arrow(frame, out_line, (0, 0, 255), "OUT", cv2)
                draw_counts_table(frame, counter, cv2)
                writer.write(frame)

                if frame_idx % 5 == 0:
                    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    status_text.text(f"Frame {frame_idx}/{total_frames}")
                    counts_placeholder.dataframe(pd.DataFrame(counter.as_rows()), hide_index=True)
                    csv_path = Path("outputs") / "counts_summary.csv"
                    counter.export_csv(csv_path)

            writer.release()
            counter.export_csv("outputs/counts_summary.csv")
            counter.export_events_log("outputs/counts_events.csv")
            progress_bar.progress(1.0)
            status_text.text("Traitement terminé ")

            st.subheader("2 Résultats")
            st.dataframe(pd.DataFrame(counter.as_rows()), hide_index=True)
            st.video(str(out_video_path))

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                with open(out_video_path, "rb") as f:
                    st.download_button("⬇ Vidéo annotée", f, file_name=out_video_path.name)
            with col_b:
                with open("outputs/counts_summary.csv", "rb") as f:
                    st.download_button("⬇ CSV résumé", f, file_name="counts_summary.csv")
            with col_c:
                with open("outputs/counts_events.csv", "rb") as f:
                    st.download_button("⬇ Log détaillé", f, file_name="counts_events.csv")

# ------------------------------------------------------------------
# MODE 2 : Caméra IP / RTSP / USB en direct
# ------------------------------------------------------------------
else:
    camera_source = st.text_input(
        "Source caméra", value="0",
        help="Index caméra locale/USB (0, 1...) ou URL RTSP/HTTP (ex: rtsp://user:pass@192.168.1.10:554/stream1)",
    )

    if "line_config_live" not in st.session_state:
        st.session_state.line_config_live = None

    if st.button("📸 Capturer un instantané pour définir les lignes"):
        source = int(camera_source) if camera_source.strip().isdigit() else camera_source.strip()
        cap = cv2.VideoCapture(source)
        ret, frame = cap.read()
        cap.release()
        if ret:
            st.session_state.snapshot = frame
        else:
            st.error(" Impossible de capturer une image depuis cette source.")

    if "snapshot" in st.session_state:
        st.subheader("1 Définir les lignes IN / OUT")
        in_line, out_line = line_definition_ui(st.session_state.snapshot, key_prefix="live")
        st.session_state.line_config_live = (in_line, out_line)

        st.subheader("2 Comptage en direct")
        run_stream = st.toggle("▶ Démarrer le comptage live", value=False, key="vehicle_run")

        frame_placeholder = st.empty()
        counts_placeholder = st.empty()

        if run_stream:
            source = int(camera_source) if camera_source.strip().isdigit() else camera_source.strip()
            in_line, out_line = st.session_state.line_config_live
            counter = LineCounter(in_line, out_line)

            stream = model.track(
                source=source, conf=conf_threshold, classes=list(VEHICLE_CLASSES.keys()),
                tracker=tracker_choice, persist=True, stream=True, verbose=False,
            )

            frame_count = 0
            for result in stream:
                if not st.session_state.get("vehicle_run", False):
                    break

                frame = result.orig_img
                frame_count += 1

                if result.boxes is not None and result.boxes.id is not None:
                    for box, track_id, cls_id in zip(result.boxes.xyxy, result.boxes.id, result.boxes.cls):
                        track_id, cls_id = int(track_id), int(cls_id)
                        class_name = VEHICLE_CLASSES.get(cls_id, "autre")
                        centroid = get_centroid(box.tolist())
                        counter.update(track_id, class_name, centroid)

                        x1, y1, x2, y2 = map(int, box.tolist())
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                        cv2.putText(frame, f"{class_name} #{track_id}", (x1, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2, cv2.LINE_AA)

                draw_line_with_arrow(frame, in_line, (0, 255, 0), "IN", cv2)
                draw_line_with_arrow(frame, out_line, (0, 0, 255), "OUT", cv2)
                draw_counts_table(frame, counter, cv2)

                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if frame_count % 15 == 0:
                    counts_placeholder.dataframe(pd.DataFrame(counter.as_rows()), hide_index=True)
                    counter.export_csv("outputs/counts_summary.csv")
                    counter.export_events_log("outputs/counts_events.csv")

            counter.export_csv("outputs/counts_summary.csv")
            counter.export_events_log("outputs/counts_events.csv")
            st.success("Flux arrêté. CSV mis à jour dans outputs/counts_summary.csv")
    else:
        st.info("Renseigne la source caméra puis clique sur 'Capturer un instantané' pour commencer.")

st.sidebar.markdown("---")
st.sidebar.caption("Comptage de véhicules par franchissement de ligne — démonstration pédagogique.")

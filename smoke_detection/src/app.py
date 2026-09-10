"""
app.py
------
Application de déploiement Streamlit pour la détection de fumée/feu.

Trois modes :
    1. Upload d'une image ou d'une vidéo -> traitement puis affichage annoté
    2. Flux caméra en temps réel via WebRTC (webcam du navigateur, y compris déploiement cloud)
    3. Flux caméra IP / RTSP / USB en temps réel (caméra de surveillance réseau ou locale, via OpenCV)

Lancement :
    streamlit run src/app.py
"""

import time
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ------------------------------------------------------------------
# Configuration générale
# ------------------------------------------------------------------
st.set_page_config(page_title="Détection de fumée - YOLO", page_icon="🔥", layout="wide")

WEIGHTS_PATH = "models/best.pt"       # à adapter selon l'emplacement des poids entraînés
COLORS = {"smoke": (180, 180, 180), "fire": (0, 69, 255)}
DEFAULT_COLOR = (0, 255, 0)

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


@st.cache_resource
def load_model(weights_path: str):
    return YOLO(weights_path)


def draw_detections(frame, result, alert_conf=0.6):
    alert = False
    names = result.names
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            label_name = names[cls_id]
            color = COLORS.get(label_name, DEFAULT_COLOR)
            if conf >= alert_conf:
                alert = True

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{label_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return frame, alert


# ------------------------------------------------------------------
# Barre latérale - paramètres
# ------------------------------------------------------------------
st.sidebar.title("⚙️ Paramètres")
weights_input = st.sidebar.text_input("Chemin des poids du modèle (.pt)", value=WEIGHTS_PATH)
conf_threshold = st.sidebar.slider("Seuil de confiance", 0.05, 0.95, 0.35, 0.05)
iou_threshold = st.sidebar.slider("Seuil IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
alert_conf = st.sidebar.slider("Seuil d'alerte", 0.1, 0.95, 0.6, 0.05)

mode = st.sidebar.radio(
    "Mode d'entrée",
    ["📤 Upload image/vidéo", "📷 Webcam navigateur (WebRTC)", "🌐 Caméra IP / RTSP / USB (OpenCV)"],
)

st.title("🔥 Détection de fumée en temps réel — YOLO")
st.caption(
    "Application de démonstration : détection de fumée et de feu à partir "
    "d'images, de vidéos importées, d'une webcam navigateur, ou d'une caméra "
    "IP/RTSP/USB en flux continu."
)

if not Path(weights_input).exists():
    st.warning(
        f"Poids introuvables à `{weights_input}`. "
        "Entraînez le modèle avec `src/train.py` ou indiquez le bon chemin dans la barre latérale."
    )
    st.stop()

model = load_model(weights_input)

# ------------------------------------------------------------------
# MODE 1 : Upload image / vidéo
# ------------------------------------------------------------------
if mode == "Upload image/vidéo":
    uploaded_file = st.file_uploader(
        "Importer une image ou une vidéo",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov", "mkv"],
    )

    if uploaded_file is not None:
        suffix = Path(uploaded_file.name).suffix.lower()
        tmp_path = Path("outputs") / f"upload_{int(time.time())}{suffix}"
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ---- Cas image ----
        if suffix in {".jpg", ".jpeg", ".png"}:
            frame = cv2.imread(str(tmp_path))
            result = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
            frame, alert = draw_detections(frame, result, alert_conf)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Image originale")
                st.image(cv2.cvtColor(cv2.imread(str(tmp_path)), cv2.COLOR_BGR2RGB))
            with col2:
                st.subheader("Détections")
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            if alert:
                st.error("Fumée / feu détecté avec forte confiance !")
            else:
                st.success("Aucune détection critique.")

            n_det = 0 if result.boxes is None else len(result.boxes)
            st.metric("Nombre d'objets détectés", n_det)

        # ---- Cas vidéo ----
        else:
            st.subheader("Traitement de la vidéo…")
            progress_bar = st.progress(0)
            status_text = st.empty()
            frame_placeholder = st.empty()
            alert_placeholder = st.empty()

            cap = cv2.VideoCapture(str(tmp_path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            out_path = Path("outputs") / f"pred_{tmp_path.stem}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

            frame_idx = 0
            any_alert = False
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                result = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
                frame, alert = draw_detections(frame, result, alert_conf)
                any_alert = any_alert or alert
                writer.write(frame)

                frame_idx += 1
                if frame_idx % 5 == 0:  # rafraîchir l'aperçu toutes les 5 frames (perf)
                    frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))
                    status_text.text(f"Frame {frame_idx}/{total_frames}")
                    if alert:
                        alert_placeholder.error("🚨 Fumée / feu détecté sur cette frame !")

            cap.release()
            writer.release()
            progress_bar.progress(1.0)
            status_text.text("Traitement terminé")

            st.subheader("Vidéo annotée")
            st.video(str(out_path))
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Télécharger la vidéo annotée", f, file_name=out_path.name)

            if any_alert:
                st.error("Au moins une frame contient une détection critique.")
            else:
                st.success("Aucune détection critique sur l'ensemble de la vidéo.")

# ------------------------------------------------------------------
# MODE 2 : Flux temps réel via webcam du navigateur (WebRTC)
# ------------------------------------------------------------------
elif mode == "📷 Webcam navigateur (WebRTC)":
    st.info(
        "Autorisez l'accès à la caméra dans votre navigateur. "
        "La détection s'exécute image par image sur le flux en direct. "
        "Fonctionne aussi une fois l'application déployée dans le cloud."
    )

    def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        result = model.predict(img, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
        img, _ = draw_detections(img, result, alert_conf)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    webrtc_streamer(
        key="smoke-detection-realtime",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# ------------------------------------------------------------------
# MODE 3 : Flux caméra IP / RTSP / USB en temps réel (OpenCV)
# ------------------------------------------------------------------
else:
    st.info(
        "Se connecte directement à une caméra physique (USB, index 0/1…) ou à une "
        "caméra IP de vidéosurveillance (URL RTSP/HTTP), sans passer par le navigateur. "
        "C'est le mode adapté à un déploiement réel sur un site (caméra fixe branchée en continu)."
    )

    col_src, col_btn = st.columns([3, 1])
    with col_src:
        camera_source = st.text_input(
            "Source caméra",
            value="0",
            help=(
                "Index de caméra locale/USB (ex: 0, 1) ou URL de flux "
                "(ex: rtsp://user:pass@192.168.1.10:554/stream1, "
                "ou http://192.168.1.10:8080/video pour une caméra IP HTTP/MJPEG)"
            ),
        )
    with col_btn:
        st.write("")  # alignement vertical
        st.write("")
        run_stream = st.toggle("▶  Démarrer le flux", value=False, key="rtsp_run")

    frame_placeholder = st.empty()
    stats_placeholder = st.empty()
    alert_placeholder = st.empty()

    if run_stream:
        source = int(camera_source) if camera_source.strip().isdigit() else camera_source.strip()
        cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            st.error(
                f"Impossible d'ouvrir la source `{camera_source}`. "
                "Vérifie l'index caméra, l'URL RTSP/HTTP, les identifiants et le réseau."
            )
        else:
            prev_time = time.time()
            # Boucle de lecture temps réel : tourne tant que le toggle "Démarrer le flux" est actif.
            # Streamlit ré-exécute ce bloc du haut vers le bas ; on relit l'état du toggle à
            # chaque itération pour permettre un arrêt propre depuis l'interface.
            while st.session_state.get("rtsp_run", False):
                ret, frame = cap.read()
                if not ret:
                    st.warning("Flux interrompu ou fin du flux.")
                    break

                result = model.predict(frame, conf=conf_threshold, iou=iou_threshold, verbose=False)[0]
                frame, alert = draw_detections(frame, result, alert_conf)

                now = time.time()
                fps_display = 1.0 / max(now - prev_time, 1e-6)
                prev_time = now

                frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                n_det = 0 if result.boxes is None else len(result.boxes)
                stats_placeholder.caption(f"FPS: {fps_display:.1f}  |  Objets détectés: {n_det}")

                if alert:
                    alert_placeholder.error("Fumée / feu détecté avec forte confiance sur le flux en direct !")
                else:
                    alert_placeholder.empty()

            cap.release()
            if not st.session_state.get("rtsp_run", False):
                st.success("Flux arrêté.")
    else:
        st.caption("Renseigne une source puis active le toggle pour démarrer la détection en direct.")

st.sidebar.markdown("---")
st.sidebar.caption("Projet de détection de fumée avec YOLO — démonstration pédagogique.")

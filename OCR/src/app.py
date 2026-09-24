"""
app.py
------
Déploiement de l'extraction de documents.

Deux modes :
    1. Upload d'une image ou d'un PDF -> extraction + aperçu annoté + téléchargement
    2. Capture via webcam (scan d'un document papier) -> extraction immédiate

Lancement :
    streamlit run src/app.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from ocr_engine import OCREngine, draw_ocr_boxes
from field_extractor import extract_fields, fields_to_highlight_set

st.set_page_config(page_title="Extraction de documents - OCR", page_icon="🧾", layout="wide")


@st.cache_resource
def load_engine(languages, gpu):
    return OCREngine(languages=languages, gpu=gpu)


def process_and_display(image, engine):
    ocr_results = engine.run(image)
    fields = extract_fields(ocr_results)
    highlight = fields_to_highlight_set(fields, ocr_results)
    annotated = draw_ocr_boxes(image, ocr_results, highlight_texts=highlight)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                  caption="Zones de texte détectées (orange = champ extrait)")
    with col2:
        st.subheader("Champs extraits")
        labels = {
            "numero_facture": "Numéro de facture",
            "date": "Date",
            "montant_total": "Montant total",
            "tva": "TVA",
            "fournisseur": "Fournisseur",
        }
        for key, label in labels.items():
            value = fields.get(key)
            if value:
                st.metric(label, value)
            else:
                st.warning(f"{label} : non détecté")
        st.caption(f"Confiance OCR moyenne : {fields['confiance_ocr_moyenne']:.0%} "
                   f"({fields['nb_blocs_texte_detectes']} blocs de texte détectés)")

    return fields


st.sidebar.title(" Paramètres")
languages = st.sidebar.multiselect("Langues OCR", ["fr", "en", "es", "de"], default=["fr", "en"])
use_gpu = st.sidebar.checkbox("Utiliser le GPU (si disponible)", value=False)

mode = st.sidebar.radio("Mode d'entrée", ["📤 Upload document", "📷 Scanner via webcam"])

st.title("🧾 Extraction automatique de documents — OCR")
st.caption(
    "Extrait numéro de facture, date, montant, TVA et fournisseur à partir d'une image "
    "ou d'un PDF (facture, ticket de caisse...). Cas d'usage : automatisation de la saisie "
    "comptable, RPA documentaire."
)

if not languages:
    st.warning("Sélectionne au moins une langue OCR dans la barre latérale.")
    st.stop()

engine = load_engine(tuple(languages), use_gpu)

# ------------------------------------------------------------------
# MODE 1 : Upload document (image ou PDF)
# ------------------------------------------------------------------
if mode == " Upload document":
    uploaded_files = st.file_uploader(
        "Importer un ou plusieurs documents", type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        all_rows = []
        for uploaded_file in uploaded_files:
            st.markdown(f"###  {uploaded_file.name}")
            suffix = Path(uploaded_file.name).suffix.lower()
            tmp_path = Path("outputs") / f"upload_{int(time.time())}_{uploaded_file.name}"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pages = OCREngine.load_as_images(tmp_path)
            for page_idx, image in enumerate(pages):
                if len(pages) > 1:
                    st.caption(f"Page {page_idx + 1}/{len(pages)}")
                fields = process_and_display(image, engine)
                all_rows.append({"fichier": uploaded_file.name, "page": page_idx + 1, **fields})

        st.markdown("---")
        st.subheader(" Résumé (tous documents)")
        results_df = pd.DataFrame(all_rows)
        st.dataframe(results_df, hide_index=True)

        csv_path = Path("outputs") / "extraction_results.csv"
        results_df.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            st.download_button(" Télécharger le CSV résumé", f, file_name="extraction_results.csv")

# ------------------------------------------------------------------
# MODE 2 : Capture webcam (scan document papier)
# ------------------------------------------------------------------
else:
    st.info(
        "Place le document face à la caméra, bien à plat et cadré, puis prends la photo. "
        "L'extraction se lance automatiquement sur l'image capturée."
    )
    camera_image = st.camera_input("Scanner un document")

    if camera_image is not None:
        file_bytes = np.frombuffer(camera_image.getvalue(), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        st.markdown("### Résultat de l'extraction")
        fields = process_and_display(image, engine)

        csv_row = pd.DataFrame([{"fichier": "capture_webcam", "page": 1, **fields}])
        csv_path = Path("outputs") / "extraction_results.csv"
        csv_row.to_csv(csv_path, index=False)
        with open(csv_path, "rb") as f:
            st.download_button(" Télécharger le résultat (CSV)", f, file_name="extraction_capture.csv")

st.sidebar.markdown("---")
st.sidebar.caption("Extraction de documents par OCR — démonstration pédagogique.")

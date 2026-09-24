"""
ocr_engine.py
-------------
Wrapper autour d'EasyOCR pour extraire le texte + les bounding boxes d'un document
(image ou PDF). Gère la conversion PDF -> image via PyMuPDF (pas de dépendance
système externe type poppler, contrairement à pdf2image).

EasyOCR est choisi plutôt que Tesseract car :
    - installation pip pure (pas de binaire système à installer séparément)
    - bon support du français par défaut
    - renvoie nativement les bounding boxes par bloc de texte détecté
"""

from pathlib import Path

import cv2
import numpy as np


class OCREngine:
    def __init__(self, languages=("fr", "en"), gpu=False):
        import easyocr  # import différé : lourd à charger, évite le coût si non utilisé
        self.reader = easyocr.Reader(list(languages), gpu=gpu)

    @staticmethod
    def load_as_images(path):
        """Charge un fichier (image ou PDF) et retourne une liste d'images OpenCV
        (BGR), une par page pour un PDF, une seule pour une image."""
        path = Path(path)
        if path.suffix.lower() == ".pdf":
            import fitz  # PyMuPDF
            doc = fitz.open(path)
            images = []
            for page in doc:
                # zoom x2 pour une meilleure résolution OCR (PDF souvent en basse résolution native)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
                if pix.n == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif pix.n == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                images.append(img)
            doc.close()
            return images
        else:
            img = cv2.imread(str(path))
            if img is None:
                raise FileNotFoundError(f"Impossible de lire l'image : {path}")
            return [img]

    def run(self, image):
        """Lance l'OCR sur une image (array OpenCV BGR).
        Retourne une liste de dicts : {'bbox': [(x,y)x4], 'text': str, 'confidence': float}."""
        results = self.reader.readtext(image)
        parsed = []
        for bbox, text, confidence in results:
            parsed.append({
                "bbox": [tuple(map(int, pt)) for pt in bbox],
                "text": text,
                "confidence": float(confidence),
            })
        return parsed

    def run_on_file(self, path):
        """OCR sur un fichier complet (image ou PDF multi-page).
        Retourne une liste de (image, résultats_ocr) — une entrée par page."""
        images = self.load_as_images(path)
        return [(img, self.run(img)) for img in images]


def draw_ocr_boxes(image, ocr_results, highlight_texts=None):
    """Dessine les bounding boxes OCR sur l'image. Les textes présents dans
    `highlight_texts` (ex: les champs extraits) sont surlignés en couleur différente."""
    highlight_texts = highlight_texts or set()
    out = image.copy()
    for item in ocr_results:
        pts = np.array(item["bbox"], dtype=np.int32)
        is_highlighted = item["text"].strip() in highlight_texts
        color = (0, 140, 255) if is_highlighted else (0, 200, 0)
        thickness = 3 if is_highlighted else 1
        cv2.polylines(out, [pts], isClosed=True, color=color, thickness=thickness)
    return out

"""
extract_document.py
--------------------
Script CLI : extrait les champs structurés d'un document (image ou PDF), ou d'un
dossier entier de documents en mode batch. Sauvegarde une image annotée (bounding
boxes OCR + champs surlignés) et exporte les résultats en CSV/JSON.

Usage :
    # Un seul document
    python src/extract_document.py --source facture.jpg --out outputs/

    # Dossier entier (batch, résultats agrégés dans un seul CSV)
    python src/extract_document.py --source data/sample_documents/ --out outputs/ --batch
"""

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd

from ocr_engine import OCREngine, draw_ocr_boxes
from field_extractor import extract_fields, fields_to_highlight_set

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".pdf"}


def parse_args():
    parser = argparse.ArgumentParser(description="Extraction de champs structurés depuis un document")
    parser.add_argument("--source", type=str, required=True,
                         help="Fichier (image/pdf) ou dossier en mode --batch")
    parser.add_argument("--out", type=str, default="outputs")
    parser.add_argument("--batch", action="store_true", help="Traiter tous les documents d'un dossier")
    parser.add_argument("--gpu", action="store_true", help="Utiliser le GPU pour l'OCR si disponible")
    parser.add_argument("--languages", type=str, default="fr,en",
                         help="Langues OCR séparées par des virgules")
    return parser.parse_args()


def process_one_document(engine, path, out_dir):
    """Traite un document (toutes pages confondues) et retourne une liste de
    résultats (un dict par page)."""
    page_results = []
    pages = engine.run_on_file(path)

    for page_idx, (image, ocr_results) in enumerate(pages):
        fields = extract_fields(ocr_results)
        highlight = fields_to_highlight_set(fields, ocr_results)

        annotated = draw_ocr_boxes(image, ocr_results, highlight_texts=highlight)
        stem = f"{Path(path).stem}_p{page_idx+1}" if len(pages) > 1 else Path(path).stem
        annotated_path = Path(out_dir) / "annotated" / f"{stem}_annotated.jpg"
        annotated_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(annotated_path), annotated)

        json_path = Path(out_dir) / "json" / f"{stem}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(fields, f, indent=2, ensure_ascii=False)

        row = {"fichier": Path(path).name, "page": page_idx + 1, **fields}
        page_results.append(row)

        print(f"  Page {page_idx+1} : {fields['nb_blocs_texte_detectes']} blocs de texte, "
              f"confiance moyenne {fields['confiance_ocr_moyenne']:.2f}")

    return page_results


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(" Chargement du moteur OCR (peut prendre quelques secondes au premier lancement)...")
    engine = OCREngine(languages=args.languages.split(","), gpu=args.gpu)

    source = Path(args.source)
    all_rows = []

    if args.batch:
        if not source.is_dir():
            raise NotADirectoryError(f"--batch attend un dossier, reçu : {source}")
        files = sorted([p for p in source.iterdir() if p.suffix.lower() in SUPPORTED_EXTS])
        print(f" {len(files)} document(s) trouvé(s) dans {source}")
        for f in files:
            print(f"\n📄 Traitement : {f.name}")
            all_rows.extend(process_one_document(engine, f, out_dir))
    else:
        if not source.exists():
            raise FileNotFoundError(f"Fichier introuvable : {source}")
        print(f" Traitement : {source.name}")
        all_rows.extend(process_one_document(engine, source, out_dir))

    csv_path = out_dir / "extraction_results.csv"
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)

    print(f"\n Extraction terminée — {len(all_rows)} page(s) traitée(s)")
    print(f"CSV résumé      : {csv_path}")
    print(f"Images annotées : {out_dir}/annotated/")
    print(f"JSON par page   : {out_dir}/json/")


if __name__ == "__main__":
    main()

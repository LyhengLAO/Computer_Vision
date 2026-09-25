"""
field_extractor.py
-------------------
Transforme le texte brut détecté par l'OCR en champs structurés exploitables :
    - numéro de facture
    - date du document
    - montant total (TTC)
    - montant de TVA
    - nom du fournisseur (heuristique)

Approche par règles (regex + heuristiques de position), volontairement simple et
transparente — voir le README, section "Aller plus loin", pour les pistes
d'amélioration (NER entraîné, LayoutLM) si la précision doit être poussée plus loin
sur des documents très hétérogènes.
"""

import re
from datetime import datetime


DATE_PATTERNS = [
    r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b",
    r"\b(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})\b",
]

INVOICE_NUMBER_PATTERNS = [
    r"factur\w*\s*(?:n[°o]?\.?)?\s*[:\-#]?\s*([A-Z0-9][A-Z0-9\-/]{3,20})",
    r"invoice\w*\s*(?:no?\.?)?\s*[:\-#]?\s*([A-Z0-9][A-Z0-9\-/]{3,20})",
    r"\bn[°o]\.?\s*[:\-#]?\s*([A-Z0-9][A-Z0-9\-/]{3,20})",
]

TOTAL_KEYWORDS = ["total ttc", "montant total", "total à payer", "net à payer", "total"]
TVA_KEYWORDS = ["tva", "vat", "tax"]
AMOUNT_PATTERN = r"\$?\s*(\d{1,5}(?:[ .]\d{3})*(?:[.,]\d{2})?)\s*(?:€|eur|\$)?"


def _normalize_text_blocks(ocr_results):
    """Reconstruit une liste de lignes (texte + bbox) triées par position verticale,
    pour raisonner ligne par ligne plutôt que sur un bloc de texte unique désordonné."""
    blocks = []
    for item in ocr_results:
        xs = [p[0] for p in item["bbox"]]
        ys = [p[1] for p in item["bbox"]]
        blocks.append({
            "text": item["text"],
            "confidence": item["confidence"],
            "x": min(xs), "y": min(ys),
            "bbox": item["bbox"],
        })
    return sorted(blocks, key=lambda b: (b["y"] // 15, b["x"]))  # regroupe approx. par ligne

def find_exact_keyword(text, keywords):
    """Retourne le premier mot-clé de `keywords` trouvé comme mot/expression exacte
    dans `text` (insensible à la casse), ou None si aucun ne correspond.
    Évite les faux positifs du type 'total' matchant dans 'subtotal'."""
    text_lower = text.lower()
    for kw in keywords:
        pattern = r"(?<!\w)" + re.escape(kw) + r"(?!\w)"
        if re.search(pattern, text_lower):
            return kw
    return None

def _find_amount_near_exact_keyword(blocks, keywords):
    """Cherche un montant sur la même ligne (ou la ligne suivante) qu'un mot-clé
    (ex: 'total ttc'), stratégie courante pour les factures/tickets.
    Exclut les nombres immédiatement suivis de '%' (ex: 'TVA 20%'), qui sont un
    taux et non un montant en euros."""
    for i, block in enumerate(blocks):
        if find_exact_keyword(block["text"], keywords) is not None:
            for candidate in blocks[i:i + 3]:
                for match in re.finditer(AMOUNT_PATTERN, candidate["text"]):
                    following = candidate["text"][match.end():match.end() + 2].strip()
                    if following.startswith("%"):
                        continue
                    return match.group(1), candidate["text"]
    return None, None


def _find_amount_near_keyword(blocks, keywords):
    """Cherche un montant sur la même ligne (ou la ligne suivante) qu'un mot-clé
    (ex: 'total ttc'), stratégie courante pour les factures/tickets.
    Exclut les nombres immédiatement suivis de '%' (ex: 'TVA 20%'), qui sont un
    taux et non un montant en euros."""
    for i, block in enumerate(blocks):
        text_lower = block["text"].lower()
        if any(kw in text_lower for kw in keywords):
            for candidate in blocks[i:i + 3]:
                for match in re.finditer(AMOUNT_PATTERN, candidate["text"]):
                    following = candidate["text"][match.end():match.end() + 2].strip()
                    if following.startswith("%"):
                        continue
                    return match.group(1), candidate["text"]
    return None, None


def extract_fields(ocr_results: list) -> dict:
    """Point d'entrée principal : prend la sortie brute d'OCREngine.run() et
    retourne un dict de champs structurés + un score de confiance moyen OCR."""
    blocks = _normalize_text_blocks(ocr_results)
    full_text = "\n".join(b["text"] for b in blocks)

    # --- Numéro de facture ---
    invoice_number = None
    for pattern in INVOICE_NUMBER_PATTERNS:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            invoice_number = match.group(1)
            break

    # --- Date ---
    date_found = None
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            date_found = match.group(1)
            break

    # --- Montant total ---
    total_raw, total_source_text = _find_amount_near_exact_keyword(blocks, TOTAL_KEYWORDS) # total_raw, total_source_text = _find_amount_near_keyword(blocks, TOTAL_KEYWORDS) selon la recherche voulait exact ou bien inclut

    # --- TVA ---
    tva_raw, tva_source_text = _find_amount_near_keyword(blocks, TVA_KEYWORDS)

    # --- Fournisseur (heuristique : le bloc de texte le plus haut dans le document,
    # généralement l'en-tête / le logo textuel de l'entreprise émettrice) ---
    vendor = None
    for block in blocks[:5]:  # ne regarde que les 5 premiers blocs (haut de page)
        # Ignore les blocs trop courts (souvent du bruit OCR) ou purement numériques
        if len(block["text"].strip()) >= 3 and not block["text"].strip().isdigit():
            vendor = block["text"].strip()
            break

    avg_confidence = sum(b["confidence"] for b in blocks) / len(blocks) if blocks else 0.0

    return {
        "numero_facture": invoice_number,
        "date": date_found,
        "montant_total": total_raw,
        "tva": tva_raw,
        "fournisseur": vendor,
        "confiance_ocr_moyenne": round(avg_confidence, 3),
        "nb_blocs_texte_detectes": len(blocks),
    }


def fields_to_highlight_set(fields: dict, ocr_results: list) -> set:
    """Retourne l'ensemble des textes OCR bruts correspondant aux champs extraits,
    pour les surligner visuellement (voir ocr_engine.draw_ocr_boxes)."""
    highlight = set()
    for key in ["numero_facture", "date"]:
        if fields.get(key):
            highlight.add(str(fields[key]))
    # Pour les montants, on ré-associe le texte source complet du bloc (ex: "125,00 €")
    # plutôt que juste le nombre extrait par regex.
    return highlight

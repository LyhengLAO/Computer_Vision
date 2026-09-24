# 🧾 Extraction automatique de documents — OCR + champs structurés

> Extraction de champs métier (numéro de facture, date, montant, TVA, fournisseur) à partir d'images ou de PDF, avec déploiement upload + scan webcam.

---

## 📌 Résumé (pour CV / portfolio)

> Pipeline d'extraction automatique de données à partir de documents (factures, tickets de caisse) : OCR multilingue (EasyOCR), reconstruction de la structure ligne par ligne, extraction de champs métier par règles (numéro de facture, date, montant TTC, TVA, fournisseur), export CSV/JSON, déploiement via application web (upload multi-documents + scan par webcam). Cas d'usage : automatisation de la saisie comptable, RPA documentaire.

**Stack** : Python · EasyOCR · PyMuPDF · OpenCV · Streamlit · Pandas

---

## 🎯 Objectif et contexte métier

La saisie manuelle de factures/tickets reste un poste de coût récurrent en comptabilité et en RPA (Robotic Process Automation) : temps humain, erreurs de ressaisie, délai de traitement. Ce projet démontre un pipeline d'**extraction automatique de champs structurés** à partir de documents scannés ou photographiés, sans dépendre d'un fournisseur cloud payant (Google Document AI, AWS Textract) — entièrement open source et exécutable localement.

**Pas d'entraînement nécessaire** : EasyOCR est utilisé pré-entraîné (zero-shot) pour la reconnaissance de texte. Tout l'effort de personnalisation porte sur les **règles d'extraction de champs** (regex + heuristiques de position), qui s'adaptent facilement à un nouveau type de document sans réentraîner de modèle.

---

## 🗂️ Structure du projet

```
document-ocr-extraction/
├── data/
│   └── sample_documents/        # Documents de test (factures, tickets...) pour le prototypage
├── models/                      # (réservé pour un futur modèle NER custom, voir "Aller plus loin")
├── notebooks/
│   └── 01_ocr_prototyping.ipynb # Prototypage interactif : sortie OCR brute, ajustement des règles d'extraction
├── outputs/
│   ├── annotated/                # Images avec bounding boxes OCR + champs surlignés
│   ├── json/                     # Un JSON de champs extraits par page
│   └── extraction_results.csv    # CSV résumé (tous documents traités)
├── src/
│   ├── ocr_engine.py             # Wrapper EasyOCR + conversion PDF→image
│   ├── field_extractor.py        # Extraction de champs structurés par règles (regex + position)
│   ├── extract_document.py       # Script CLI (document unique ou dossier en batch)
│   └── app.py                    # Application de déploiement Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Comment fonctionne l'extraction

1. **OCR** (`ocr_engine.py`) : EasyOCR détecte chaque bloc de texte avec sa bounding box et un score de confiance. Les PDF sont convertis en image par page via PyMuPDF (zoom x2 pour compenser la résolution souvent faible d'un PDF scanné).
2. **Reconstruction en lignes** (`field_extractor.py`) : les blocs de texte, détectés dans le désordre par l'OCR, sont regroupés approximativement par ligne (regroupement par position verticale) pour pouvoir raisonner "quel montant se trouve à côté de quel mot-clé".
3. **Extraction par règles** :
   - **Numéro de facture** : regex après les mots-clés "facture"/"invoice"/"n°"
   - **Date** : regex sur les formats numériques (`12/03/2026`) et textuels français (`12 mars 2026`)
   - **Montant total / TVA** : recherche d'un montant numérique à proximité des mots-clés ("total ttc", "tva"), en excluant les nombres suivis de `%` (pour ne pas confondre un taux de TVA avec un montant)
   - **Fournisseur** : heuristique — le premier bloc de texte substantiel en haut du document (généralement l'en-tête/logo textuel)
4. **Sortie** : CSV résumé (tous documents), JSON détaillé par page, image annotée (zones OCR + champs extraits surlignés en orange).

Cette approche par règles est **transparente et débogable** (on sait exactement pourquoi un champ a été extrait ou manqué) — voir la section "Aller plus loin" pour les alternatives plus robustes mais moins interprétables.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

> Le premier lancement d'EasyOCR télécharge automatiquement ses poids pré-entraînés (~100 Mo) — nécessite une connexion internet une seule fois.

---

## 🔎 Prototypage interactif (notebook)

Avant de traiter un lot de documents, `notebooks/01_ocr_prototyping.ipynb` permet de :
- visualiser la sortie OCR brute (bounding boxes + scores de confiance)
- inspecter le regroupement ligne par ligne utilisé par l'extracteur
- tester une nouvelle regex de champ directement avant de la reporter dans `field_extractor.py`

Dépose un document de test dans `data/sample_documents/` avant de lancer le notebook.

---

## ▶️ Utilisation (CLI)

```bash
# Un seul document (image ou PDF)
python src/extract_document.py --source facture.jpg --out outputs/

# Dossier entier en mode batch (CSV agrégé)
python src/extract_document.py --source data/sample_documents/ --out outputs/ --batch
```

Sorties générées : `outputs/extraction_results.csv` (résumé), `outputs/json/<nom>.json` (détail par page), `outputs/annotated/<nom>_annotated.jpg` (image avec zones de texte + champs surlignés).

### Structure du CSV résumé

| fichier | page | numero_facture | date | montant_total | tva | fournisseur | confiance_ocr_moyenne |
|---|---|---|---|---|---|---|---|
| facture_01.jpg | 1 | FA-2026-00123 | 12/03/2026 | 270,00 | 45,00 | ACME Fournitures SARL | 0.92 |

---

## 🚀 Déploiement (Streamlit)

```bash
streamlit run src/app.py
```

- **📤 Upload document(s)** : import multi-fichiers (image ou PDF), extraction affichée pour chacun (image annotée + champs), CSV résumé téléchargeable en fin de traitement.
- **📷 Scanner via webcam** : capture d'une photo du document papier directement depuis le navigateur (`st.camera_input`), extraction immédiate sur l'image capturée — utile pour un scan ponctuel sans avoir à d'abord enregistrer un fichier.

---

## ⚙️ Comment aller plus loin (non fait dans ce projet)

1. **NER entraîné** (spaCy, ou fine-tuning d'un petit transformer) à la place des regex — plus robuste face à des formats de documents très hétérogènes, mais nécessite des données annotées et perd en interprétabilité directe.
2. **Modèle de compréhension de mise en page** (LayoutLM, Donut) — comprend la structure visuelle du document (tableaux, colonnes) en plus du texte, utile pour des factures à mise en page complexe avec plusieurs lignes d'articles.
3. **Détection de régions par YOLO** (cohérent avec le reste du portfolio) — entraîner un détecteur léger pour localiser les zones d'intérêt (en-tête, tableau, total) avant l'OCR, réduisant le bruit sur les documents très chargés visuellement.
4. **Correction post-OCR** (dictionnaire métier, correcteur orthographique contextuel) pour rattraper les erreurs de reconnaissance de caractères sur des scans de mauvaise qualité.
5. **Validation croisée des montants** — vérifier que `montant_total ≈ somme des lignes + TVA`, pour détecter automatiquement une extraction probablement erronée.
6. **Support de formats spécifiques** (facture électronique Factur-X/UBL) en complément de l'OCR pour les documents déjà structurés numériquement.

---

## 📚 Stack technique

`Python` · `EasyOCR` · `PyMuPDF` · `OpenCV` · `Streamlit` · `Pandas`

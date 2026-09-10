# 🔥 Détection de fumée et de feu en temps réel — YOLO

> Pipeline complet de computer vision, du dataset au déploiement, pour la **détection précoce d'incendie** par vidéosurveillance.

---

## 📌 Résumé (portfolio)

> Conception et implémentation d'un système de détection d'objets temps réel (YOLOv8/v11) pour identifier fumée et feu sur flux vidéo, avec pipeline complet — préparation dataset, entraînement, évaluation quantitative (mAP, precision/recall), et déploiement via application web (upload vidéo + flux webcam live). Cas d'usage : détection précoce d'incendie pour vidéosurveillance industrielle/forestière, réduisant le délai entre départ de feu et alerte humaine.

**Stack** : Python · Ultralytics YOLO · PyTorch · OpenCV · Streamlit · streamlit-webrtc · ONNX

---

## 🎯 Objectif et contexte métier

Un départ de feu détecté 5 minutes plus tôt peut réduire drastiquement les dégâts matériels et le risque humain — c'est le principe de la détection précoce. Les détecteurs de fumée classiques (capteurs ioniques/optiques) sont efficaces en intérieur mais inadaptés à la surveillance de grands espaces extérieurs (forêts, entrepôts, sites industriels) où une caméra peut couvrir une zone bien plus large qu'un capteur ponctuel.

Ce projet démontre la faisabilité d'un système de **détection visuelle de fumée/feu** utilisable comme brique de vidéosurveillance intelligente :

1. **Préparation et validation du dataset** (format YOLO)
2. **Entraînement** avec hyperparamètres adaptés à la fumée (objet diffus, peu contrasté, souvent petit dans l'image)
3. **Évaluation** quantitative rigoureuse (mAP, precision, recall, F1, matrice de confusion)
4. **Inférence** avec bounding boxes sur image, vidéo, ou flux caméra
5. **Déploiement** via une application Streamlit : upload vidéo **ou** flux webcam temps réel, avec système d'alerte

**Valeur business** : réduction du délai de détection humaine, couverture continue 24/7 sans fatigue de l'observateur, coût marginal faible (réutilise des caméras existantes), architecture extensible (notification automatique, intégration SI de sécurité).

---

## 🗂️ Structure du projet

```
smoke-detection-yolo/
├── data/
│   └── smoke.yaml              # Config du dataset (classes, chemins)
├── models/                     # Poids entraînés (best.pt, last.pt)
├── notebooks/
│   ├── 00_dataset_download.ipynb              # Comparatif des 3 datasets + téléchargement interactif
│   ├── 01_eda_exploration.ipynb              # Exploration du dataset (EDA)
│   ├── 02_pretrained_model_exploration.ipynb # Capacités du modèle COCO pré-entraîné
│   ├── 03_transfer_learning_prototype.ipynb  # Prototypage du fine-tuning
│   └── 04_evaluation_prototyping.ipynb       # Évaluation détaillée / analyse d'erreurs
├── outputs/                    # Résultats d'inférence et d'évaluation
├── src/
│   ├── download_dataset.py     # Téléchargement automatique du dataset D-Fire
│   ├── check_dataset.py        # Validation du dataset avant entraînement
│   ├── train.py                # Entraînement du modèle YOLO
│   ├── evaluate.py             # Évaluation (mAP, precision, recall, matrice de confusion)
│   ├── detect.py               # Inférence CLI : image / vidéo / webcam
│   └── app.py                  # Application de déploiement Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Pourquoi YOLO pour la fumée ?

La fumée est un objet particulièrement difficile à détecter :
- **Contours flous et non rigides** (contrairement à une personne ou une voiture)
- **Faible contraste** avec le ciel ou l'arrière-plan
- **Taille très variable** (petit panache lointain vs fumée proche envahissant l'image)

YOLO (You Only Look Once) est adapté car :
- Détection **en un seul passage** → compatible temps réel (webcam, flux vidéo continu)
- Bonne détection multi-échelle (via FPN/PAN) → utile pour les petits panaches lointains
- Écosystème Ultralytics mature (export ONNX/TensorRT, augmentations configurables, API Python simple)

---

## 📦 Installation

```bash
git clone <votre-repo>
cd smoke-detection-yolo
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

Prérequis GPU (recommandé pour l'entraînement) : CUDA compatible avec PyTorch 2.x.

---

## 📊 Dataset

Format YOLO attendu :

```
data/
├── images/{train,val,test}/*.jpg
└── labels/{train,val,test}/*.txt   # class_id x_center y_center width height (normalisés)
```

### Comparatif des datasets open source disponibles

| Dataset | Taille | Classes | Diversité scènes | Licence |
|---|---|---|---|---|
| **D-Fire** ⭐ (retenu) | ~21 500 images, 26 557 boîtes | `smoke` + `fire` | Intérieur/extérieur/forêt, jour/nuit | Recherche/éducation |
| Roboflow — Wildfire Smoke (AI for Mankind) | ~2 700 images | `smoke` uniquement | Extérieur forestier (caméras fixes HPWREN) | CC BY-NC-SA 4.0 (non-commercial) |
| Roboflow — Fire and Smoke Detection (générique) | Variable (qqs milliers) | `fire` + `smoke` | Variable selon contributeur | Variable (parfois MIT) |

**D-Fire est retenu** : volume ~10x supérieur à l'alternative Roboflow, les deux classes `smoke`/`fire` (contrairement au dataset AI for Mankind qui n'a que la fumée), scènes plus diverses, et c'est le dataset de référence utilisé par la plupart des papiers cités en section Évaluation — ce qui permet de comparer directement nos résultats à la littérature. Le comparatif détaillé et le raisonnement complet sont dans `notebooks/00_dataset_download.ipynb`.

### Téléchargement automatique — D-Fire

```bash
pip install kagglehub
python src/download_dataset.py --dest data
```

Le script récupère le miroir Kaggle (déjà au format YOLO) du dataset **D-Fire** — 21 527 images, 26 557 bounding boxes (14 692 feu, 11 865 fumée), organise automatiquement le split train/val/test. Prérequis : token API Kaggle (`~/.kaggle/kaggle.json`, gratuit sur kaggle.com/settings).

Validation avant entraînement :
```bash
python src/check_dataset.py --data data
```

### 🔎 Exploration interactive (notebooks)

En complément des scripts (pensés pour des runs longs et reproductibles), le dossier
`notebooks/` propose un travail itératif, exécutable cellule par cellule :

| Notebook | Contenu |
|---|---|
| `00_dataset_download.ipynb` | Comparatif chiffré des 3 datasets open source, justification du choix, téléchargement et organisation interactifs (équivalent notebook de `download_dataset.py`) |
| `01_eda_exploration.ipynb` | Répartition des classes, taille/position des bounding boxes, contrôle qualité visuel des annotations |
| `02_pretrained_model_exploration.ipynb` | Chargement du YOLO pré-entraîné COCO : classes connues, architecture, benchmark FPS, test zero-shot, explication du transfer learning |
| `03_transfer_learning_prototype.ipynb` | Prototypage rapide du fine-tuning sur petit sous-ensemble : fine-tuning complet vs backbone gelé, comparaison des courbes de loss, sensibilité au learning rate |
| `04_evaluation_prototyping.ipynb` | Évaluation détaillée d'un modèle entraîné : métriques par classe, sensibilité au seuil de confiance, inspection visuelle des faux négatifs, comparaison de checkpoints |

```bash
pip install jupyter nbformat
jupyter notebook notebooks/
```

---

## 🏋️ Entraînement

```bash
python src/train.py --data data/smoke.yaml --model yolov8n.pt --epochs 100 --imgsz 640 --batch 16 --device 0
```

Configuration clé : augmentations `mosaic`/`copy_paste` (la fumée est souvent un petit objet), `hsv_s`/`hsv_v` modérés (peu de couleur mais luminosité variable jour/nuit), early stopping (`--patience`), export ONNX automatique en fin d'entraînement.

---

## 📈 Évaluation — méthodologie et lecture des métriques

```bash
python src/evaluate.py --weights models/best.pt --data data/smoke.yaml --split val
```

Chaque métrique répond à une question business différente. Voici ce que mesure concrètement chacune d'elles :

| Métrique | Ce qu'elle mesure | Pourquoi c'est important ici |
|---|---|---|
| **IoU** (Intersection over Union) | Le recouvrement entre la boîte prédite et la boîte réelle. Une détection n'est "correcte" que si l'IoU dépasse un seuil (souvent 0.5). | Base de calcul de toutes les autres métriques : détermine ce qu'on compte comme "bonne" détection. |
| **Precision** | Parmi toutes les alertes déclenchées, combien sont de vraies fumées/feux (`VP / (VP+FP)`). | Une precision basse = beaucoup de fausses alertes → perte de confiance des opérateurs de sécurité, coûts d'intervention inutiles. |
| **Recall** | Parmi toutes les vraies fumées/feux présents, combien ont été détectés (`VP / (VP+FN)`). | **La métrique la plus critique pour la sécurité incendie** : un recall faible = départs de feu non détectés → risque humain/matériel direct. |
| **F1-score** | Moyenne harmonique de precision et recall. | Résume le compromis entre "trop d'alertes" et "feux ratés" en un seul chiffre, utile pour comparer des modèles. |
| **mAP50** | Moyenne de la precision sur tous les niveaux de recall, à IoU=0.5 (seuil de recouvrement peu strict). | Mesure standard de la littérature : permet de comparer directement à des benchmarks publiés. |
| **mAP50-95** | Comme mAP50, mais moyenné sur des seuils IoU de 0.5 à 0.95. | Beaucoup plus strict sur la **précision de localisation** de la boîte — pertinent si on veut ensuite estimer la distance/taille du foyer. |
| **Matrice de confusion** | Détail des confusions entre `smoke`, `fire` et l'arrière-plan. | Identifie *quel type d'erreur* domine (ex: confond souvent brouillard et fumée) → oriente les priorités d'amélioration. |
| **FPS / latence d'inférence** | Nombre d'images traitées par seconde sur le hardware cible. | Détermine si le système est utilisable en flux temps réel (webcam/caméra IP) ou seulement en traitement différé. |

Sorties générées : `outputs/evaluation/summary.json`, `metrics_per_class.csv`, courbes PR/F1, matrice de confusion.

### Résultats

⚠️ **Aucun entraînement n'a été exécuté dans cet environnement** (pas de GPU disponible, dataset non téléchargé ici). Les chiffres ci-dessous ne sont donc **pas les résultats de ce modèle** — ce sont des benchmarks publiés dans la littérature scientifique, sur des datasets fumée/feu comparables, à titre de repère. Exécute `evaluate.py` après ton propre entraînement pour remplir le tableau du bas avec tes vrais résultats.

| Source | Modèle | Dataset | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|
| Rahman et al. (2025), *ScienceDirect* | YOLOv11 + Histogram Equalization | D-Fire | 0.784 | 0.703 | 0.771 | — |
| Xu et al. (2024), *IJICT* | YOLOv8 amélioré (FPN modifié) | Wildfire smoke | 0.914 | 0.877 | 0.952 | 0.674 |
| Catargiu et al. (2024), *Sensors/MDPI* | YOLOv10 | FireAndSmoke (22k images) | ~0.89 (accuracy) | — | >0.91 | — |
| Benchmark YOLOv8n baseline (non optimisé) | YOLOv8n | Fire+smoke générique | 0.712 | 0.674 | 0.625 | 0.400 |

*(Sources citées à titre de repère méthodologique — pas des résultats de ce projet.)*

**➡️ Ton tableau de résultats** (à remplir avec `outputs/evaluation/summary.json` après entraînement) :

| Split | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| Validation | - | - | - | - | - |
| Test | - | - | - | - | - |

> ⚠️ Pour ce cas d'usage sécurité, priorise toujours le **recall** : un faux négatif (fumée non détectée) est bien plus coûteux qu'une fausse alerte.

---

## 🔍 Inférence (CLI)

```bash
python src/detect.py --weights models/best.pt --source photo.jpg --save        # Image
python src/detect.py --weights models/best.pt --source video.mp4 --save        # Vidéo
python src/detect.py --weights models/best.pt --source 0                       # Webcam
```

Bounding boxes colorées par classe, alerte visuelle/console au-delà d'un seuil de confiance configurable, affichage du FPS temps réel, sauvegarde optionnelle.

---

## 🚀 Déploiement

```bash
streamlit run src/app.py
```

- **📤 Upload image/vidéo** : traitement complet avec barre de progression, comparaison avant/après, téléchargement de la vidéo annotée
- **📷 Webcam navigateur (WebRTC)** : capture via `streamlit-webrtc` (fonctionne aussi en déploiement cloud), détection frame par frame avec bounding boxes live
- **🌐 Caméra IP / RTSP / USB en temps réel** : connexion directe (OpenCV) à une caméra physique (index USB) ou à une caméra de vidéosurveillance réseau (`rtsp://...`, `http://...`) — c'est le mode adapté à un déploiement réel sur site, caméra fixe branchée en continu, indépendamment du navigateur. Bouton démarrer/arrêter, FPS et nombre de détections affichés en direct.

Les trois modes partagent les mêmes seuils configurables (confiance, IoU, alerte) dans la barre latérale.

---

## ⚙️ Comment optimiser le modèle (non fait dans ce projet)

Ce projet livre un pipeline complet et fonctionnel, mais **sans campagne d'optimisation poussée**. Voici les leviers concrets pour aller plus loin, par ordre d'impact typique :

1. **Recherche d'hyperparamètres (HPO)** — via `model.tune()` d'Ultralytics ou Optuna : optimise learning rate, momentum, poids des augmentations. Gain typique : +2 à +5 points de mAP50.
2. **Choix d'architecture / scaling** — comparer `yolov8n/s/m` (ou `yolo11`) : plus grand = plus précis mais plus lent. Arbitrage precision vs FPS selon le hardware de déploiement (edge device vs serveur GPU).
3. **Enrichissement du dataset (active learning)** — ajouter des images ciblées sur les erreurs identifiées dans la matrice de confusion (ex: brouillard/vapeur confondus avec fumée) plutôt que d'ajouter des images génériques.
4. **Prétraitement adaptatif** — égalisation d'histogramme (CLAHE/HE) pour améliorer le contraste sur scènes sombres/surexposées ; certains papiers rapportent +3 à +5 points de mAP50 sur D-Fire avec cette seule technique.
5. **Fonction de perte adaptée aux petits objets** — remplacer l'IoU loss standard par une variante (ex: MPDIoU, Focal Loss ajustée) plus sensible aux petites boîtes, fréquentes pour la fumée lointaine.
6. **Post-traitement / lissage temporel** — sur vidéo, ne déclencher une alerte qu'après *N* frames consécutives positives (réduit les faux positifs ponctuels sans modèle supplémentaire).
7. **Suivi multi-objets (tracking)** — ByteTrack/BoT-SORT pour associer les détections dans le temps, stabiliser les boîtes et réduire le bruit frame-à-frame.
8. **Quantization / pruning** — export INT8 (ONNX Runtime, TensorRT) pour déploiement sur edge device (Jetson, Raspberry Pi + accélérateur) avec perte de précision minime.
9. **Test-Time Augmentation (TTA)** — moyenne des prédictions sur plusieurs versions augmentées de l'image en inférence ; gain de precision au prix de la latence, adapté au traitement différé (pas au temps réel).

---

## 📚 Stack technique

`Python` · `Ultralytics YOLO (v8/v11)` · `PyTorch` · `OpenCV` · `Streamlit` · `streamlit-webrtc` · `ONNX`

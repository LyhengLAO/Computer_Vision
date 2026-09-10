# 🚗 Comptage de véhicules IN/OUT par franchissement de ligne — YOLO + Tracking

> Comptage en temps réel de véhicules sur vidéo ou caméra live, par franchissement de deux lignes (entrée/sortie), avec distinction du type de véhicule et export CSV en direct.

---

## 📌 Résumé (pour CV / portfolio)

> Système de comptage directionnel de trafic routier par vision par ordinateur : détection et suivi multi-objets (YOLO + ByteTrack) sur flux vidéo/caméra live, comptage par franchissement de ligne configurable (entrée vs sortie), classification par type de véhicule (voiture, camion, moto, bus, vélo), export CSV mis à jour en continu. Cas d'usage : comptage de trafic, contrôle d'accès de parking, analyse de flux pour l'urbanisme.

**Stack** : Python · Ultralytics YOLO · ByteTrack/BoT-SORT · OpenCV · Streamlit

---

## 🎯 Objectif

Contrairement à la détection simple (une boîte par objet, frame par frame), le comptage directionnel nécessite de **suivre chaque véhicule dans le temps** (tracking) pour savoir s'il a franchi une ligne, et dans quel sens. Ce projet répond à un besoin concret : compter automatiquement les véhicules **entrants** et **sortants** d'une zone (parking, route, site industriel), par type de véhicule, sans capteur physique (boucle magnétique, barrière).

⚠️ **Ce projet fonctionne uniquement sur flux vidéo ou caméra live — pas sur des images fixes.** Le comptage par franchissement de ligne n'a de sens que sur une séquence temporelle continue (le tracking a besoin de plusieurs frames pour associer un véhicule à un identifiant persistant).

**Pas d'entraînement nécessaire** : le modèle YOLO pré-entraîné sur COCO connaît déjà nativement les classes `voiture`, `camion`, `moto`, `bus`, `vélo` — ce projet réutilise directement ces poids (zero-shot), tout l'effort porte sur la logique de tracking et de comptage.

---

## 🗂️ Structure du projet

```
vehicle-counting-yolo/
├── config/
│   └── lines_config.json       # Coordonnées des lignes IN/OUT (généré par line_setup.py)
├── data/
│   └── sample_videos/          # Vidéos de test pour le prototypage (notebooks/)
├── models/                     # Poids custom éventuels (si fine-tuning futur, voir section "Aller plus loin")
├── notebooks/
│   └── 01_prototype_tracking_counting.ipynb  # Prototypage interactif : tracking, lignes, évolution des comptages
├── outputs/
│   ├── counts_summary.csv      # CSV résumé : type_vehicule, nombre_entrant, nombre_sortant
│   ├── counts_events.csv       # Log détaillé : un franchissement = une ligne (timestamp, id, classe, sens)
│   └── counted_output.mp4      # Vidéo annotée (si --save-video)
├── src/
│   ├── counter_core.py         # Logique de franchissement de ligne, compteurs, export CSV
│   ├── line_setup.py           # Outil interactif : cliquer pour tracer les lignes IN/OUT
│   ├── vehicle_counter.py      # Script CLI principal (vidéo ou caméra live)
│   └── app.py                  # Application de déploiement Streamlit
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Comment fonctionne le comptage par franchissement de ligne

1. **Tracking** : `model.track()` d'Ultralytics (ByteTrack par défaut) attribue un identifiant persistant à chaque véhicule détecté, frame après frame — indispensable pour savoir "ce véhicule à droite est le même qu'il y a 3 frames".
2. **Position relative à la ligne** : pour chaque ligne (IN, OUT), on calcule à chaque frame de quel côté se trouve le centre du véhicule (produit vectoriel 2D, positif ou négatif selon le côté).
3. **Détection de franchissement** : quand ce signe change d'une frame à l'autre (le véhicule est passé d'un côté à l'autre) **et** que le nouveau côté correspond au sens configuré comme "compté", on incrémente le compteur pour la classe du véhicule.
4. **Anti-double comptage** : chaque `track_id` n'est compté qu'une seule fois par ligne, même s'il oscille pile sur la ligne pendant plusieurs frames.
5. **Sens ignoré volontairement** : un véhicule qui franchit la ligne IN dans le mauvais sens (ex: sort par où on compte l'entrée) n'est pas compté comme sortant sur cette ligne — c'est le rôle de la ligne OUT, positionnée ailleurs, de capter ce flux.

Ce choix de conception (2 lignes séparées, chacune avec un sens dédié) correspond à un usage réel : sur une route à double sens ou une entrée de parking à 2 voies, on place typiquement une ligne sur la voie d'entrée et une autre sur la voie de sortie.

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

---

## 🔎 Prototypage interactif (notebook)

Avant de lancer `vehicle_counter.py` en conditions réelles, `notebooks/01_prototype_tracking_counting.ipynb` permet de :
- vérifier visuellement la stabilité des IDs de tracking (détecter un ID switch avant qu'il ne fausse un comptage)
- ajuster et prévisualiser les lignes IN/OUT avant de les figer dans `config/lines_config.json`
- visualiser l'**évolution du comptage cumulé dans le temps** — une courbe qui monte en rafale plutôt que par paliers nets révèle un souci de tracking, pas un vrai pic de trafic

Dépose une courte vidéo de test dans `data/sample_videos/` avant de lancer le notebook.

---

## 🖊️ Étape 1 — Définir les lignes IN/OUT

Avant de lancer un comptage, il faut définir où se trouvent les lignes (une seule fois par angle de caméra fixe) :

```bash
python src/line_setup.py --source video.mp4 --out config/lines_config.json
# ou avec une caméra live :
python src/line_setup.py --source 0 --out config/lines_config.json
```

Une fenêtre s'ouvre sur la première frame. **6 clics** sont demandés dans l'ordre :
1. Ligne IN — point 1
2. Ligne IN — point 2
3. Point "direction" pour IN : cliquer du côté qui doit compter comme **entrant**
4. Ligne OUT — point 1
5. Ligne OUT — point 2
6. Point "direction" pour OUT : cliquer du côté qui doit compter comme **sortant**

La flèche affichée indique le sens qui sera compté. `r` pour recommencer, `q` pour annuler. La config est sauvegardée en JSON et réutilisable tant que la caméra ne bouge pas.

---

## ▶️ Étape 2 — Lancer le comptage (CLI)

```bash
# Vidéo déjà enregistrée
python src/vehicle_counter.py --source video.mp4 --lines config/lines_config.json --save-video

# Caméra locale/USB en direct
python src/vehicle_counter.py --source 0 --lines config/lines_config.json

# Caméra IP de vidéosurveillance (RTSP)
python src/vehicle_counter.py --source rtsp://user:pass@192.168.1.10:554/stream1 --lines config/lines_config.json
```

Pendant l'exécution : affichage live des boîtes, IDs de tracking, lignes IN/OUT (avec flèche de sens), tableau des compteurs par type de véhicule superposé sur la vidéo, et FPS. `q` pour arrêter.

Le CSV résumé (`outputs/counts_summary.csv`) est **réécrit toutes les N frames** (`--csv-update-every`, 15 par défaut) pendant le traitement — donc consultable en direct dans un autre programme (Excel, script de monitoring...) pendant que le comptage tourne.

### Structure du CSV résumé

| type_vehicule | nombre_entrant | nombre_sortant | vehicules_presents |
|---|---|---|---|
| voiture | 42 | 38 | 4 |
| camion | 5 | 6 | -1 |
| moto | 12 | 9 | 3 |
| bus | 2 | 2 | 0 |
| velo | 7 | 5 | 2 |
| **TOTAL** | **68** | **60** | **8** |

- `nombre_entrant` / `nombre_sortant` : compteurs **cumulés** depuis le début du comptage (ne diminuent jamais)
- `vehicules_presents` = `nombre_entrant - nombre_sortant` : occupation courante estimée par type. ⚠️ Peut être négatif (comme `camion` ci-dessus) si des véhicules étaient déjà présents dans la zone avant le début du comptage et en ressortent ensuite — le comptage n'a pas de visibilité sur l'état initial de la scène. Pour un chiffre d'occupation fiable dès le départ, démarrer le comptage sur une scène vide.
- La ligne `TOTAL` agrège toutes les classes de véhicules confondues.

Un second fichier, `outputs/counts_events.csv`, journalise **chaque franchissement individuellement** (timestamp, track_id, classe, direction) — utile pour recalculer des statistiques par tranche horaire a posteriori, ou auditer un comptage douteux.

---

## 🚀 Déploiement (Streamlit)

```bash
streamlit run src/app.py
```

- **📤 Upload vidéo** : la première frame s'affiche, les lignes IN/OUT se définissent via des sliders numériques (x1,y1,x2,y2 + côté compté) avec aperçu visuel immédiat de la flèche de sens — alternative fiable au clic souris, qui ne fonctionne pas nativement dans un navigateur Streamlit. Traitement complet ensuite avec barre de progression, tableau des compteurs mis à jour en direct, puis téléchargement de la vidéo annotée + CSV résumé + log détaillé.
- **🌐 Caméra IP / RTSP / USB en direct** : capture d'un instantané pour positionner les lignes, puis bouton démarrer/arrêter pour lancer le comptage en continu, avec tableau des compteurs et CSV mis à jour en direct pendant que le flux tourne.

---

## 🚙 Classes de véhicules distinguées

| Classe COCO (id) | Libellé utilisé |
|---|---|
| bicycle (1) | `velo` |
| car (2) | `voiture` |
| motorcycle (3) | `moto` |
| bus (5) | `bus` |
| truck (7) | `camion` |

Ces classes existent nativement dans YOLO pré-entraîné COCO — aucun fine-tuning nécessaire pour ce projet.

---

## ⚙️ Comment aller plus loin (non fait dans ce projet)

1. **Validation de la précision de comptage** — comparer manuellement (comptage humain sur un échantillon de vidéo) au comptage automatique pour estimer un taux d'erreur ; les erreurs viennent généralement d'occlusions (véhicules qui se chevauchent) ou de pertes de tracking (ID switch).
2. **Tracker plus robuste** — BoT-SORT (`--tracker botsort.yaml`) gère mieux les occlusions que ByteTrack au prix d'un peu de vitesse, à tester si des doubles comptages/oublis sont observés.
3. **Filtrage par taille/vitesse** — ignorer les détections trop petites (objets lointains, bruit) ou les trajectoires incohérentes (téléportation entre frames = probable erreur de tracking).
4. **Zones multiples** — généraliser à plus de 2 lignes pour un carrefour à plusieurs entrées/sorties.
5. **Export temps réel vers une base de données / dashboard** (au lieu du CSV local) pour une supervision multi-caméras centralisée.
6. **Quantization / édge deployment** — export ONNX/TensorRT pour tourner sur caméra intelligente ou Jetson en bord de route, sans serveur central.

---

## 📚 Stack technique

`Python` · `Ultralytics YOLO` · `ByteTrack / BoT-SORT` · `OpenCV` · `Streamlit` · `Pandas`

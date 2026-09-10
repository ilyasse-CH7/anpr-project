# ANPR Maroc — Lecture automatique de plaques d'immatriculation marocaines

Système de reconnaissance automatique de plaques (*Automatic Number Plate
Recognition*) conçu pour les formats d'immatriculation marocains, alimenté par
une caméra IP en direct (flux RTSP) ou par des images fixes.

---

## 1. Objectif du projet

Lire, en temps réel et sans intervention humaine, le matricule d'un véhicule qui
passe devant une caméra, puis l'enregistrer dans une base de données consultable
via une API web.

Un matricule marocain se compose de trois champs, lus de droite à gauche sur la
plaque physique mais notés ici de gauche à droite :

```
        13456   -   ب   -   27
          │         │        │
          │         │        └── région (1 à 2 chiffres)
          │         └── lettre arabe de série
          └── numéro de série (4 à 5 chiffres)
```

Deux dispositions physiques sont gérées :

| Format      | Description                                      | Véhicules typiques      |
|-------------|--------------------------------------------------|-------------------------|
| `1_ligne`   | `série \| lettre \| région` sur une seule ligne  | Voitures                |
| `2_lignes`  | `lettre \| région` en haut, `série` en bas       | Motos, petits véhicules |

**La difficulté centrale** : ce n'est pas un problème d'OCR classique. Une plaque
marocaine mélange des **chiffres latins** et **une lettre arabe isolée**. Aucun
moteur OCR généraliste ne traite bien les deux à la fois — d'où l'architecture
hybride décrite ci-dessous.

---

## 2. Architecture du pipeline

```
┌──────────────┐
│   CAMÉRA     │  Flux RTSP (caméra IP) ou image fixe sur disque
│   RTSP / IMG │
└──────┬───────┘
       │ frame BGR (numpy)
       ▼
┌──────────────────────────────────────────────────────────┐
│  1. DÉTECTION — YOLOv8                                   │
│     anpr_maroc/detection/plate_detector.py               │
│     modèle : models/plate_detector.pt                    │
│                                                          │
│     « Où est la plaque dans l'image ? »                  │
│     Sortie : boîte englobante + score de confiance       │
└──────┬───────────────────────────────────────────────────┘
       │ crop de la plaque
       ▼
┌──────────────────────────────────────────────────────────┐
│  2. SEGMENTATION — OpenCV (vision classique)             │
│     anpr_maroc/processing/segmenter.py                   │
│                                                          │
│     a) recadre à l'intérieur du cadre de la plaque       │
│     b) détermine le layout (1 ligne / 2 lignes)          │
│     c) découpe en 3 zones : left | letter | right        │
│     Sortie : 3 imagettes indépendantes                   │
└──────┬─────────────────────┬─────────────────────────────┘
       │                     │
   left + right           letter
   (chiffres)             (lettre arabe)
       │                     │
       ▼                     ▼
┌────────────────────┐  ┌────────────────────────────────┐
│ 3a. PaddleOCR      │  │ 3b. CNN LETTRE ARABE dédié     │
│     lang='en'      │  │     ocr/arabic_letter_model.py │
│  paddleocr_reader  │  │  models/*_finetuned.pt         │
│                    │  │                                │
│ Lit les chiffres   │  │ Classifie UNE lettre isolée    │
│ latins uniquement  │  │ Sortie : (lettre, confiance)   │
└────────┬───────────┘  └────────────┬───────────────────┘
         │                           │
         └───────────┬───────────────┘
                     ▼
┌──────────────────────────────────────────────────────────┐
│  4. CONSENSUS MULTI-FRAMES  (mode caméra live seulement)  │
│     anpr_maroc/scripts/live_camera.py                    │
│                                                          │
│     Vote majoritaire champ par champ sur une fenêtre     │
│     glissante (défaut : 8 frames, 3 votes concordants).  │
│     Compense les erreurs ponctuelles d'une seule frame.  │
└──────┬───────────────────────────────────────────────────┘
       ▼
┌──────────────────────────────────────────────────────────┐
│  5. VALIDATION — regex stricte                           │
│     anpr_maroc/processing/validator.py                   │
│                                                          │
│     Rejette tout ce qui ne colle pas au format marocain. │
│     Rien de non validé n'est affiché ni stocké.          │
└──────┬───────────────────────────────────────────────────┘
       ▼
┌──────────────────────────────────────────────────────────┐
│  6. STOCKAGE — MySQL, repli SQLite                       │
│     anpr_maroc/backend/storage.py                        │
│     Table `plates` : matricule, confiance, timestamp,    │
│     chemin du crop image                                 │
└──────┬───────────────────────────────────────────────────┘
       ▼
┌──────────────────────────────────────────────────────────┐
│  7. API DE CONSULTATION — FastAPI                        │
│     anpr_maroc/backend/api/app.py                        │
│     GET /plates   GET /health   GET /images/<fichier>    │
└──────────────────────────────────────────────────────────┘
```

---

## 3. Les briques techniques, une par une

### 3.1 YOLOv8 — la détection de plaque

**À quoi ça sert.** Trouver *où* se situe la plaque dans une image qui contient
une rue, des voitures, du ciel, des passants. C'est une étape de **localisation**,
pas de lecture : YOLO ne lit rien, il dessine un rectangle.

**Pourquoi ce choix.** YOLO (*You Only Look Once*) analyse l'image entière en une
seule passe du réseau de neurones, contrairement aux approches en deux temps qui
proposent d'abord des régions candidates puis les classent. Résultat : rapide,
donc compatible avec un flux vidéo. Le modèle utilisé (`models/plate_detector.pt`)
est un YOLOv8 ré-entraîné sur un jeu de plaques (dossier `data/plate_detection/`,
format Roboflow), et non le modèle générique COCO — il ne connaît qu'une seule
classe : « plaque ».

**Analogie jury.** YOLO est le vigile qui pointe du doigt : « la plaque est là ».
Il ne sait pas la lire.

### 3.2 Segmentation OpenCV — le découpage

**À quoi ça sert.** Couper le crop de plaque en trois morceaux : chiffres de
gauche, lettre arabe du milieu, chiffres de droite.

**Pourquoi ce choix.** Ici, pas d'IA — de la vision par ordinateur classique
(seuillage, composantes connexes, projection de profils). C'est **volontaire** :
une plaque est un objet très structuré (fond clair, caractères sombres,
séparateurs verticaux). Un algorithme déterministe est plus rapide, plus
explicable et ne demande aucun jeu d'entraînement. Le segmenter détecte d'abord
le cadre de la plaque et recadre légèrement *à l'intérieur* pour écarter les
rivets, vis et bordures qui seraient sinon pris pour des caractères.

**Pourquoi c'est indispensable.** C'est ce découpage qui permet d'envoyer les
chiffres et la lettre à **deux moteurs différents**. Sans lui, l'architecture
hybride serait impossible.

### 3.3 PaddleOCR — la lecture des chiffres

**À quoi ça sert.** Transformer les imagettes `left` et `right` en chaînes de
chiffres.

**Pourquoi ce choix.** PaddleOCR est lancé en `lang='en'`, donc en mode latin
pur. Les chiffres d'une plaque marocaine *sont* des chiffres latins (0-9), et un
moteur restreint au latin ne peut pas halluciner un caractère arabe à leur place.
PaddleOCR a remplacé EasyOCR en cours de projet pour sa meilleure précision sur
des caractères petits et contrastés.

**Choix de conception important.** Le moteur OCR **ne lit jamais la lettre
arabe**, même s'il en serait techniquement capable. Le recognizer arabe de
PP-OCR s'est révélé peu fiable sur un glyphe *isolé* : il est entraîné sur des
lignes de texte arabe cursif, où le contexte des lettres voisines l'aide
énormément. Une lettre seule le déroute.

### 3.4 Le CNN lettre arabe — la brique maison

**À quoi ça sert.** Classifier l'imagette du milieu parmi les lettres arabes
admises sur une plaque.

**Pourquoi un modèle dédié plutôt qu'un OCR.** Trois raisons :

1. **Problème fermé.** Ce n'est pas de la lecture de texte, c'est de la
   *classification d'image* : une entrée, N sorties possibles. Un petit CNN
   (3 blocs convolutifs, ~100 k paramètres) suffit et tourne en quelques
   millisecondes sur CPU.
2. **Les diacritiques.** La différence entre `ا` (alif) et `أ` (alif hamza) tient
   à un petit signe détaché au-dessus. Le prétraitement du modèle
   (`preprocess_letter`) conserve explicitement toutes les composantes connexes
   internes du glyphe — hamza, madda, points — tout en éliminant les
   composantes qui touchent un bord (cadre de plaque, barre de séparation). Un
   OCR généraliste écrase cette information ; un ancien fallback par
   `matchShapes` la perdait aussi et confondait systématiquement les variantes
   d'alif.
3. **Score de confiance exploitable.** Le CNN renvoie une probabilité par classe,
   directement utilisable comme critère de rejet et comme poids dans le vote
   multi-frames.

**Analogie jury.** Plutôt que de demander à un traducteur généraliste de déchiffrer
une lettre isolée hors contexte, on forme un spécialiste qui ne fait que ça.

### 3.5 Consensus multi-frames — la fiabilisation

**À quoi ça sert.** Sur un flux vidéo, la même plaque est vue des dizaines de
fois. Plutôt que d'annoncer la lecture de la première frame, on accumule les
lectures dans une fenêtre glissante et on n'émet un matricule que lorsque au
moins 3 frames s'accordent sur chaque champ indépendamment.

**Pourquoi ce choix.** C'est ce qui autorise à abaisser le seuil de confiance du
CNN sans dégrader la précision finale : une erreur ponctuelle due à un flou de
mouvement ou à un reflet est noyée par le vote majoritaire. Un `cooldown`
(défaut 6 s) évite de ré-annoncer dix fois la même plaque immobile.

### 3.6 Validation regex — le garde-fou

**À quoi ça sert.** Vérifier que le triplet reconstitué correspond réellement à
un format marocain légal avant tout affichage ou enregistrement.

**Pourquoi ce choix.** C'est la dernière barrière : elle rend structurellement
impossible le stockage d'un matricule aberrant du type `999-ز-9999`. Deux regex
distinctes couvrent le format 1 ligne et le format 2 lignes (série plus courte
tolérée). Tout rejet est accompagné d'un motif exploitable (`lettre hors alphabet
plaque`, `longueurs invalides`…), ce qui rend le débogage possible.

### 3.7 Stockage et API

**Stockage.** Cible MySQL (table `plates`, schéma dans `config/schema_mysql.sql`).
Un repli **SQLite** automatique (`plates.db`) permet d'exécuter toute la chaîne
sans serveur MySQL — pratique pour une démo. L'API Python est identique dans les
deux cas : basculer sur MySQL ne demande que de renseigner les variables
d'environnement.

**API.** FastAPI expose les détections en JSON avec l'URL du crop associé, ce qui
permet de vérifier visuellement chaque lecture.

---

## 4. Limites connues

Ces limites sont assumées et documentées ; elles ne relèvent pas de bugs.

### 4.1 Couverture de l'alphabet du CNN lettre — limite principale

Le modèle chargé par défaut est `models/arabic_letter_classifier_real2.pt`.
Il connaît **3 lettres — `أ`, `ب`, `د` — plus une classe de rejet** (`autre`).
Toutes les valeurs ci-dessous sont **mesurées** sur un jeu de test scellé de
74 images ouvert une seule fois ; la méthode complète est dans
[`docs/arabic_letter_model.md`](docs/arabic_letter_model.md).

**Pourquoi seulement 3 lettres.** Les 434 plaques réelles disponibles ne
contiennent que 6 lettres distinctes. Après annotation manuelle des 317 crops
exploitables, seules `أ` (42), `د` (23) et `ب` (21) atteignent l'effectif
minimal fixé *avant* de regarder les données. `ه` (10) et `و` (9) sont sous le
seuil et ont été écartés — délibérément, plutôt que d'entraîner un modèle dont
on ne pourrait pas mesurer la fiabilité.

#### Précision réelle, par lettre

Sur une **zone lettre correctement cadrée** (22 images de test) :

| Lettre | Justes / total | Taux |
|---|---|---|
| `أ` | 9 / 11 | 0.818 |
| `ب` | 3 / 5 | 0.600 |
| `د` | 4 / 6 | 0.667 |
| **Total** | **16 / 22** | **0.727** |

Sur l'**ensemble du jeu de test** (74 images, dont 52 zones mal segmentées),
au seuil de production 0.45 : précision 0.382, rappel 0.650.

L'écart entre ces deux chiffres est le point important : **l'erreur dominante
n'est pas une confusion entre lettres, c'est une zone mal découpée lue comme
une lettre.** Le maillon faible mesuré est la segmentation, pas le CNN.

⚠️ **Effectifs minuscules : 5 à 11 images par lettre.** Ces taux ont des marges
d'erreur de ±20 à ±40 points et ne départagent pas les lettres entre elles.
Ils donnent un ordre de grandeur, pas une mesure fine.

À titre de comparaison, l'ancien modèle 3 classes obtenait **8/22 (0.364)** sur
les mêmes images, et se trompait sur 100 % des `ب` et des `د`.

#### Le seuil de confiance fonctionne — enfin

L'ancien modèle avait un défaut disqualifiant : **sa confiance était plus haute
quand il se trompait (0.625) que quand il avait raison (0.591)**. Un seuil
appliqué à ce modèle ne filtrait pas les erreurs, il les sélectionnait.

Le modèle actuel remet le rapport dans le bon sens : **0.541 quand il a raison,
0.492 quand il se trompe.** La marge reste faible — le seuil trie utilement
mais grossièrement.

Le seuil retenu est **0.45**, calibré sur la validation et jamais sur le test.
Le balayage donne un résultat qu'il faut énoncer tel quel : **la précision
plafonne à 0.364 quel que soit le seuil.** Monter le seuil n'achète pas de la
précision, il éteint le modèle — au-delà de 0.50, `ب` et `د` ne sont plus
jamais émis et il ne reste qu'un détecteur de `أ`.

#### Ce qui se passe sur une lettre non couverte

**C'est la limite à connaître avant toute démonstration.** Sur une plaque
portant `ه`, `و`, `ط`, `م`… le modèle ne répond pas « je ne sais pas » : il
répond la moins improbable de ses 3 lettres. Vérifié sur les images de test :

| Image | Vérité | Lu | Résultat |
|---|---|---|---|
| `3.png` | 13456-**ب**-27 | 13456-**ب**-27 | ✅ exact |
| `5.jpg` | 62407-**أ**-34 | 62407-**أ**-34 | ✅ exact |
| `8.jpg` | 60567-**ه**-6 | 60567-**د**-6 | ❌ lettre substituée |
| `cx.jpeg` | 12345-**ه**-72 | 12345-**د**-72 | ❌ lettre substituée |

Les **chiffres sont justes sur les 4** ; les deux échecs portent exactement sur
la lettre `ه`, non couverte. La substitution passe le seuil et ressort comme
une lecture valide : **le système ne signale pas cette erreur-là**.

**Atténuation en place — la lettre indéterminée.** Quand la confiance est
insuffisante, ou quand le modèle répond sa classe de rejet `autre`, le système
n'invente pas de lettre : il émet le sentinelle `?`.

```
[14:32:07] Plaque détectée  : 13456-ب-27
[14:32:19] Plaque partielle : 60567-?-6  (lettre indéterminée)
```

Le validateur distingue trois états, et non deux :

| État | `valid` | `letter_known` | Signification |
|---|---|---|---|
| Lecture complète | `True` | `True` | les trois champs sont sûrs |
| **Lecture partielle** | `False` | `False` | chiffres sûrs, lettre inconnue |
| Rejet | `False` | `True` | lecture aberrante, jetée |

Un appelant qui ne teste que `valid` ne peut pas stocker par inadvertance une
lettre incertaine. Ce filet couvre l'incertitude du modèle — il ne couvre
**pas** la substitution d'une lettre non couverte décrite ci-dessus.

En live, le seuil bas n'est jamais accepté sur une seule frame : il exige
`--min-votes` frames concordantes, sinon il est automatiquement remonté à 0.60
(cf. §3.5). Pour une démonstration où la prudence prime sur le taux de lecture :

```bash
python -m anpr_maroc.scripts.live_camera --letter-threshold 0.60
```

Deux autres modèles restent chargeables via `ANPR_ARABIC_LETTER_MODEL`, tous
deux mesurés moins bons sur le jeu de test : `..._finetuned.pt` (3 classes,
rappel macro 0.178) et `..._ahcd.pt` (28 classes, rappel macro 0.307). Ce
dernier couvre tout l'alphabet mais est entraîné sur de l'écriture
**manuscrite** quand les plaques portent des glyphes **imprimés** — décalage de
domaine classique, et il souffre du même défaut de confiance inversée
(0.473 juste / 0.624 faux).

Le validateur (`MOROCCAN_PLATE_LETTERS`) accepte volontairement un alphabet
plus large que le CNN, pour ne pas devenir le facteur limitant le jour où le
CNN sera ré-entraîné.

### 4.2 Performance

**0,6 à 0,8 FPS mesuré** en traitement complet sur CPU, soit 1,3 à 1,7 s par
frame. Deux runs de 45 s via `--stats` donnent 0.79 FPS (1 260 ms : 241 ms YOLO
+ 1 016 ms OCR/CNN) et 0.59 FPS (1 691 ms : 322 ms YOLO + 1 366 ms OCR/CNN).
L'écart entre les deux vient de la charge de la machine, pas du pipeline : il
faut donc citer la fourchette, pas un chiffre unique.

Suffisant pour un portail ou une barrière où les véhicules marquent l'arrêt,
insuffisant pour de la voie rapide. Leviers non exploités : GPU, oneDNN/MKL-DNN
(désactivé car il déclenche une `NotImplementedError` sur la version CPU de
PaddlePaddle installée), et l'échantillonnage de frames (`--interval`).

### 4.3 Autres limites

- **MySQL non connecté par défaut** : la chaîne bascule silencieusement sur
  SQLite tant que `ANPR_DB_PASSWORD` n'est pas renseigné.
- **Aucune authentification sur l'API** — à ajouter impérativement avant toute
  exposition hors réseau local.
- **Identifiants caméra présents dans l'historique git.** Le code ne contient
  plus d'URL RTSP en dur (elle vient de `ANPR_RTSP_URL`, cf. §5.1), mais les
  commits antérieurs la conservent. **Le mot de passe de la caméra doit être
  changé** ; le purger de l'historique demanderait une réécriture (`git filter-repo`).
- **Nombreux modules encore vides** (`sender/`, `acquisition/`, `utils/`,
  `dataset/`, `backend/api/auth.py`…) : squelette de l'architecture cible, non
  implémenté. Le pipeline fonctionnel ne dépend d'aucun d'eux.
- **Pas de tests automatisés réels** : `anpr_maroc/tests/` ne contient que des
  placeholders, hormis `test_detection.py`. Les scripts `scripts/test_*.py` sont
  des outils de vérification manuelle, pas une suite pytest.

---

## 5. Comment lancer le projet

### 5.1 Installation

Python **3.12** est requis (PaddlePaddle ne fournit pas de wheel pour 3.14).

```bash
git clone <url-du-depot> anpr-project
cd anpr-project

# Les modèles et images sont suivis par Git LFS
git lfs install
git lfs pull

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cp .env.example .env      # puis renseigner l'URL RTSP et, si besoin, MySQL
```

Toutes les commandes ci-dessous supposent :

```bash
source .venv/bin/activate
export PYTHONPATH=$PWD
```

### 5.2 Démo sur une image fixe — *recommandé pour la présentation*

Ne dépend ni du réseau, ni de la caméra, ni du serveur MySQL.

```bash
# Pipeline complet sur une image, avec crops de debug sauvegardés
python -m anpr_maroc.scripts.test_pipeline --image data/sample_plates/8.jpg

# Autre exemple, seuil de détection abaissé
python -m anpr_maroc.scripts.test_pipeline --image data/sample_plates/cx.jpeg --conf 0.15

# Sortie CSV compacte, une ligne par image
python -m anpr_maroc.scripts.run_pipeline --image data/sample_plates/3.png

# Lot complet en CSV
python -m anpr_maroc.scripts.run_pipeline --dir data/sample_plates

# Comparaison au ground truth + métriques du lot
# (data/sample_plates/ground_truth.json)
python -m anpr_maroc.scripts.test_paddleocr_reader \
  --image data/sample_plates/8.jpg --segment

# Idem + sauvegarde des crops de zones dans un dossier temporaire
python -m anpr_maroc.scripts.test_paddleocr_reader \
  --image data/sample_plates/cx.jpeg --segment --debug
```

Colonnes de sortie de `run_pipeline` :
`filename, left, letter, right, valid, letter_conf, raw_text`

Les images de référence et leur vérité terrain sont dans `data/sample_plates/`.
Les crops intermédiaires (`plate.png`, `overlay.png`, `left.png`, `letter.png`,
`right.png`, `separators.txt`) sont écrits dans `data/pipeline_output/`.

### 5.3 Caméra RTSP en direct

**Prérequis :** `ANPR_RTSP_URL` doit être renseignée dans `.env`. Aucune URL
n'est codée en dur — les deux scripts ci-dessous s'arrêtent avec un message
explicite si elle manque. Alternative ponctuelle : `--source <url>`.

```bash
# Vérifier d'abord que le flux répond (aucun modèle chargé, test isolé)
python -m anpr_maroc.scripts.test_rtsp_connection

# Capture live, sans fenêtre vidéo (serveur / SSH)
python -m anpr_maroc.scripts.live_camera --no-display

# Avec fenêtre vidéo et traces détaillées
python -m anpr_maroc.scripts.live_camera --verbose

# Démo bornée à 60 s avec rapport de performance à l'arrêt
python -m anpr_maroc.scripts.live_camera --no-display --duration 60 --stats

# Webcam locale au lieu du RTSP
python -m anpr_maroc.scripts.live_camera --source 0

# Sans écriture en base
python -m anpr_maroc.scripts.live_camera --no-display --no-db

# Mode prudent : seuil CNN sûr, aucune lettre incertaine acceptée (cf. §4.1)
python -m anpr_maroc.scripts.live_camera --no-display --letter-threshold 0.60

# Démo sans caméra : rejoue un clip vidéo dans la chaîne live complète
python -m anpr_maroc.scripts.live_camera --source data/sample_plates/demo_clip.mp4 \
  --no-display --duration 45 --stats
```

Le terminal n'affiche **que** les lectures confirmées par le vote :

```
[14:32:07] Plaque détectée  : 13456-ب-27
[14:32:19] Plaque partielle : 60567-?-6  (lettre indéterminée)
```

Options utiles : `--conf` (seuil YOLO), `--letter-threshold` (seuil CNN),
`--vote-window` / `--min-votes` (consensus), `--cooldown`, `--interval`,
`--save-dir`.

### 5.4 API de consultation

```bash
python -m anpr_maroc.scripts.run_server --port 8000
# équivalent :
# uvicorn anpr_maroc.backend.api.app:app --host 0.0.0.0 --port 8000
```

Documentation interactive auto-générée : <http://localhost:8000/docs>

### 5.5 Consulter les résultats stockés

```bash
# État du service et backend de stockage réellement actif (MySQL ou SQLite)
curl http://localhost:8000/health

# 20 dernières détections
curl http://localhost:8000/plates | python -m json.tool

# 5 dernières
curl "http://localhost:8000/plates?limit=5" | python -m json.tool

# Lecture directe de la base SQLite de repli
sqlite3 plates.db "SELECT id, matricule, confiance, timestamp FROM plates ORDER BY id DESC LIMIT 10;"
```

Chaque détection expose `url_image`, servie par l'API, pointant vers le crop de
la plaque dans `data/pipeline_output/live/`.

### 5.6 Ré-entraîner le CNN lettre arabe

```bash
# Extraire des crops annotés depuis le ground truth des images d'exemple
python -m anpr_maroc.scripts.extract_arabic_letter_dataset \
  --images data/sample_plates --output data/arabic_letters/raw

# Collecter et annoter des crops depuis des images ou un flux
python -m anpr_maroc.scripts.harvest_arabic_letters --help

# Entraîner
python -m anpr_maroc.scripts.train_arabic_letter_classifier \
  --data data/arabic_letters/raw \
  --output models/arabic_letter_classifier.pt \
  --epochs 30
```

Le script **refuse** un dataset de moins de 50 images par classe. L'option
`--allow-small-dataset` existe seulement pour valider le pipeline technique,
jamais pour produire un modèle exploitable.

Pour tester un checkpoint alternatif sans modifier le code :

```bash
ANPR_ARABIC_LETTER_MODEL=models/arabic_letter_classifier_ahcd.pt \
  python -m anpr_maroc.scripts.run_pipeline --image data/sample_plates/8.jpg
```

---

## 6. Organisation du dépôt

```
anpr_maroc/
├── detection/plate_detector.py     YOLOv8 — localisation de la plaque
├── processing/
│   ├── segmenter.py                Découpage en zones (OpenCV)
│   └── validator.py                Validation regex du format marocain
├── ocr/
│   ├── paddleocr_reader.py         Lecture des chiffres + orchestration
│   └── arabic_letter_model.py      CNN lettre arabe (prétraitement + inférence)
├── backend/
│   ├── storage.py                  MySQL / repli SQLite
│   └── api/app.py                  API FastAPI de consultation
└── scripts/                        Points d'entrée en ligne de commande

models/                             Poids entraînés (Git LFS)
data/sample_plates/                 Images de référence + ground_truth.json
data/pipeline_output/               Sorties, crops de debug, captures live
config/schema_mysql.sql             Schéma MySQL de la table `plates`
```

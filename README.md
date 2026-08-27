"""Module placeholder."""
## ANPR Maroc

Pipeline de lecture de plaques marocaines. La segmentation accepte trois formats
à une ligne et un format à deux lignes :

- une ligne : `série | lettre arabe | région` ;
- deux lignes : `lettre | région` en haut, `série` en bas.

Avant toute découpe interne, le segmenter recherche le contour de la plaque et
recadre légèrement à l'intérieur de son cadre afin d'écarter le bruit extérieur.
Le choix du layout est fait sur ce recadrage ; le segmenter historique à une
ligne reste inchangé.

### Vérifier les échantillons

Installez les dépendances dans l'interpréteur configuré pour le projet :

```bash
python -m pip install -r requirements.txt
```

Pour inspecter le cas deux-lignes et récupérer les crops/overlays :

```bash
python -m anpr_maroc.scripts.test_easyocr_reader \
  --image data/sample_plates/cx.jpeg --segment --debug
```

Le dossier temporaire indiqué dans la sortie contient `plate.png`, `overlay.png`,
`left.png`, `letter.png`, `right.png`, `separators.txt` et
`detection_debug.txt`. Sans `--debug`, le script compare aussi le résultat à
`data/sample_plates/ground_truth.json` et calcule les métriques du lot.

### Limites connues

La détection et le routage de layout sont couverts ici ; les chiffres sont lus
par EasyOCR. Les lettres peuvent être reconnues par un CNN dédié dès qu'un
modèle entraîné est disponible à `models/arabic_letter_classifier.pt` (ou au
chemin défini par `ANPR_ARABIC_LETTER_MODEL`). Sans ce checkpoint, EasyOCR reste
utilisé et le fallback historique par `matchShapes` n'est plus employé car il
ne distingue pas correctement les variantes d'alif.

### Classifieur arabe dédié

Extraire d'abord les crops annotés depuis le ground truth :

```bash
python -m anpr_maroc.scripts.extract_arabic_letter_dataset \
  --images data/sample_plates --output data/arabic_letters/raw
```

Ajoutez ensuite des exemples réels dans `data/arabic_letters/raw/<lettre>/`.
Chaque classe doit comporter au moins 50 crops (100 ou plus recommandé), y
compris séparément `ا`, `أ`, `إ`, `آ`, `ه` et `م`. Entraîner et exporter le
modèle :

```bash
python -m anpr_maroc.scripts.train_arabic_letter_classifier \
  --data data/arabic_letters/raw \
  --output models/arabic_letter_classifier.pt
```

Le script refuse volontairement les datasets trop petits. L'option
`--allow-small-dataset` existe seulement pour vérifier le pipeline technique,
jamais pour produire un modèle fiable.

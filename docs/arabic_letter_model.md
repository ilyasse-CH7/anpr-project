# Modèle de lettre arabe — données, entraînement, mesures

Toutes les précisions de ce document sont **mesurées**, sur des jeux figés dans
`data/arabic_letters/splits.json`. Aucune n'est une estimation.

## 1. Origine des données réelles

Les crops réels viennent de `data/plate_detection/` (434 plaques marocaines
photographiées, boîtes vérité terrain), et non de l'export
`data/arabic_letters/to_label/` d'origine, qui s'est révélé inexploitable.

### L'ancienne file d'annotation était vide de lettres

Les 502 crops de la file initiale ont été inspectés un par un : **zéro lettre
arabe isolée**. Ce sont des morceaux de carrosserie, du bitume, des zones de
padding noires, des logos et des plaques entières. Deux causes :

1. le segmenteur avait été appliqué aux **photos de voiture entières** au lieu
   des crops de plaque — la « zone du milieu » d'une photo de voiture, c'est le
   capot ;
2. 178 des 475 fichiers venaient des augmentations `flip` / `flip_vert`. Un
   glyphe arabe miroité n'est plus la lettre dont il vient.

Le tout est conservé dans `data/arabic_letters/_archive_bad_queue/`.

### La chaîne correcte

`build_letter_queue.py` : boîte vérité terrain → crop plaque →
`segment_plate_by_layout` → zone lettre. Une seule variante géométriquement
neutre par scène (`contrast`/`noise`, jamais un miroir), donc chaque plaque
réelle apparaît une fois. 434 scènes → 21 trop petites, 96 sans glyphe isolé
→ **317 crops mis en file**.

### Fuite entre les splits du dataset Roboflow

L'export découpe les **augmentations**, pas les scènes : 348 des 434 scènes ont
des variantes dans plusieurs de `train`/`valid`/`test`. Ses splits sont donc
inutilisables comme frontière d'évaluation — et toute précision déjà publiée
pour le détecteur YOLO à partir d'eux est optimiste.

Les splits de ce modèle sont reconstruits sur les **groupes de rafale** (scènes
prises à moins de 10 s d'écart = même voiture), assignés en bloc à un seul côté
de la frontière.

## 2. Répartition annotée réelle

317 crops annotés à la main, dont 211 rejets (67 %) : zones mal cadrées,
chiffres, fragments.

| Lettre | Images | Groupes (voitures) | Seuil 15-20 |
|---|---|---|---|
| أ | 42 | 31 | atteint |
| د | 23 | 15 | atteint |
| ب | 21 | 14 | atteint |
| ه | 10 | 8 | **sous le seuil** |
| و | 9 | 7 | **sous le seuil** |
| ا | 1 | 1 | **inutilisable** |
| `autre` (rejets) | 211 | 109 | — |

Les plaques civiles marocaines n'utilisent qu'une poignée de lettres : les
~22 autres lettres de l'alphabet ont **zéro exemple réel**. Un modèle réel à
28 classes n'est pas réalisable avec cette source, quelle que soit
l'augmentation appliquée.

`ه` et `و` sont écartés de l'entraînement, conformément au seuil fixé avant de
regarder les données. Conséquence à trancher : le modèle 3 classes actuel
couvre `ه`, pas le nouveau.

## 3. Entraînement (phase 2)

Deux étapes, `train_letter_classifier_v2.py` :

1. **Pré-entraînement AHCD sur les classes cibles** — 3 592 glyphes manuscrits
   mappés sur `أ`/`ب`/`د` (le `ا` d'AHCD supervise notre `أ` : AHCD ne distingue
   pas la hamza) plus 1 800 glyphes des 25 autres lettres comme `autre` de
   substitution. 6 époques, exactitude finale 0.874.
2. **Fine-tuning sur les 165 crops réels**, backbone à `lr` 1e-4 contre 2e-3
   pour la tête, poids de classe en fréquence inverse, arrêt sur le rappel
   macro de validation (meilleure époque : 16).

Augmentation appliquée au crop couleur **avant** binarisation — exposition,
contraste, flou, rotation ±7°, échelle ±8 %, translation ±3 %. Jamais de
miroir. L'appliquer après la binarisation entraînerait le modèle contre un
bruit que le pipeline réel ne produit pas.

### Décision : pré-entraînement 4 classes, pas seulement le backbone

| Variante | Rappel macro (val) | Exactitude |
|---|---|---|
| **Pré-entraînement 4 classes (`real2.pt`)** | **0.665** | 0.431 |
| Reprise du backbone seul (`real.pt`) | 0.491 | 0.328 |

**+0.174 de rappel macro.** Ne reprendre que les convolutions laisse la tête à
4 sorties partir du bruit sur 165 images ; la pré-entraîner sur les mêmes
classes lui donne un vrai point de départ. `real2.pt` est retenu comme
candidat pour la phase 3.

Deux hypothèses testées et **invalidées** — à ne pas retenter :

- deux réécritures de `preprocess_letter` visant à récupérer les glyphes
  mutilés (hamza perdue, `أ` réduit à une barre) sont toutes deux **pires** que
  l'existant : prendre la plus grande composante attrape le cadre de la plaque,
  et ne retirer que les barres laisse passer les bandes de fond. La suppression
  agressive des composantes touchant un bord, qui semblait être le bug, fait le
  travail utile. `preprocess_letter` est un optimum local.
- l'ablation sur le `AdaptiveAvgPool` a été abandonnée (contention CPU) ; la
  référence `pool=1` donne 0.509. Question restée ouverte, sans effet sur la
  décision ci-dessus.

## 4. Mesures comparatives sur la validation

58 images : `autre` 42, `أ` 8, `ب` 4, `د` 4.

| Modèle | Rappel macro | Exactitude | Confiance correct / incorrect |
|---|---|---|---|
| **reel v2 (pré-entr. 4 classes)** | **0.665** | 0.431 | 0.512 / 0.482 |
| reel v1 (backbone ahcd) | 0.491 | 0.328 | 0.358 / 0.317 |
| 3 classes (défaut actuel) | 0.262 | 0.172 | 0.590 / **0.650** |
| ahcd (28 classes) | 0.231 | 0.448 | 0.436 / 0.629 |

Rappel et précision par lettre (`reel v2`) :

| Classe | n | Rappel | Précision |
|---|---|---|---|
| `أ` | 8 | 0.875 | 0.292 |
| `ب` | 4 | 0.500 | 0.222 |
| `د` | 4 | 1.000 | 0.308 |
| `autre` | 42 | 0.286 | 1.000 |

### Le modèle par défaut actuel ne discrimine pas

Le modèle 3 classes obtient un rappel de 1.000 sur `أ` — pour une précision de
**0.205** : il répond `أ` ou `ه` à presque tout, dont 27 des 42 crops `autre`,
et se trompe sur 100 % des `ب` et des `د`. Sa bonne tenue en démonstration
venait de ce qu'on ne lui présentait que des `أ`. Le rappel seul masque
entièrement ce comportement ; c'est pourquoi l'évaluation reporte les deux.

Plus grave : **sa confiance est plus élevée quand il se trompe (0.650) que
quand il a raison (0.590)**. Le seuil de confiance décrit en §4.1 du README est
donc inopérant sur ce modèle, pas seulement imparfait. `reel v2` inverse ce
rapport (0.512 contre 0.482), mais de peu.

### Limites à garder en vue pour la phase 3

- **Précision faible sur les lettres** (~0.3) : `reel v2` sur-prédit les
  lettres, 16 crops `autre` sont lus `أ`. En production cela produit des
  lectures inventées sur des zones mal segmentées.
- **Effectifs minuscules** : 4 à 8 images par lettre en validation. Ces
  rappels ont des marges d'erreur de l'ordre de ±20 points et ne doivent pas
  être présentés comme des mesures fines.
- **Régression de couverture** : l'ancien modèle gère `ه`, `reel v2` non.
  À arbitrer en phase 4.

## 5. Reproduire

```bash
python -m anpr_maroc.scripts.build_letter_queue         # file d'annotation
python data/arabic_letters/annotator/serve_annotator.py # annotation clavier
python -m anpr_maroc.scripts.split_letter_dataset       # split figé (une fois)
python -m anpr_maroc.scripts.train_letter_classifier_v2 \
    --output models/arabic_letter_classifier_real2.pt --ahcd-epochs 6
python -m anpr_maroc.scripts.evaluate_letter_models --split val
```

**Le jeu de test (74 images : `أ` 11, `ب` 5, `د` 6, `autre` 52) n'a pas encore
été ouvert.** Il est réservé à la phase 3 et ne doit servir qu'une fois, pour la
décision finale.

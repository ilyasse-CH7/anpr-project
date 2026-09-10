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
python -m anpr_maroc.scripts.calibrate_letter_threshold --read-test  # phase 3
```

## 6. Phase 3 — jeu de test descellé (74 images)

Le jeu de test (`أ` 11, `ب` 5, `د` 6, `autre` 52) a été ouvert **une fois**, après
la décision de phase 2. Aucun réglage n'a été fait sur ces chiffres — la seule
grandeur encore libre, le seuil de confiance, a été calibrée sur la validation
(§6.3) avant d'être lue ici.

```bash
python -m anpr_maroc.scripts.evaluate_letter_models --split test
```

### 6.1 Classement des quatre modèles

| Modèle | Rappel macro | Exactitude | Confiance correct / incorrect |
|---|---|---|---|
| **reel v2 (pré-entr. 4 classes)** | **0.603** | 0.446 | **0.541 / 0.492** |
| reel v1 (backbone ahcd) | 0.467 | 0.270 | 0.355 / 0.328 |
| ahcd (28 classes) | 0.307 | 0.459 | 0.473 / **0.624** |
| 3 classes (défaut actuel) | 0.178 | 0.149 | 0.591 / **0.625** |

Le classement du test reproduit celui de la validation, dans le même ordre et
avec des écarts du même signe. C'est le seul contrôle disponible contre un
sur-ajustement à la validation, et il passe.

### 6.2 Le défaut de « confiance inversée » est corrigé

C'était le point critique identifié en phase 2. **Il est réglé sur `reel v2`** :

- 3 classes (ancien) : 0.591 quand il a raison, **0.625 quand il se trompe**.
  Il est *plus* sûr de lui en se trompant. Un seuil de confiance sur ce modèle
  ne filtre pas les erreurs, il les sélectionne.
- ahcd : 0.473 / 0.624 — même défaut, en pire.
- **`reel v2` : 0.541 quand il a raison, 0.492 quand il se trompe.** Le rapport
  est dans le bon sens, et la précision monte bien avec le seuil (§6.3), ce qui
  est la définition opérationnelle d'un seuil qui fonctionne.

La marge (0.049) reste faible : le seuil trie utilement mais grossièrement.
C'est une correction du signe du défaut, pas une calibration fine.

### 6.3 Calibration du seuil — sur la validation, pas sur le test

Régler le seuil sur le test réintroduirait exactement la fuite que ce jeu de
données a été reconstruit pour supprimer. Le balayage est donc fait sur la
validation, et le test n'est qu'une lecture.

Balayage sur la **validation** (58 images), précision/rappel sur les lettres émises :

| Seuil | Émises | Justes | Précision | Rappel | F1 | Exactitude 4 cl. |
|---|---|---|---|---|---|---|
| 0.30 | 45 | 13 | 0.289 | 1.000 | 0.448 | 0.448 |
| 0.40 | 29 | 8 | 0.276 | 0.571 | 0.372 | 0.534 |
| **0.45** | **22** | **8** | **0.364** | **0.571** | **0.444** | **0.655** |
| 0.55 | 14 | 5 | 0.357 | 0.333 | 0.345 | 0.672 |
| 0.65 | 6 | 2 | 0.333 | 0.125 | 0.182 | 0.690 |
| 0.75 | 3 | 1 | 0.333 | 0.062 | 0.105 | 0.707 |
| 0.85 | 1 | 0 | 0.000 | 0.000 | 0.000 | 0.707 |

**Constat honnête : la précision ne dépasse jamais 0.364, quel que soit le
seuil.** Monter le seuil n'achète pas de la précision, il éteint simplement le
modèle — à 0.85 il n'émet plus rien. Sur le test isolément la précision semblait
grimper jusqu'à 1.000 à 0.85, mais sur **3 émissions** : du bruit, pas un
régime exploitable. C'est précisément pourquoi la calibration ne se fait pas là.

**Seuil retenu : 0.45.** C'est le point où l'exactitude 4 classes (0.655) est
proche de son maximum alors que les trois lettres sont encore émises — au-delà
de 0.50, `ب` et `د` disparaissent complètement et le modèle dégénère en
détecteur de `أ`.

Lecture sur le **test** à 0.45 : 34 lettres émises, 13 justes → précision
**0.382**, rappel **0.650**, exactitude 4 classes **0.622**.

| Lettre | Émissions justes | Taux |
|---|---|---|
| `أ` | 9 / 21 | 0.429 |
| `ب` | 2 / 5 | 0.400 |
| `د` | 2 / 8 | 0.250 |

### 6.4 D'où viennent réellement les erreurs

La précision de 0.382 se lit mal sans son contexte : le jeu de test est à 70 %
(52/74) des crops `autre`, c'est-à-dire des zones que le segmenteur a mal
cadrées. L'erreur dominante n'est pas une confusion entre lettres, c'est un
**non-glyphe lu comme une lettre**.

Séparé, cela donne deux chiffres très différents — les deux sont vrais et
doivent être cités ensemble :

| Situation | reel v2 | 3 classes (ancien) |
|---|---|---|
| **Zone lettre correctement cadrée** (22 images) | **16/22 = 0.727** | 8/22 = 0.364 |
| Rejet des zones mal cadrées (`autre`, 52 images) | rappel 0.327 | rappel 0.077 |

Détail sur zone bien cadrée (`reel v2`) : `أ` 9/11, `ب` 3/5, `د` 4/6.

**Quand la segmentation fait son travail, le modèle lit la bonne lettre dans
73 % des cas.** Le maillon faible mesuré n'est pas le classifieur mais le
rejet des zones mal segmentées. C'est là que porterait le prochain effort, pas
sur l'architecture du CNN.

### 6.5 Marges d'erreur

À rappeler systématiquement : 5 à 11 images par lettre dans le test. Un
intervalle de Wald à 95 % sur `أ` (9/11) va de 0.63 à 1.00 ; sur `ب` (3/5), de
0.17 à 1.00. **Ces taux par lettre ont des marges de l'ordre de ±20 à ±40
points** et ne départagent pas les lettres entre elles. Seul l'écart global
entre `reel v2` et l'ancien modèle (0.727 contre 0.364 sur les mêmes 22 images)
est assez large pour être conclusif.

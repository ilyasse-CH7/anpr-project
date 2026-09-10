# Démonstration — ANPR Maroc

Trois commandes, testées telles quelles le 2026-09-10 sur ce dépôt.
Chacune est indépendante des deux autres.

**Préalable commun à toutes les commandes** (une seule fois par terminal) :

```bash
cd ~/PycharmProjects/anpr-project
source .venv/bin/activate
export PYTHONPATH=$PWD
```

---

## A. Démo SANS caméra — filet de sécurité

Ne dépend d'aucun réseau ni d'aucun matériel. **C'est la démo de repli le jour J.**

### A.1 — Lecture sur images fixes

```bash
python -m anpr_maroc.scripts.run_pipeline --dir data/sample_plates --segment
```

Sortie réelle (extrait ; le chargement PaddleOCR affiche quelques lignes avant) :

```
filename,left,letter,right,valid,letter_conf,raw_text
0.jpeg,23242,أ,55,1,0.882,23242 أ 55
3.png,13456,ب,27,1,0.454,13456 ب 27
5.jpg,62407,أ,34,1,0.459,62407 أ 34
8.jpg,60567,د,6,1,0.611,60567 د 6
cc.png,1345,ب,27,1,0.491,1345 ب 27
cx.jpeg,12345,د,72,1,0.750,12345 د 72
cxv.jpg,,?,,0,0.589, ?
n.jpeg,23242,أ,55,1,0.882,23242 أ 55
...
```

À montrer : `3.png` et `5.jpg` sont **exacts sur les trois champs**. Les lignes
à `?` sont le refus assumé de deviner (§4.1 du README), pas un plantage.

### A.2 — Chaîne live complète, sans caméra

Rejoue un clip vidéo dans **exactement** le même code que la caméra : YOLO,
segmentation, OCR/CNN, vote multi-frames, validation, écriture en base.

```bash
python -m anpr_maroc.scripts.live_camera \
  --source data/sample_plates/demo_clip.mp4 \
  --no-display --duration 45 --stats
```

Sortie réelle :

```
[03:25:12] Plaque détectée : 23242-أ-55
[03:25:18] Plaque détectée : 23242-أ-55
[03:25:25] Plaque détectée : 23242-أ-55
...
--- Mesure de performance ---
Durée du run              : 45.9 s
Frames traitées           : 36 (dont 36 avec plaque détectée)
Temps moyen / frame       : 1259.8 ms  ->  0.79 FPS
  dont détection YOLO     : 241.0 ms
  dont OCR+CNN (si plaque): 1015.5 ms
```

> Le FPS varie avec la charge de la machine : deux runs successifs ont donné
> 0.79 et 0.59 FPS. **Annoncer « entre 0,6 et 0,8 image par seconde »**, pas un
> chiffre unique qui serait démenti par le run fait devant le jury.

C'est **la** commande de repli : visuellement identique à la démo caméra.
Retirer `--no-display` pour afficher la fenêtre vidéo avec les boîtes YOLO.

---

## B. Démo AVEC caméra RTSP en direct

### B.1 — Vérifier d'abord que la caméra répond (5 s, à faire avant de présenter)

```bash
python -m anpr_maroc.scripts.test_rtsp_connection
```

Si la caméra est injoignable, la sortie est explicite et sans traceback :

```
--- Trying URL brute (mot de passe avec '@' litteral): rtsp://***:***@192.168.100.55:554/... ---
ECHEC: cannot open stream (backend=aucun (le flux n'a jamais ete ouvert)); likely
causes: host unreachable (mauvais reseau/VLAN), timeout, RTSP 401 (auth), ...

URL brute en echec, tentative avec mot de passe encode (%40)...
ECHEC FINAL: aucune des deux URLs n'a fonctionne.
```

→ Dans ce cas, **basculer immédiatement sur A.2**. Ne pas déboguer devant le jury.

### B.2 — Lancer la démo caméra

```bash
python -m anpr_maroc.scripts.live_camera --duration 120 --stats
```

L'URL vient de `ANPR_RTSP_URL` dans `.env` (aucune URL en dur dans le code).
Variantes utiles :

```bash
# Sans fenêtre vidéo (terminal seul, si l'affichage pose problème)
python -m anpr_maroc.scripts.live_camera --no-display --duration 120 --stats

# Mode prudent : aucune lettre incertaine acceptée
python -m anpr_maroc.scripts.live_camera --letter-threshold 0.60

# Webcam du portable au lieu de la caméra IP
python -m anpr_maroc.scripts.live_camera --source 0
```

> ⚠️ **État au 2026-09-10** : la caméra `192.168.100.55` n'était **pas joignable**
> depuis la machine de développement (`No route to host` — mauvais réseau, la
> machine n'était pas sur le VLAN de la caméra). La commande B.2 n'a donc pas pu
> être validée en conditions réelles ce soir. Le reste de la chaîne, lui, est
> vérifié de bout en bout par A.2, qui emprunte le même code.
> **À refaire une fois sur le bon réseau : lancer B.1 avant la présentation.**

---

## C. Consultation des résultats stockés (API / base)

Dans un **second terminal** (le préalable commun s'applique aussi) :

```bash
python -m anpr_maroc.scripts.run_server --port 8000
```

Puis, dans un troisième terminal, ou depuis un navigateur :

```bash
curl -s localhost:8000/health
curl -s "localhost:8000/plates?limit=5" | python -m json.tool
```

Sortie réelle :

```json
{"status":"ok","stockage":"SQLite plates.db (repli, MySQL indisponible : ...)"}
```

```json
{
    "count": 5,
    "plates": [
        {
            "id": 19,
            "matricule": "23242-أ-55",
            "confiance": 0.9589,
            "timestamp": "2026-09-10 03:25:46",
            "url_image": "http://localhost:8000/images/plate_20260910T032546420664.jpg"
        }
    ]
}
```

**À montrer au jury :** ouvrir `url_image` dans le navigateur. Chaque lecture
stockée est accompagnée du crop de plaque qui l'a produite — on peut donc
vérifier visuellement, à l'œil nu, que la lecture est juste. C'est un point fort :
le système est auditable, il ne demande pas qu'on lui fasse confiance.

Documentation interactive de l'API : <http://localhost:8000/docs>

---

# Guide de présentation

## 1. Ordre recommandé

| # | Quoi | Durée | Pourquoi en premier |
|---|---|---|---|
| 1 | **A.1 — images fixes** | 2 min | Ça marche toujours. On installe la crédibilité avant tout risque technique. |
| 2 | **B.2 — caméra live** (ou A.2 si la caméra est muette) | 3 min | Le moment spectaculaire : la plaque est lue en direct. |
| 3 | **C — API + crops** | 2 min | Montre que ce n'est pas une démo jouet : c'est stocké, interrogeable, auditable. |
| 4 | **Limites** (§3 ci-dessous) | 2 min | Les annoncer soi-même vaut mieux que se les faire extraire. |

**Lancer C dans un terminal séparé avant de commencer** : le serveur met quelques
secondes à démarrer, autant que ce soit déjà fait.

## 2. Expliquer l'architecture à un jury non-spécialiste

Une phrase par étage, dans l'ordre où la donnée circule :

> « La caméra filme l'entrée. **(1)** Un premier réseau de neurones, YOLO, ne
> cherche qu'une chose dans l'image : *où* est la plaque — pas ce qui est écrit
> dessus, juste sa position. **(2)** On découpe ensuite cette plaque en trois
> morceaux : les chiffres de gauche, la lettre arabe au milieu, les chiffres de
> droite. **(3)** Chaque morceau va au spécialiste qui lui convient : un moteur
> d'OCR pour les chiffres, et un second réseau de neurones que nous avons
> entraîné nous-mêmes pour la lettre arabe. **(4)** Comme la caméra voit la même
> voiture des dizaines de fois par seconde, on ne retient un matricule que si
> plusieurs images successives sont d'accord — ça élimine les erreurs
> ponctuelles dues à un reflet ou à un flou. **(5)** Enfin une dernière
> vérification contrôle que le résultat a bien la forme d'une plaque marocaine
> légale, et seulement alors on l'enregistre en base avec la photo. »

**L'image à dessiner au tableau si besoin :**

```
Caméra → YOLO (où ?) → Découpage en 3 zones → OCR chiffres + CNN lettre
       → Vote sur plusieurs images → Validation du format → Base + photo
```

**Si on demande « pourquoi deux modèles différents ? »** — Parce que les chiffres
sont latins et parfaitement standards : un moteur du commerce les lit très bien.
La lettre arabe sur plaque, elle, est un cas particulier : les moteurs génériques
perdent la hamza (le petit signe au-dessus du `أ`), qui est justement ce qui
distingue la lettre. D'où un modèle dédié, entraîné sur nos propres plaques.

## 3. Les limites à assumer — et comment les dire

Les présenter comme **le résultat d'une méthode**, parce que c'en est un.

### « Vous ne lisez que 3 lettres sur 28 »

> « Exact, et c'est un choix, pas un oubli. Nos 434 plaques réelles ne
> contiennent en tout que 6 lettres distinctes — les plaques civiles marocaines
> n'en utilisent qu'une poignée. Après annotation manuelle, seules `أ`, `د` et
> `ب` atteignaient l'effectif minimal que nous nous étions fixé **avant** de
> regarder les données. Nous aurions pu entraîner sur `ه` avec ses 10 images :
> le modèle aurait produit un chiffre de précision, mais un chiffre auquel nous
> n'aurions pas pu croire. Nous avons préféré couvrir moins et savoir ce que ça
> vaut. Sur une zone correctement cadrée, nous lisons juste dans **73 % des
> cas** — mesuré sur un jeu de test que nous n'avons ouvert qu'une seule fois,
> à la toute fin. »

**Si on insiste : « et sur une plaque avec ه, il se passe quoi ? »** — Répondre
franchement : « Il répond `د`, à tort, et il ne le signale pas. C'est la limite
que nous connaissons le mieux et c'est la première chose à corriger : il faut
annoter davantage de plaques portant les lettres manquantes. »

### « 0,79 image par seconde, c'est lent »

> « Entre 0,6 et 0,8 selon la charge — oui, sur processeur, sans carte graphique. C'est suffisant pour l'usage visé
> — un portail ou une barrière, où la voiture s'arrête de toute façon. Ça ne
> conviendrait pas à de la voie rapide. Les leviers connus sont là et non
> exploités : le GPU, l'accélération oneDNN, et l'échantillonnage d'images. »

### « Pourquoi SQLite et pas MySQL ? »

> « Le schéma MySQL est écrit et le code sait s'y connecter. Le repli SQLite
> s'active automatiquement quand le serveur n'est pas joignable, ce qui permet
> de faire tourner toute la chaîne sans dépendre d'une infrastructure — comme
> aujourd'hui. Basculer sur MySQL ne demande que de renseigner quatre variables
> d'environnement. »

### Ce qu'on peut ajouter si on veut marquer un point de méthode

> « Une chose que nous avons trouvée en route : le découpage train/test fourni
> avec le jeu de données public était faussé — il séparait les *retouches* d'une
> même photo, pas les photos elles-mêmes. La même voiture se retrouvait des deux
> côtés. Toute précision mesurée dessus était optimiste. Nous avons refait le
> découpage par véhicule. C'est ce qui explique que nos chiffres soient plus bas
> que ceux qu'on lit souvent — ils sont simplement honnêtes. »

## 4. Plan de secours

| Problème | Réaction immédiate |
|---|---|
| **La caméra ne répond pas** | Ne pas déboguer. Dire : « la caméra est sur un autre réseau, je vous montre la même chaîne sur un enregistrement » → lancer **A.2**. Le code exécuté est identique. |
| **La fenêtre vidéo ne s'ouvre pas** | Relancer avec `--no-display`. Les matricules s'affichent dans le terminal. |
| **PaddleOCR met longtemps au premier lancement** | Normal : il charge ses modèles. **Faire un lancement à blanc avant la présentation** pour que tout soit en cache. |
| **L'API ne démarre pas (port occupé)** | `--port 8001`, et adapter l'URL du `curl`. |
| **Tout échoue** | Le README contient toutes les sorties réelles, et `docs/arabic_letter_model.md` toutes les mesures. Présenter les chiffres et la méthode ; ils tiennent seuls. |

**À faire dans les 10 minutes avant de présenter :**

```bash
cd ~/PycharmProjects/anpr-project && source .venv/bin/activate && export PYTHONPATH=$PWD
python -m anpr_maroc.scripts.test_rtsp_connection          # la caméra répond-elle ?
python -m anpr_maroc.scripts.run_pipeline --image data/sample_plates/3.png --segment  # préchauffe PaddleOCR
python -m anpr_maroc.scripts.run_server --port 8000 &      # API prête dans un coin
```

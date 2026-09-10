"""Validation stricte du format d'un matricule marocain.

Deux formats sont reconnus, chacun avec sa propre regex :

  * ``1_ligne``   : ``NNNNN - L - NN``   (plaque standard automobile)
  * ``2_lignes``  : mêmes champs répartis sur deux lignes (motos, petits
    véhicules) ; la série y est souvent plus courte, la contrainte sur le
    nombre de chiffres de gauche est donc assouplie.

Toute lecture qui ne correspond à AUCUN des deux formats est rejetée : la
fonction retourne ``valid=False`` et un motif de rejet exploitable. Aucun
appelant ne doit afficher ni stocker un matricule non validé ici.

Troisième état : la lettre indéterminée
---------------------------------------
Le CNN lettre ne couvre pas tout l'alphabet (cf. README §4.1). En dessous de
son seuil de confiance il ne renvoie plus une lettre plausible mais le
sentinelle ``UNKNOWN_LETTER``. Une lecture de ce type traverse quand même la
validation de structure — les chiffres, eux, sont vérifiés normalement — et
ressort avec ``valid=False`` **et** ``letter_known=False``.

La distinction est volontaire :

  * ``valid=False, letter_known=True``  -> lecture aberrante, à jeter.
  * ``valid=False, letter_known=False`` -> chiffres sûrs, lettre inconnue.
    Exploitable, mais un appelant ne doit JAMAIS la présenter comme une
    lecture complète.

Un appelant qui ne teste que ``valid`` conserve donc exactement l'ancien
comportement : il ne peut pas stocker par inadvertance une lettre incertaine.
"""

from __future__ import annotations

import re
import typing as t
from dataclasses import dataclass

# Lettres de série admises sur une plaque marocaine.
# NOTE : le classifieur CNN embarqué (models/arabic_letter_classifier_real2.pt)
# ne connaît aujourd'hui qu'un sous-ensemble de ces lettres ('أ', 'ب', 'د') :
# ce sont les seules pour lesquelles le jeu réel atteint l'effectif minimal
# (docs/arabic_letter_model.md §2). Toute autre lettre ressort en sentinelle.
# Le validateur reste volontairement plus large pour ne pas devenir le facteur
# limitant le jour où le CNN est ré-entraîné sur l'alphabet complet.
MOROCCAN_PLATE_LETTERS = "أابتجدهوطشمق"

# Sentinelle émise à la place d'une lettre quand la confiance du CNN est
# insuffisante. Un caractère hors alphabet arabe est choisi à dessein : il ne
# peut jamais être confondu avec une lecture réelle.
UNKNOWN_LETTER = "?"
UNKNOWN_LETTER_LABEL = "lettre indéterminée"

# Le sentinelle est accepté par la regex de structure afin que les contraintes
# de longueur sur les chiffres restent appliquées même sans lettre lisible.
_LETTERS_CLASS = f"[{re.escape(MOROCCAN_PLATE_LETTERS + UNKNOWN_LETTER)}]"

# Séparateur toléré entre les champs à l'entrée : tiret, barre, espace ou rien.
_SEP = r"\s*[-|/]?\s*"

# Format standard 1 ligne : 4 ou 5 chiffres, une lettre arabe, 1 ou 2 chiffres.
RE_FORMAT_1LIGNE = re.compile(
    rf"^(?P<left>\d{{4,5}}){_SEP}(?P<letter>{_LETTERS_CLASS}){_SEP}(?P<right>\d{{1,2}})$"
)

# Format 2 lignes : série de 1 à 5 chiffres (les plaques deux-lignes portent
# fréquemment une série courte), une lettre arabe, 1 ou 2 chiffres de région.
RE_FORMAT_2LIGNES = re.compile(
    rf"^(?P<left>\d{{1,5}}){_SEP}(?P<letter>{_LETTERS_CLASS}){_SEP}(?P<right>\d{{1,2}})$"
)

FORMATS: dict[str, re.Pattern[str]] = {
    "1_ligne": RE_FORMAT_1LIGNE,
    "2_lignes": RE_FORMAT_2LIGNES,
}


@dataclass(frozen=True)
class ValidationResult:
    """Résultat de validation d'un matricule."""

    valid: bool
    matricule: str = ""
    left: str = ""
    letter: str = ""
    right: str = ""
    layout: str = ""
    reason: str = ""
    # False uniquement quand la structure est bonne mais que la lettre n'a pas
    # pu être identifiée avec assez de confiance (cf. docstring du module).
    letter_known: bool = True

    def __bool__(self) -> bool:
        return self.valid

    @property
    def partiel(self) -> bool:
        """Chiffres validés, lettre indéterminée."""
        return not self.valid and not self.letter_known

    def affichage(self) -> str:
        """Forme lisible pour un opérateur, jamais trompeuse."""
        if self.partiel:
            return f"{self.matricule}  ({UNKNOWN_LETTER_LABEL})"
        return self.matricule


def format_matricule(left: str, letter: str, right: str) -> str:
    """Représentation canonique affichée/stockée : ``13456-ب-27``."""
    return f"{left}-{letter}-{right}"


def _normalize(value: t.Any) -> str:
    return str(value or "").strip()


def validate_fields(
    left: t.Any,
    letter: t.Any,
    right: t.Any,
    layout: t.Optional[str] = None,
) -> ValidationResult:
    """Valide les trois champs issus de la segmentation.

    Args:
        left, letter, right: champs bruts (chiffres gauche, lettre arabe,
            chiffres droite).
        layout: ``'1_ligne'`` / ``'2_lignes'`` si connu. Si ``None``, la
            lecture est testée contre les deux formats et le premier qui
            correspond gagne.

    Returns:
        ValidationResult ; ``valid=False`` avec un ``reason`` si aucun format
        ne correspond.
    """
    left_s, letter_s, right_s = _normalize(left), _normalize(letter), _normalize(right)

    if not (left_s and letter_s and right_s):
        return ValidationResult(False, reason="champ manquant")

    candidate = format_matricule(left_s, letter_s, right_s)
    layouts = [layout] if layout in FORMATS else list(FORMATS)

    for name in layouts:
        match = FORMATS[name].match(candidate)
        if match:
            letter_known = match.group("letter") != UNKNOWN_LETTER
            return ValidationResult(
                # Une lettre non identifiée n'est jamais une lecture valide :
                # seuls les chiffres ont été vérifiés.
                valid=letter_known,
                matricule=format_matricule(
                    match.group("left"), match.group("letter"), match.group("right")
                ),
                left=match.group("left"),
                letter=match.group("letter"),
                right=match.group("right"),
                layout=name,
                reason="" if letter_known else UNKNOWN_LETTER_LABEL,
                letter_known=letter_known,
            )

    # Le sentinelle n'est pas une lettre hors alphabet : le motif de rejet doit
    # pointer le vrai défaut (chiffres), pas la lettre déjà signalée inconnue.
    if letter_s != UNKNOWN_LETTER and letter_s not in MOROCCAN_PLATE_LETTERS:
        reason = f"lettre '{letter_s}' hors alphabet plaque"
    elif not left_s.isdigit() or not right_s.isdigit():
        reason = "champs numériques non exclusivement chiffrés"
    else:
        reason = (
            f"longueurs invalides (gauche={len(left_s)}, droite={len(right_s)})"
        )
    return ValidationResult(False, reason=reason)


def validate_matricule(text: t.Any, layout: t.Optional[str] = None) -> ValidationResult:
    """Valide un matricule déjà assemblé sous forme de chaîne."""
    raw = _normalize(text)
    if not raw:
        return ValidationResult(False, reason="lecture vide")

    parts = [p for p in re.split(r"[-|/\s]+", raw) if p]
    if len(parts) != 3:
        return ValidationResult(False, reason="structure attendue: gauche-lettre-droite")
    return validate_fields(parts[0], parts[1], parts[2], layout=layout)


def is_valid(text: t.Any, layout: t.Optional[str] = None) -> bool:
    """Raccourci booléen."""
    return validate_matricule(text, layout=layout).valid

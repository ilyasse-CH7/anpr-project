import os
import re
import typing as t

import cv2
import numpy as np
from paddleocr import PaddleOCR

from anpr_maroc.ocr.arabic_letter_model import ArabicLetterClassifier, DEFAULT_MODEL_PATH
from anpr_maroc.processing.segmenter import detect_plate_layout, segment_plate_2lines
from anpr_maroc.processing.validator import UNKNOWN_LETTER

DIGITS_RE = re.compile(r"\D")


class PaddleOCRReader:
    """PaddleOCR wrapper that accepts file paths or numpy images and returns
    structured OCR results tuned for Moroccan plates (latin digits + one
    Arabic letter in the middle).

    The digit engine runs PaddleOCR in ``lang='en'`` — plate digits are Latin,
    and PP-OCR's Arabic recognizer was found unreliable on the isolated
    letter glyph during evaluation. The Arabic letter itself is never read by
    this OCR engine: it is classified by the dedicated CNN in
    ``arabic_letter_model``, which already handles detached diacritics that
    no general-purpose OCR engine preserves.
    """

    ARABIC_RE = re.compile(r"[؀-ۿ]")
    ARABIC_LETTER_SET = "ابتثجحخسشصضطظعغفقكلمنهوياءئؤةأآإ"
    MATRICULE_RE = re.compile(r"(?P<left>\d{1,6})\s*[-\s]?\s*(?P<letter>[؀-ۿ])\s*[-\s]?\s*(?P<right>\d{1,6})")

    def __init__(
        self,
        languages: t.Optional[t.List[str]] = None,
        gpu: bool = False,
        conf_threshold: float = 0.3,
        letter_model_path: t.Optional[str] = None,
        letter_model_threshold: float = 0.65,
        enable_mkldnn: bool = False,
    ):
        """Initialize reader.

        Args:
            languages: unused, kept only for drop-in compatibility with the
                former EasyOCR-based reader. The digit engine is always 'en'.
            gpu: whether to run PaddleOCR on GPU.
            conf_threshold: minimum confidence to keep a detection (0..1)
            enable_mkldnn: PaddlePaddle's oneDNN CPU backend. Left off by
                default: on this project's CPU wheel it raises
                ``NotImplementedError: ConvertPirAttribute2RuntimeAttribute``
                on PP-OCRv6. Re-enable only after confirming it runs on your
                hardware/PaddlePaddle build — it is meaningfully faster.
        """
        self.languages = languages
        self.gpu = gpu
        self.conf_threshold = conf_threshold
        set_str = "".join(self.ARABIC_LETTER_SET)
        self.ARABIC_LETTER_RE = re.compile(rf"^[{set_str}]$")
        model_path = letter_model_path or os.getenv("ANPR_ARABIC_LETTER_MODEL", str(DEFAULT_MODEL_PATH))
        self.letter_model = ArabicLetterClassifier(model_path)
        self.letter_model_threshold = letter_model_threshold
        self.reader = PaddleOCR(
            lang="en",
            device="gpu" if gpu else "cpu",
            enable_mkldnn=enable_mkldnn,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def _load_image(self, image: t.Union[str, np.ndarray]) -> np.ndarray:
        """Load image from path or pass-through numpy array (BGR numpy expected).
        Converts grayscale to BGR for consistency.
        """
        if isinstance(image, np.ndarray):
            img = image
        else:
            img = cv2.imread(image)
            if img is None:
                raise FileNotFoundError(f"Image not found or cannot be read: {image}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _preprocess_for_paddle(self, image: np.ndarray, target_height: int = 120) -> np.ndarray:
        """Normalize a plate or numeric zone before a PaddleOCR inference.

        Small crops are enlarged to a stable glyph height, then CLAHE
        recovers local contrast. Unlike the former EasyOCR pipeline, this was
        validated to never regress PaddleOCR's accuracy versus raw pixels —
        it only helps on heavily downscaled crops (confidence 0.93 -> 1.0 on
        a 5x-reduced test crop). Kept intentionally light for that reason.
        """
        img = self._load_image(image)
        height, width = img.shape[:2]
        if 0 < height < target_height:
            scale = target_height / height
            img = cv2.resize(
                img,
                (max(1, int(round(width * scale))), target_height),
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)

        # Moroccan plates are dark digits on a light background. Normalize
        # polarity before the morphological pass so it behaves consistently
        # regardless of exposure (e.g. underlit night captures).
        if float(np.mean(gray)) < 110:
            gray = 255 - gray

        # Reconnect thin strokes broken by JPEG blocking or motion blur.
        # Verified neutral-to-positive for PaddleOCR (never dropped a correct
        # reading in testing), unlike the former EasyOCR pipeline where the
        # equivalent pass measurably hurt confidence.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def extract_numeric_string(text: str) -> str:
        """Keep ASCII digits only, preserving their left-to-right order."""
        return DIGITS_RE.sub("", str(text))

    def _run_paddle(self, img: np.ndarray) -> t.List[t.Dict[str, t.Any]]:
        """Run the PaddleOCR pipeline and normalize its output.

        PaddleOCR expects a BGR array (same convention as cv2.imread) — do
        not convert to RGB. Returns a list of {'bbox', 'text', 'conf'} in
        reading order (sorted left-to-right by box), matching the digit
        order on the plate regardless of how PP-OCR grouped text regions.
        """
        items = []
        for page in self.reader.predict(img):
            texts = page.get("rec_texts", []) or []
            scores = page.get("rec_scores", []) or []
            boxes = page.get("rec_polys", None)
            if boxes is None:
                boxes = [None] * len(texts)
            for text, score, bbox in zip(texts, scores, boxes):
                items.append({"bbox": bbox, "text": str(text), "conf": float(score)})
        items.sort(key=lambda it: float(np.min(it["bbox"][:, 0])) if it["bbox"] is not None else 0.0)
        return items

    def read_plate(
        self, image: t.Union[str, np.ndarray], preprocess: bool = True
    ) -> t.List[t.Dict[str, t.Any]]:
        """Run PaddleOCR on the full plate image and return structured results.

        Returns list of dicts: { 'bbox': quad points or None, 'text': str, 'conf': float }
        """
        img = self._load_image(image)
        if preprocess:
            img = self._preprocess_for_paddle(img)
        raw = self._run_paddle(img)
        return [r for r in raw if r["conf"] >= self.conf_threshold]

    def read_plate_text_only(self, image: t.Union[str, np.ndarray]) -> str:
        """Return concatenated text detected on plate (filtered by confidence)."""
        results = self.read_plate(image)
        return " ".join([r["text"] for r in results])

    def read_plate_zones(
        self, zones: t.List[t.Union[str, np.ndarray]], preprocess: bool = True
    ) -> t.List[t.Dict[str, t.Any]]:
        """OCR a list of zone images (left, letter, right) and return per-zone results.

        zones: list of 3 images (left_numbers, arabic_letter_zone, right_numbers)
        Each item may be a path or numpy array.
        Returns: list of dicts [{'zone':'left','text':..., 'conf':...}, ...]
        """
        out = []
        names = ["left", "letter", "right"]
        for name, z in zip(names, zones):
            img = self._load_image(z)
            # The dedicated CNN is the sole recognizer for the letter zone: it
            # retains detached hamza/madda components that a general OCR
            # engine discards. No PaddleOCR fallback here.
            if name == "letter":
                model_letter, model_confidence = ("", 0.0)
                if self.letter_model.available:
                    model_letter, model_confidence = self.letter_model.predict(img)
                if os.getenv("ANPR_DEBUG"):
                    print(f"[DEBUG] arabic_letter_model -> '{model_letter}' (conf={model_confidence:.3f})")
                # Sous le seuil, on n'émet PAS la classe la moins improbable :
                # le CNN ne couvre qu'une partie de l'alphabet (README §4.1) et
                # sort alors une lettre fausse avec une confiance d'apparence
                # honorable (~0.60 mesuré). Le sentinelle rend l'incertitude
                # visible en aval au lieu de la déguiser en lecture sûre.
                letter_known = bool(model_letter) and model_confidence >= self.letter_model_threshold
                out.append(
                    {
                        "zone": name,
                        "text": model_letter if letter_known else UNKNOWN_LETTER,
                        "conf": model_confidence,
                        "source": "arabic_letter_model",
                        "letter_known": letter_known,
                    }
                )
                continue

            # Digit zones always run on enhanced pixels: cheap, and verified
            # never to regress PaddleOCR's accuracy (see _preprocess_for_paddle).
            processed = self._preprocess_for_paddle(img)
            raw = self._run_paddle(processed)

            digit_fragments = []
            confs = []
            for item in raw:
                digits = self.extract_numeric_string(item["text"])
                if digits:
                    digit_fragments.append(digits)
                    confs.append(item["conf"])

            if not digit_fragments:
                out.append({"zone": name, "text": "", "conf": 0.0})
                continue

            text = "".join(digit_fragments)
            conf = float(min(confs))
            if conf < self.conf_threshold and len(text) < 4:
                out.append({"zone": name, "text": "", "conf": 0.0})
            else:
                out.append({"zone": name, "text": text, "conf": conf})
        return out

    def filter_plate_candidates(self, ocr_full: t.Sequence[t.Dict[str, t.Any]]) -> t.List[str]:
        """Keep OCR tokens likely to contain a Moroccan matricule.

        This is a fallback used when segmentation fails and we need to parse a full-image OCR output.
        """
        candidates: t.List[str] = []
        for item in ocr_full or []:
            txt = str(item.get("text", "")).strip()
            if not txt:
                continue
            cleaned = re.sub(r"[^0-9؀-ۿ\s\-|_/|]", "", txt)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if len(cleaned) < 2:
                continue
            candidates.append(cleaned)
        return candidates

    def classify_matricule(self, candidates: t.Sequence[str]) -> t.Dict[str, t.Any]:
        """Classify a full-image OCR token list into left/letter/right blocks.

        Intentionally conservative: returns the best candidate and a valid flag only when the
        candidate matches a Moroccan plate shape (digits + one Arabic letter + digits).
        """
        best: t.Dict[str, t.Any] = {"raw": "", "left": "", "letter": "", "right": "", "valid": False}
        for cand in candidates or []:
            parsed = self.parse_matricule(cand)
            if parsed.get("valid"):
                return parsed
            score = 0
            score += len(re.findall(r"\d", cand))
            score += 3 if self.ARABIC_RE.search(cand) else 0
            if score > 0 and (len(best["raw"]) < len(cand) or not best["raw"]):
                best = parsed
        return best

    def parse_matricule(self, text: t.Union[str, t.Dict[str, str], t.Sequence[str], None]) -> t.Dict[str, t.Any]:
        """Validate a Moroccan plate from either:
        - a raw OCR string
        - a dict with left/letter/right values
        - a sequence of exactly 3 values [left, letter, right]

        The segmentation path should already provide the 3 blocks in order.
        """
        if text is None:
            return {"raw": "", "left": "", "letter": "", "right": "", "valid": False}

        if isinstance(text, dict):
            left = self.extract_numeric_string(text.get("left", ""))
            letter = str(text.get("letter", "")).strip()
            right = self.extract_numeric_string(text.get("right", ""))
            valid = bool(re.fullmatch(r"\d{1,6}", left) and self.ARABIC_LETTER_RE.fullmatch(letter) and re.fullmatch(r"\d{1,6}", right))
            return {"raw": " ".join([left, letter, right]).strip(), "left": left, "letter": letter, "right": right, "valid": valid}

        if isinstance(text, (list, tuple)):
            if len(text) != 3:
                return {"raw": str(text), "left": "", "letter": "", "right": "", "valid": False}
            left = self.extract_numeric_string(text[0])
            letter = str(text[1]).strip()
            right = self.extract_numeric_string(text[2])
            valid = bool(re.fullmatch(r"\d{1,6}", left) and self.ARABIC_LETTER_RE.fullmatch(letter) and re.fullmatch(r"\d{1,6}", right))
            return {"raw": " ".join([left, letter, right]).strip(), "left": left, "letter": letter, "right": right, "valid": valid}

        if not text:
            return {"raw": text, "left": "", "letter": "", "right": "", "valid": False}

        s = str(text).strip()
        m = self.MATRICULE_RE.search(s)
        if m:
            left = self.extract_numeric_string(m.group("left"))
            letter = m.group("letter")
            right = self.extract_numeric_string(m.group("right"))
            valid = bool(re.fullmatch(r"\d{1,6}", left) and self.ARABIC_LETTER_RE.fullmatch(letter) and re.fullmatch(r"\d{1,6}", right))
            return {"raw": s, "left": left, "letter": letter, "right": right, "valid": valid}

        parts = re.split(r"\s+|[-/|]", s)
        parts = [p for p in parts if p and p.strip()]
        if len(parts) >= 3:
            left = self.extract_numeric_string(parts[0])
            letter = parts[1].strip()
            right = self.extract_numeric_string(parts[2])
            valid = bool(re.fullmatch(r"\d{1,6}", left) and self.ARABIC_LETTER_RE.fullmatch(letter) and re.fullmatch(r"\d{1,6}", right))
            return {"raw": s, "left": left, "letter": letter, "right": right, "valid": valid}

        ar_match = self.ARABIC_RE.search(s)
        if ar_match:
            idx = ar_match.start()
            left = self.extract_numeric_string(s[:idx])
            right = self.extract_numeric_string(s[idx + 1:])
            letter = ar_match.group(0)
            valid = bool(re.fullmatch(r"\d{1,6}", left) and self.ARABIC_LETTER_RE.fullmatch(letter) and re.fullmatch(r"\d{1,6}", right))
            return {"raw": s, "left": left, "letter": letter, "right": right, "valid": valid}

        return {"raw": s, "left": "", "letter": "", "right": "", "valid": False}

    def _read_and_parse_once(
        self,
        image: t.Union[str, np.ndarray],
        try_segment: t.Optional[t.Callable[..., t.List[np.ndarray]]] = None,
        debug_dir: t.Optional[str] = None,
        preprocess: bool = False,
    ) -> t.Dict[str, t.Any]:
        """Run one raw or preprocessed OCR pass and return its parsed result.

        Behavior:
          - Run full-image OCR first to collect fallback candidates.
          - If try_segment is provided, route by layout (1-line vs 2-line) and segment.
          - If segmentation yields 3 zones, return that structured result.
          - Otherwise fall back to classifying tokens from the full-image OCR pass.
            (There is no OCR-assisted Arabic-bbox segmentation fallback here: the
            digit engine runs lang='en' and will never detect the Arabic letter,
            by design — see class docstring.)
        """
        img = self._load_image(image)
        ocr_full = self.read_plate(img, preprocess=preprocess)
        raw_text = " ".join([it["text"] for it in ocr_full])
        layout = ""

        if try_segment is not None:
            try:
                layout = detect_plate_layout(img)
                if layout == "2_lignes":
                    try:
                        zones = segment_plate_2lines(img, debug_dir=debug_dir)
                    except TypeError:
                        zones = segment_plate_2lines(img)
                else:
                    try:
                        zones = try_segment(img, debug_dir=debug_dir)
                    except TypeError:
                        zones = try_segment(img)
                if len(zones) == 3:
                    zone_results = self.read_plate_zones(zones, preprocess=preprocess)
                    texts = [zr['text'] for zr in zone_results]
                    parsed = self.parse_matricule({"left": texts[0], "letter": texts[1], "right": texts[2]})
                    parsed["raw_text"] = " ".join(texts)
                    # Le layout ('1_ligne' / '2_lignes') est remonté tel quel :
                    # la validation regex en aval choisit sa regex avec, plutôt
                    # que d'accepter le format le plus permissif par défaut.
                    parsed["layout"] = layout
                    return {"raw_text": parsed["raw_text"], "zone_results": zone_results, "parsed": parsed, "ocr_results": ocr_full, "layout": layout}
            except Exception:
                pass

        candidates = self.filter_plate_candidates(ocr_full)
        parsed = self.classify_matricule(candidates)
        parsed["raw_text"] = raw_text
        parsed["candidates"] = candidates
        parsed["layout"] = layout
        return {"raw_text": raw_text, "candidates": candidates, "parsed": parsed, "ocr_results": ocr_full, "layout": layout}

    def read_and_parse(
        self,
        image: t.Union[str, np.ndarray],
        try_segment: t.Optional[t.Callable[..., t.List[np.ndarray]]] = None,
        debug_dir: t.Optional[str] = None,
    ) -> t.Dict[str, t.Any]:
        """Read a plate in two passes, preserving a valid raw recognition.

        The first pass intentionally uses unmodified pixels. Only an invalid
        or incomplete matricule triggers the enhanced second pass.
        """
        raw_result = self._read_and_parse_once(
            image, try_segment=try_segment, debug_dir=debug_dir, preprocess=False
        )
        raw_parsed = raw_result.get("parsed", {})
        required = ("left", "letter", "right")
        if raw_parsed.get("valid") and all(raw_parsed.get(field) for field in required):
            raw_result["ocr_pass"] = "raw"
            return raw_result

        enhanced_result = self._read_and_parse_once(
            image, try_segment=try_segment, debug_dir=debug_dir, preprocess=True
        )
        enhanced_result["ocr_pass"] = "preprocessed"

        # Aucune des deux passes n'est pleinement valide : on garde la plus
        # complète. Sans cela, une passe rehaussée qui ne trouve rien écrasait
        # une lecture partielle exploitable (chiffres sûrs, lettre indéterminée).
        enhanced_parsed = enhanced_result.get("parsed", {})
        if not enhanced_parsed.get("valid"):
            filled = lambda p: sum(1 for f in required if p.get(f))  # noqa: E731
            if filled(raw_parsed) > filled(enhanced_parsed):
                raw_result["ocr_pass"] = "raw"
                return raw_result
        return enhanced_result

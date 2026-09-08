import os
import re
import typing as t

import cv2
import numpy as np
import easyocr
from PIL import Image, ImageDraw, ImageFont

from anpr_maroc import config as _config
from anpr_maroc.ocr.arabic_letter_model import ArabicLetterClassifier, DEFAULT_MODEL_PATH
from anpr_maroc.processing.segmenter import detect_plate_layout, segment_plate_2lines

class EasyOCRReader:
    """EasyOCR wrapper that accepts file paths or numpy images and returns
    structured OCR results tuned for Moroccan plates (latin digits + one
    Arabic letter in the middle).

    Features added:
    - accept image path or numpy.ndarray
    - configurable languages and GPU usage
    - confidence threshold
    - helpers to OCR segmented zones and to parse a matricule string
    """

    ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
    ARABIC_LETTER_SET = "ابتثجحخسشصضطظعغفقكلمنهوياءئؤةأآإ"
    # rough regex: left digits, optional separators, one Arabic letter, optional sep, right digits
    MATRICULE_RE = re.compile(r"(?P<left>\d{1,6})\s*[-\s]?\s*(?P<letter>[\u0600-\u06FF])\s*[-\s]?\s*(?P<right>\d{1,6})")

    def __init__(
        self,
        languages: t.Optional[t.List[str]] = None,
        gpu: bool = False,
        conf_threshold: float = 0.3,
        letter_model_path: t.Optional[str] = None,
        letter_model_threshold: float = 0.65,
    ):
        """Initialize reader.

        Args:
            languages: list of language codes for EasyOCR (default ['ar','en'])
            gpu: whether to enable GPU (CUDA)
            conf_threshold: minimum confidence to keep a detection (0..1)
        """
        if languages is None:
            languages = ["ar", "en"]
        self.languages = languages
        self.gpu = gpu
        self.conf_threshold = conf_threshold
        set_str = "".join(self.ARABIC_LETTER_SET)
        self.ARABIC_LETTER_RE = re.compile(rf"^[{set_str}]$")
        model_path = letter_model_path or os.getenv("ANPR_ARABIC_LETTER_MODEL", str(DEFAULT_MODEL_PATH))
        self.letter_model = ArabicLetterClassifier(model_path)
        self.letter_model_threshold = letter_model_threshold
        # instantiate EasyOCR reader once
        self.reader = easyocr.Reader(self.languages, gpu=self.gpu)

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
        # Ensure 3-channel image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _preprocess_for_easyocr(self, image: np.ndarray) -> np.ndarray:
        """Normalize a plate or numeric zone before an EasyOCR inference.

        Small crops are enlarged to a stable glyph height, then CLAHE and a
        light blur improve local contrast while suppressing fine sensor noise.
        """
        img = self._load_image(image)
        height, width = img.shape[:2]
        if 0 < height < 120:
            scale = 120.0 / height
            img = cv2.resize(
                img,
                (max(1, int(round(width * scale))), 120),
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def read_plate(
        self, image: t.Union[str, np.ndarray], preprocess: bool = True
    ) -> t.List[t.Dict[str, t.Any]]:
        """Run EasyOCR on full plate image and return structured results.

        Returns list of dicts: { 'bbox': [(x1,y1),(x2,y2),(x3,y3),(x4,y4)], 'text': str, 'conf': float }
        """
        img = self._load_image(image)
        if preprocess:
            img = self._preprocess_for_easyocr(img)
        # EasyOCR expects RGB
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        raw = self.reader.readtext(rgb)
        results = []
        for bbox, text, conf in raw:
            if conf >= self.conf_threshold:
                results.append({"bbox": bbox, "text": text, "conf": float(conf)})
        return results

    def read_plate_text_only(self, image: t.Union[str, np.ndarray]) -> str:
        """Return concatenated text detected on plate (filtered by confidence)."""
        results = self.read_plate(image)
        return " ".join([r["text"] for r in results])

    def _preprocess_zone_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Per-zone preprocessing to improve OCR accuracy using CLAHE and light denoising.

        Steps:
        - convert to grayscale
        - CLAHE (adaptive histogram equalization)
        - slight gaussian blur
        - morphological closing to reduce small gaps
        - optional upscaling to improve thin glyphs on small plate crops
        Returns BGR image suitable for EasyOCR (converted back to 3-channel)
        """
        if img is None:
            return img
        img = self._preprocess_for_easyocr(img)
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        # CLAHE to improve local contrast (helps thin strokes)
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        except Exception:
            try:
                gray = cv2.equalizeHist(gray)
            except Exception:
                pass

        # light denoise + threshold style to stabilise digits on CPU
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        try:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        except Exception:
            pass

        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

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
            # retains detached hamza/madda components that EasyOCR and the
            # former contour template both discard. No EasyOCR fallback here.
            if name == "letter":
                model_letter, model_confidence = ("", 0.0)
                if self.letter_model.available:
                    model_letter, model_confidence = self.letter_model.predict(img)
                if os.getenv("ANPR_DEBUG"):
                    print(f"[DEBUG] arabic_letter_model -> '{model_letter}' (conf={model_confidence:.3f})")
                if model_letter and model_confidence < self.letter_model_threshold:
                    model_letter = ""
                out.append({"zone": name, "text": model_letter, "conf": model_confidence, "source": "arabic_letter_model"})
                continue
            candidates = []
            allowlist_digits = "0123456789"

            if not preprocess:
                # First pass: exactly the raw crop, with no resize, CLAHE, blur
                # or threshold-based variant.
                variant_list = [("raw", img)]
            else:
                # Second pass: contrast enhancement and numeric variants.
                base = self._preprocess_for_easyocr(img)
                pre = self._preprocess_zone_for_ocr(img)
                if name == 'left':
                # Stable left-only pipeline: keep only a few variants and aggregate candidates later.
                    variant_list = [("base", base), ("pre", pre)]
                    try:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        blur = cv2.GaussianBlur(gray, (3, 3), 0)
                        num_thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9)
                        if np.mean(num_thr) < 127:
                            num_thr = 255 - num_thr
                        num_pre = cv2.cvtColor(num_thr, cv2.COLOR_GRAY2BGR)
                        variant_list.append(("num_pre", self._preprocess_for_easyocr(num_pre)))
                    except Exception:
                        pass
                elif name == 'right':
                # keep mild variants only
                    variant_list = [("base", base), ("pre", pre)]
                    try:
                        # small adaptive binarize as optional variant
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        blur = cv2.GaussianBlur(gray, (3, 3), 0)
                        num_thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 9)
                        if np.mean(num_thr) < 127:
                            num_thr = 255 - num_thr
                        num_pre = cv2.cvtColor(num_thr, cv2.COLOR_GRAY2BGR)
                        variant_list.append(("num_pre", self._preprocess_for_easyocr(num_pre)))
                    except Exception:
                        pass

            # Collect OCR results from variants; digits only for left/right.
            for variant_name, img_variant in variant_list:
                rgb = cv2.cvtColor(self._load_image(img_variant), cv2.COLOR_BGR2RGB)
                raw = self.reader.readtext(rgb, allowlist=allowlist_digits)
                for bbox, text, conf in raw:
                    candidates.append((bbox, text, float(conf), variant_name))

            # numeric zones: keep numeric-only OCR candidates, reject letters/digits mixed noise.
            numeric_candidates = []
            for bbox, text, conf, source in candidates:
                cleaned = str(text).strip()
                if re.fullmatch(r"\d+", cleaned):
                    numeric_candidates.append((bbox, cleaned, float(conf), source))
            if not numeric_candidates:
                out.append({"zone": name, "text": "", "conf": 0.0})
            else:
                # left zone: aggregate across variants instead of forcing a single variant winner.
                if name == 'left':
                    by_text = {}
                    for _, txt, conf, _ in numeric_candidates:
                        by_text.setdefault(txt, {"count": 0, "conf_sum": 0.0})
                        by_text[txt]["count"] += 1
                        by_text[txt]["conf_sum"] += conf
                    best_text, stats = max(by_text.items(), key=lambda kv: (kv[1]["count"], kv[1]["conf_sum"] / kv[1]["count"], len(kv[0])))
                    best_conf = stats["conf_sum"] / stats["count"]
                    if best_conf < self.conf_threshold and len(best_text) < 4:
                        out.append({"zone": name, "text": "", "conf": 0.0})
                    else:
                        out.append({"zone": name, "text": best_text, "conf": float(best_conf)})
                else:
                    # right zone: pick highest confidence, tie-breaker longest length
                    numeric_candidates.sort(key=lambda x: (x[2], len(x[1])), reverse=True)
                    best = numeric_candidates[0]
                    if best[2] < self.conf_threshold:
                        if len(best[1]) >= 4:
                            out.append({"zone": name, "text": best[1], "conf": float(best[2])})
                        else:
                            out.append({"zone": name, "text": "", "conf": 0.0})
                    else:
                        out.append({"zone": name, "text": best[1], "conf": float(best[2])})
        return out

    def _render_letter_template(self, letter: str, size: int = 64) -> np.ndarray:
        """Render a template for an Arabic letter using a system font; used as a fallback recognizer."""
        try:
            font_candidates = [
                "/usr/share/fonts/google-droid-sans-fonts/DroidKufi-Regular.ttf",
                "/usr/share/fonts/google-droid-sans-fonts/DroidKufi-Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansCondensed.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
            font_path = next((p for p in font_candidates if os.path.exists(p)), None)
            if font_path is None:
                return None
            img = Image.new("L", (size, size), 255)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_path, max(28, int(size * 0.7)))
            bbox = draw.textbbox((0, 0), letter, font=font)
            x = (size - (bbox[2] - bbox[0])) / 2 - bbox[0]
            y = (size - (bbox[3] - bbox[1])) / 2 - bbox[1]
            draw.text((x, y), letter, fill=0, font=font)
            arr = np.asarray(img)
            return arr
        except Exception:
            return None

    def recognize_arabic_letter(self, letter_crop: t.Union[str, np.ndarray]) -> str:
        """Fallback Arabic-letter recognizer based on template matching.

        This is used only when EasyOCR fails or returns a digit-like shape for the central letter zone.
        """
        if isinstance(letter_crop, str):
            img = cv2.imread(letter_crop)
            if img is None:
                return ""
        else:
            img = np.array(letter_crop)

        if img is None:
            return ""
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ys, xs = np.where(thresh > 0)
        if xs.size == 0 or ys.size == 0:
            return ""
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
        crop = thresh[y0:y1 + 1, x0:x1 + 1]
        # add padding to keep letter centered
        pad = 10
        canvas = np.full((crop.shape[0] + 2 * pad, crop.shape[1] + 2 * pad), 255, dtype=np.uint8)
        canvas[pad:pad + crop.shape[0], pad:pad + crop.shape[1]] = crop
        canvas = cv2.resize(canvas, (64, 64), interpolation=cv2.INTER_CUBIC)

        best_letter = ""
        best_score = float("inf")
        for ch in self.ARABIC_LETTER_SET:
            template = self._render_letter_template(ch, size=64)
            if template is None:
                continue
            template_bin = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            # Use both absolute difference and inverted-image agreement to reduce false positives.
            diff = cv2.absdiff(canvas, template_bin)
            score = float(np.mean(diff))
            inv_diff = cv2.absdiff(255 - canvas, template_bin)
            score = min(score, float(np.mean(inv_diff)))
            if score < best_score:
                best_score = score
                best_letter = ch
        return best_letter if best_score < 60 else ""

    def filter_plate_candidates(self, ocr_full: t.Sequence[t.Dict[str, t.Any]]) -> t.List[str]:
        """Keep OCR tokens likely to contain a Moroccan matricule.

        This is a fallback used when segmentation fails and we need to parse a full-image OCR output.
        """
        candidates: t.List[str] = []
        for item in ocr_full or []:
            txt = str(item.get("text", "")).strip()
            if not txt:
                continue
            # keep arabic letters, digits and common separators; reject obviously empty garbage.
            cleaned = re.sub(r"[^0-9\u0600-\u06FF\s\-|_/|]", "", txt)
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            if len(cleaned) < 2:
                continue
            candidates.append(cleaned)
        return candidates

    def classify_matricule(self, candidates: t.Sequence[str]) -> t.Dict[str, t.Any]:
        """Classify a full-image OCR token list into left/letter/right blocks.

        This is intentionally conservative: it returns the best candidate and a valid flag only when the
        candidate matches a Moroccan plate shape (digits + one Arabic letter + digits).
        """
        best: t.Dict[str, t.Any] = {"raw": "", "left": "", "letter": "", "right": "", "valid": False}
        for cand in candidates or []:
            parsed = self.parse_matricule(cand)
            if parsed.get("valid"):
                return parsed
            # if not valid yet, keep the strongest-looking one by pattern score
            score = 0
            score += len(re.findall(r"\d", cand))
            score += 3 if self.ARABIC_RE.search(cand) else 0
            if score > 0 and (len(best["raw"]) < len(cand) or not best["raw"]):
                best = parsed
        return best

    @staticmethod
    def _clean_numeric_block(value: t.Any) -> str:
        """Keep ASCII digits only, preserving their left-to-right order."""
        return re.sub(r"[^0-9]", "", str(value))

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
            left = self._clean_numeric_block(text.get("left", ""))
            letter = str(text.get("letter", "")).strip()
            right = self._clean_numeric_block(text.get("right", ""))
            valid = bool(re.fullmatch(r"\d{1,6}", left) and self.ARABIC_LETTER_RE.fullmatch(letter) and re.fullmatch(r"\d{1,6}", right))
            return {"raw": " ".join([left, letter, right]).strip(), "left": left, "letter": letter, "right": right, "valid": valid}

        if isinstance(text, (list, tuple)):
            if len(text) != 3:
                return {"raw": str(text), "left": "", "letter": "", "right": "", "valid": False}
            left = self._clean_numeric_block(text[0])
            letter = str(text[1]).strip()
            right = self._clean_numeric_block(text[2])
            valid = bool(re.fullmatch(r"\d{1,6}", left) and self.ARABIC_LETTER_RE.fullmatch(letter) and re.fullmatch(r"\d{1,6}", right))
            return {"raw": " ".join([left, letter, right]).strip(), "left": left, "letter": letter, "right": right, "valid": valid}

        if not text:
            return {"raw": text, "left": "", "letter": "", "right": "", "valid": False}

        s = str(text).strip()
        # Best effort fallback: keep old raw-text approach when no segmentation is provided.
        m = self.MATRICULE_RE.search(s)
        if m:
            left = self._clean_numeric_block(m.group("left"))
            letter = m.group("letter")
            right = self._clean_numeric_block(m.group("right"))
            valid = bool(re.fullmatch(r"\d{1,6}", left) and self.ARABIC_LETTER_RE.fullmatch(letter) and re.fullmatch(r"\d{1,6}", right))
            return {"raw": s, "left": left, "letter": letter, "right": right, "valid": valid}

        # If text looks like three blocks already, parse them in order.
        parts = re.split(r"\s+|[-/|]", s)
        parts = [p for p in parts if p and p.strip()]
        if len(parts) >= 3:
            left = self._clean_numeric_block(parts[0])
            letter = parts[1].strip()
            right = self._clean_numeric_block(parts[2])
            valid = bool(re.fullmatch(r"\d{1,6}", left) and self.ARABIC_LETTER_RE.fullmatch(letter) and re.fullmatch(r"\d{1,6}", right))
            return {"raw": s, "left": left, "letter": letter, "right": right, "valid": valid}

        ar_match = self.ARABIC_RE.search(s)
        if ar_match:
            idx = ar_match.start()
            left = self._clean_numeric_block(s[:idx])
            right = self._clean_numeric_block(s[idx + 1:])
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
          - Run full-image OCR first to collect candidates.
          - If try_segment is provided, route by layout (1-line vs 2-line) before segmenting.
          - Otherwise, attempt OCR-assisted segmentation using the detected Arabic letter bbox.
          - If segmentation yields a valid parsed matricule, return it; else fall back to global classification.
        """
        img = self._load_image(image)
        h, w = img.shape[:2]
        # full-image OCR (candidates for classification and possible Arabic bbox)
        ocr_full = self.read_plate(img, preprocess=preprocess)
        raw_text = " ".join([it["text"] for it in ocr_full])

        # if caller provided a segmenter, route by layout before running segmentation
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
                    return {"raw_text": parsed["raw_text"], "zone_results": zone_results, "parsed": parsed, "ocr_results": ocr_full}
            except Exception:
                pass

        # Try OCR-assisted segmentation: locate best Arabic single-char bbox
        arabic_items = []
        for item in ocr_full:
            txt = str(item.get("text", ""))
            if re.fullmatch(r"[\u0600-\u06FF]{1,2}", txt):
                bbox = item.get("bbox")
                if bbox and len(bbox) >= 1:
                    xs = [int(p[0]) for p in bbox]
                    arabic_items.append((item, min(xs), max(xs)))
        if arabic_items:
            # choose highest confidence arabic item
            arabic_items.sort(key=lambda t: t[0].get("conf", 0.0), reverse=True)
            best, min_x, max_x = arabic_items[0]
            margin = max(6, int(0.02 * w))
            left_sep = max(1, int(min_x) - margin)
            right_sep = min(w - 1, int(max_x) + margin)
            # expand separators slightly to include adjacent digits if needed
            if left_sep < int(0.12 * w):
                left_sep = max(left_sep, int(0.05 * w))
            if right_sep > int(0.88 * w):
                right_sep = min(right_sep, int(0.95 * w))

            left = img[:, :left_sep]
            middle = img[:, left_sep:right_sep]
            right = img[:, right_sep:]
            zone_results = self.read_plate_zones([left, middle, right], preprocess=preprocess)
            texts = [zr['text'] for zr in zone_results]
            parsed = self.parse_matricule({"left": texts[0], "letter": texts[1], "right": texts[2]})
            parsed["raw_text"] = " ".join(texts)
            if parsed.get("valid"):
                return {"raw_text": parsed["raw_text"], "zone_results": zone_results, "parsed": parsed, "ocr_results": ocr_full}

        # Fallback: classify based on OCR tokens across full image
        candidates = self.filter_plate_candidates(ocr_full)
        parsed = self.classify_matricule(candidates)
        parsed["raw_text"] = raw_text
        parsed["candidates"] = candidates
        return {"raw_text": raw_text, "candidates": candidates, "parsed": parsed, "ocr_results": ocr_full}

    def read_and_parse(
        self,
        image: t.Union[str, np.ndarray],
        try_segment: t.Optional[t.Callable[..., t.List[np.ndarray]]] = None,
        debug_dir: t.Optional[str] = None,
    ) -> t.Dict[str, t.Any]:
        """Read a plate in two passes, preserving a valid raw recognition.

        The first pass intentionally uses unmodified pixels.  Only an invalid
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
        return enhanced_result

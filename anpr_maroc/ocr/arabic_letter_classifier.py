import os
import typing as t
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

# Allowed Arabic letters for Moroccan plates (common set) — adjust as needed
ALLOWED_LETTERS = list("أابتصثجحخدذرزسشصضطظعغفقكلمنهويئؤءآإة")

TEMPLATES: t.Dict[str, np.ndarray] = {}


def _find_font() -> t.Optional[str]:
    candidates = [
        "/usr/share/fonts/google-droid-sans-fonts/DroidKufi-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def build_templates(size: int = 64, font_path: t.Optional[str] = None):
    """Render templates for allowed letters into binary images (uint8) sized (size,size).
    Caches templates in module variable TEMPLATES."""
    global TEMPLATES
    if TEMPLATES:
        return TEMPLATES
    if font_path is None:
        font_path = _find_font()
    if font_path is None:
        return {}
    for ch in ALLOWED_LETTERS:
        try:
            img = Image.new("L", (size, size), 255)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_path, max(28, int(size * 0.7)))
            bbox = draw.textbbox((0, 0), ch, font=font)
            x = (size - (bbox[2] - bbox[0])) / 2 - bbox[0]
            y = (size - (bbox[3] - bbox[1])) / 2 - bbox[1]
            draw.text((x, y), ch, fill=0, font=font)
            arr = np.asarray(img)
            _, binar = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            TEMPLATES[ch] = binar
        except Exception:
            continue
    return TEMPLATES


def classify_letter(img: t.Union[str, np.ndarray], templates: t.Optional[t.Dict[str, np.ndarray]] = None) -> t.Tuple[str, float]:
    """Classify an Arabic letter crop using template matching.

    Returns (best_letter, score) where score is normalized correlation in [0..1] (higher better).
    If no template or no match, returns ("", 0.0).
    """
    if templates is None:
        templates = build_templates()
    if not templates:
        return "", 0.0

    if isinstance(img, str):
        arr = cv2.imread(img)
        if arr is None:
            return "", 0.0
    else:
        arr = np.array(img)
    if arr.ndim == 3:
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr

    # crop to content
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ys, xs = np.where(thr > 0)
    if xs.size == 0 or ys.size == 0:
        return "", 0.0
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = thr[y0:y1 + 1, x0:x1 + 1]
    # pad and resize to template size
    pad = 8
    canvas = np.full((crop.shape[0] + 2 * pad, crop.shape[1] + 2 * pad), 255, dtype=np.uint8)
    canvas[pad:pad + crop.shape[0], pad:pad + crop.shape[1]] = crop
    canv = cv2.resize(canvas, (64, 64), interpolation=cv2.INTER_CUBIC)
    best_letter = ""
    best_score = -1.0
    # Prepare a neat binary canvas for shape matching (text black on white)
    try:
        bin_canv = cv2.threshold(255 - canv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    except Exception:
        bin_canv = (255 - canv).astype('uint8')
    # find main contour for candidate
    cnts_canv, _ = cv2.findContours(bin_canv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts_canv:
        return "", 0.0
    # choose largest contour
    cnt_canv = max(cnts_canv, key=cv2.contourArea)

    for ch, templ in templates.items():
        try:
            # templ is binary (black text on white bg) from build_templates
            # extract contours from template
            tmp = templ.copy()
            if tmp.ndim == 3:
                tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
            _, tmp_bin = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cnts_tmp, _ = cv2.findContours(255 - tmp_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts_tmp:
                continue
            cnt_tmp = max(cnts_tmp, key=cv2.contourArea)
            # matchShapes: lower is better (0 == perfect match)
            mscore = cv2.matchShapes(cnt_canv, cnt_tmp, cv2.CONTOURS_MATCH_I1, 0.0)
            # convert to a 0..1-ish similarity (higher better)
            sim = 1.0 / (1.0 + float(mscore))
        except Exception:
            continue
        if sim > best_score:
            best_score = sim
            best_letter = ch
    # accept weaker matches (shape metric is forgiving); threshold tuned experimentally
    if best_score < 0.35:
        return "", float(best_score)
    return best_letter, float(best_score)

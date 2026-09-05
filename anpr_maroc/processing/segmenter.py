import typing as t
import os
import tempfile

import cv2
import numpy as np

# store last bar candidates info for debugging
BAR_CANDIDATES_INFO = []


def _preprocess_plate(image: np.ndarray) -> np.ndarray:
    """Normalize plate image for separator detection."""
    img = image.copy()
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Slight blur to avoid noise on edges.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return gray


def _detect_vertical_bars(gray: np.ndarray, min_height_ratio: float = 0.8) -> t.List[int]:
    """Detect separator bars as narrow, nearly full-height dark groups.

    Uses several signals and an edge-based fallback to handle reflective / noisy plates:
    1) dark vertical groups after adaptive threshold + opening
    2) vertical-edge continuity (Sobel) signal to find long vertical strokes
    3) Hough-lines / projection remain fallbacks elsewhere
    Returns list of x-centers of candidate bars.
    """
    h, w = gray.shape
    # Adaptive threshold using Otsu; fall back to median-based heuristic
    try:
        otsu_th, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        dark_threshold = max(10, int(otsu_th * 0.9))
    except Exception:
        median = float(np.median(gray))
        dark_threshold = max(25, min(200, int(median * 0.75 + 25)))

    # create dark mask and compute per-column dark pixel counts
    dark_mask = (gray < dark_threshold).astype('uint8') * 255

    # Morphological opening to remove small dark spots (screws, reflections)
    try:
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel_open)
    except Exception:
        pass

    col_sum = np.sum(dark_mask > 0, axis=0)
    required_dark_px = max(2, int(h * min_height_ratio))

    x_index = np.arange(w)
    central = (x_index >= 0.08 * w) & (x_index <= 0.92 * w)
    candidates = np.where((col_sum >= required_dark_px) & central)[0]

    separators = []
    # Evaluate dark-groups first
    candidate_info_local = []
    if candidates.size:
        groups = np.split(candidates, np.where(np.diff(candidates) != 1)[0] + 1)
        max_width_thresh = max(200, int(0.18 * w))
        for g in groups:
            if g.size == 0:
                continue
            width = int(g[-1]) - int(g[0]) + 1
            center = int(g.mean())
            # vertical continuity: fraction of rows where any pixel in the group is dark
            vert_cont = 0.0
            try:
                band = dark_mask[:, g[0]:g[-1] + 1]
                vert_cont = float((np.sum(band > 0, axis=1) > 0).sum() / float(h))
            except Exception:
                vert_cont = 0.0
            dark_fraction = float(col_sum[g].mean() / float(h))
            candidate_info_local.append({
                "center": center,
                "left": int(g[0]),
                "right": int(g[-1]),
                "width": width,
                "dark_frac": dark_fraction,
                "vert_cont": vert_cont,
            })
            # Reject overly wide groups (likely background/large dark regions from reflections)
            if width <= max_width_thresh and 1 <= width <= 30 and vert_cont >= max(0.6, min_height_ratio - 0.15):
                separators.append(center)

    # store for outer debug logging
    try:
        global BAR_CANDIDATES_INFO
        BAR_CANDIDATES_INFO = candidate_info_local
    except Exception:
        pass

    # If no reliable dark separators found, fallback to edge-based continuity detection
    if not separators:
        try:
            # vertical gradient magnitude
            sobx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
            mag = cv2.convertScaleAbs(sobx)
            # threshold by high percentile to keep only strong vertical edges
            p = np.percentile(mag, 75)
            thresh_val = max(8, int(p))
            strong = (mag >= thresh_val).astype('uint8') * 255
            # morphological closing with a tall vertical kernel to emphasize continuous vertical strokes
            vert_len = max(3, int(h * 0.5))
            kernel_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_len))
            try:
                closed = cv2.morphologyEx(strong, cv2.MORPH_CLOSE, kernel_vert)
            except Exception:
                closed = strong
            col_sum_edges = np.sum(closed > 0, axis=0)
            edge_frac = col_sum_edges / float(h)
            edge_candidates = np.where((edge_frac >= 0.55) & central)[0]
            if edge_candidates.size:
                groups = np.split(edge_candidates, np.where(np.diff(edge_candidates) != 1)[0] + 1)
                for g in groups:
                    width = int(g[-1]) - int(g[0]) + 1
                    center = int(g.mean())
                    if 1 <= width <= max(40, int(0.05 * w)):
                        separators.append(center)
            # debug info writing
            if debug_dir and edge_candidates.size:
                try:
                    with open(os.path.join(debug_dir, "detection_debug.txt"), "a", encoding="utf-8") as f:
                        f.write(f"edge_thresh={thresh_val} edge_frac_mean={float(edge_frac.mean()):.3f} edge_candidates_count={edge_candidates.size}\n")
                except Exception:
                    pass
        except Exception:
            pass

    # deduplicate and sort
    separators = sorted(list(dict.fromkeys(separators)))
    return separators


def _find_projection_separators(gray: np.ndarray, expected=2) -> t.List[int]:
    """Find separators as wide valleys in the vertical projection of dark pixels.

    Strategy:
      - compute projection of dark pixels per column
      - identify contiguous low-value regions (gaps)
      - choose the widest gaps (longest width) as separators
    """
    h, w = gray.shape
    projection = np.sum(gray < 200, axis=0).astype(float)
    # smooth projection
    kernel = np.ones(max(3, int(w * 0.02))) / max(3, int(w * 0.02))
    sm = np.convolve(projection, kernel, mode="same")

    left_lim = int(0.05 * w)
    right_lim = int(0.95 * w)

    # define threshold for 'low' projection: e.g., < 30% of max
    thresh = max(1.0, 0.3 * sm.max())
    low_mask = sm < thresh

    # only consider central zone
    low_mask[:left_lim] = False
    low_mask[right_lim:] = False

    idxs = np.where(low_mask)[0]
    if idxs.size == 0:
        return []

    groups = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
    gaps = [(int(g[0]), int(g[-1]), int(g[-1]) - int(g[0]) + 1) for g in groups if g.size]
    if not gaps:
        return []

    # sort by width (descending) and pick centers of the largest gaps
    gaps_sorted = sorted(gaps, key=lambda x: x[2], reverse=True)
    picks = []
    for start, end, width in gaps_sorted:
        center = (start + end) // 2
        if not picks or all(abs(center - p) > max(8, int(0.05 * w)) for p in picks):
            picks.append(center)
        if len(picks) >= expected:
            break
    return sorted(picks[:expected])


def _detect_vertical_lines_hough(gray: np.ndarray) -> t.List[t.Tuple[int,int]]:
    """Detect long straight vertical edges using HoughLinesP and return list of (x_center, length).

    Returns empty list if none found.
    """
    h, w = gray.shape
    edges = cv2.Canny(gray, 50, 150)
    minLen = max(10, int(0.5 * h))
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=80, minLineLength=minLen, maxLineGap=20)
    xs = []
    lens = []
    if lines is None:
        return []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        if abs(x1 - x2) <= 6 and abs(y1 - y2) > int(0.4 * h):
            xs.append(int((x1 + x2) / 2))
            lens.append(int(abs(y2 - y1)))
    if not xs:
        return []
    # group nearby xs and average lengths
    xs_sorted_idx = sorted(range(len(xs)), key=lambda i: xs[i])
    groups = []
    current = [xs_sorted_idx[0]]
    for idx in xs_sorted_idx[1:]:
        if abs(xs[idx] - xs[current[-1]]) > 8:
            groups.append(current)
            current = [idx]
        else:
            current.append(idx)
    groups.append(current)
    centers = []
    for g in groups:
        xs_g = [xs[i] for i in g]
        lens_g = [lens[i] for i in g]
        centers.append((int(sum(xs_g) / len(xs_g)), int(sum(lens_g) / len(lens_g))))
    return centers


def _choose_separators(gray: np.ndarray, debug_dir: t.Optional[str] = None) -> t.List[int]:
    # Try bars first and prefer them strongly
    bars = _detect_vertical_bars(gray)
    w = gray.shape[1]
    min_sep = max(8, int(0.06 * w))
    # discard bars that are essentially at image borders (likely plate frame)
    bars = [b for b in bars if 0.15 * w < b < 0.85 * w]

    # optional debug logging for separator detection
    if debug_dir:
        try:
            os.makedirs(debug_dir, exist_ok=True)
            with open(os.path.join(debug_dir, "detection_debug.txt"), "a", encoding="utf-8") as f:
                mean_brightness = float(np.mean(gray))
                try:
                    otsu_th, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                except Exception:
                    otsu_th = 0
                f.write(f"mean_brightness={mean_brightness:.2f} otsu_th={otsu_th:.1f} initial_bars_candidates={bars}\n")
                # Detailed per-bar info (position, width, dark_fraction, vert_cont)
                if BAR_CANDIDATES_INFO:
                    for info in BAR_CANDIDATES_INFO:
                        f.write(f"candidate: pos={info['center']} left={info['left']} right={info['right']} width={info['width']} dark_frac={info['dark_frac']:.3f} vert_cont={info['vert_cont']:.3f}\n")
                # Hough info if present
                try:
                    hough_info = _detect_vertical_lines_hough(gray)
                    if hough_info:
                        # _detect_vertical_lines_hough returns list of tuples (x, length)
                        for x, ln in hough_info:
                            f.write(f"hough_line: x={x} length={ln}\n")
                except Exception:
                    pass
        except Exception:
            pass

    if len(bars) >= 2:
        # Real separator bars are the authoritative signal. Projection gaps are a fallback only when no bars exist.
        # This avoids choosing synthetic gaps caused by digit spacing.
        bars_sorted = sorted(bars)
        if debug_dir:
            # compute and log darkness score per bar candidate (recompute col sums)
            try:
                col_sum = np.sum(gray < max(15, int( (cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0])*0.9 if gray is not None else 50 )), axis=0)
                with open(os.path.join(debug_dir, "detection_debug.txt"), "a", encoding="utf-8") as f:
                    f.write(f"bar_candidates_detail={[]}\n")
            except Exception:
                pass

        if len(bars_sorted) >= 2:
            # Moroccan one-line plates reserve roughly half of the width for
            # the series, then the Arabic letter and region.  The supplied
            # formats place their bars near 50% and 70%, not at equal thirds.
            candidates = []
            for i in range(len(bars_sorted) - 1):
                for j in range(i + 1, len(bars_sorted)):
                    p1, p2 = bars_sorted[i], bars_sorted[j]
                    if abs(p2 - p1) < min_sep:
                        continue
                    score = abs(p1 - 0.50 * w) + abs(p2 - 0.70 * w)
                    candidates.append((score, [p1, p2]))
            if candidates:
                best_pair = sorted(candidates, key=lambda x: x[0])[0][1]
                if debug_dir:
                    with open(os.path.join(debug_dir, "detection_debug.txt"), "a", encoding="utf-8") as f:
                        f.write(f"picked_bars_by_layout={sorted(best_pair)}\n")
                return sorted(best_pair)
            if debug_dir:
                with open(os.path.join(debug_dir, "detection_debug.txt"), "a", encoding="utf-8") as f:
                    f.write(f"picked_bars_raw={bars_sorted[:2]}\n")
            return bars_sorted[:2]
    elif len(bars) == 1:
        # A separator may be slightly shorter than the frame because of a
        # perspective crop (notably ``8.jpg``).  Relax the height threshold
        # only here: the already-confirmed full-height bar anchors the pair,
        # while the usual 80% threshold remains strict for normal detection.
        relaxed_bars = _detect_vertical_bars(gray, min_height_ratio=0.75)
        relaxed_bars = sorted({x for x in relaxed_bars if 0.15 * w < x < 0.85 * w})
        if len(relaxed_bars) >= 2:
            candidates = []
            for i in range(len(relaxed_bars) - 1):
                for j in range(i + 1, len(relaxed_bars)):
                    p1, p2 = relaxed_bars[i], relaxed_bars[j]
                    if abs(p2 - p1) >= min_sep:
                        score = abs(p1 - 0.50 * w) + abs(p2 - 0.70 * w)
                        candidates.append((score, [p1, p2]))
            if candidates:
                best_pair = min(candidates, key=lambda candidate: candidate[0])[1]
                if debug_dir:
                    with open(os.path.join(debug_dir, "detection_debug.txt"), "a", encoding="utf-8") as f:
                        f.write(f"one_bar_relaxed_candidates={relaxed_bars} picked={best_pair}\n")
                return best_pair
        # If only one bar found, try Hough-lines to find another strong vertical line
        hough = _detect_vertical_lines_hough(gray)
        # _detect_vertical_lines_hough returns list of (x,length) tuples — extract x centers
        hough = [x for (x, _) in hough]
        hough = [x for x in hough if 0.08 * w < x < 0.92 * w]
        if debug_dir:
            with open(os.path.join(debug_dir, "detection_debug.txt"), "a", encoding="utf-8") as f:
                f.write(f"one_bar_found={bars[0] if bars else None} hough_candidates={hough}\n")
        if hough:
            other = max(hough, key=lambda x: abs(x - bars[0]))
            return sorted([bars[0], other])
        proj = _find_projection_separators(gray, expected=2)
        if proj:
            other = max(proj, key=lambda x: abs(x - bars[0]))
            return sorted([bars[0], other])

    # Try Hough-lines before projection
    hough = _detect_vertical_lines_hough(gray)
    hough = [x for (x, _) in hough]
    hough = [x for x in hough if 0.08 * w < x < 0.92 * w]
    if len(hough) >= 2:
        # pick two most central hough lines
        if debug_dir:
            with open(os.path.join(debug_dir, "detection_debug.txt"), "a", encoding="utf-8") as f:
                f.write(f"hough_picks={hough[:2]}\n")
        return sorted(hough[:2])

    # No bars: choose the two widest projection gaps (likely splits between blocks)
    proj = _find_projection_separators(gray, expected=2)
    if len(proj) >= 2:
        if debug_dir:
            with open(os.path.join(debug_dir, "detection_debug.txt"), "a", encoding="utf-8") as f:
                f.write(f"projection_picks={proj[:2]}\n")
        return sorted(proj[:2])

    # final fallback equal thirds
    if debug_dir:
        with open(os.path.join(debug_dir, "detection_debug.txt"), "a", encoding="utf-8") as f:
            f.write(f"fallback_thirds={[int(w * 0.33), int(w * 0.66)]}\n")
    return [int(w * 0.33), int(w * 0.66)]


def _detect_plate_contour(image: np.ndarray) -> t.Tuple[np.ndarray, t.Tuple[int,int,int,int]]:
    """Detect the license-plate rectangle robustly and return a crop strictly inside the plate frame.

    Strategy:
    - find contours from Canny edges
    - attempt to approximate a rectangular polygon (4 points)
    - if found, compute a tight bounding box of that polygon and shrink it slightly to stay inside the frame
    - otherwise fall back to previous scoring by bright interior vs border
    """
    img = image.copy()
    h0, w0 = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # Preprocess: bilateral to preserve edges, then Canny
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, (0, 0, w0, h0)

    # Try to find a rectangular contour first (approxPolyDP -> 4 points)
    rect_candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 2000:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            # get bounding rect of the polygon and score by area
            x, y, w, h = cv2.boundingRect(approx)
            ar = w / float(h) if h > 0 else 0
            # Moroccan two-line plates are substantially squarer than the
            # historical one-line formats (``cx.jpeg`` is about 1.5 wide/high).
            # Keep the lower bound above a typical document/photo contour while
            # allowing both layouts through the same, frame-only crop.
            if 1.2 <= ar <= 8.0:
                rect_candidates.append((area, (x, y, w, h), approx))

    if rect_candidates:
        # pick largest rectangle-like contour
        rect_candidates.sort(key=lambda x: x[0], reverse=True)
        _, (rx, ry, rw, rh), approx = rect_candidates[0]
        # shrink slightly to stay strictly inside the frame (remove frame border)
        shrink_x = max(2, int(0.02 * rw))
        shrink_y = max(2, int(0.02 * rh))
        bx = max(0, rx + shrink_x)
        by = max(0, ry + shrink_y)
        bw = max(1, rx + rw - shrink_x - bx)
        bh = max(1, ry + rh - shrink_y - by)
        crop = img[by:by + bh, bx:bx + bw]
        if crop.size == 0:
            return img, (0, 0, w0, h0)
        return crop, (bx, by, bw, bh)

    # fallback: previous scoring method
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 2000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h) if h > 0 else 0
        if not (1.2 <= ar <= 8.0):
            continue
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w0, x + w)
        y1 = min(h0, y + h)
        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue
        mean_in = float(roi.mean())
        pad = max(3, int(min(w, h) * 0.05))
        bx0 = max(0, x0 - pad)
        by0 = max(0, y0 - pad)
        bx1 = min(w0, x1 + pad)
        by1 = min(h0, y1 + pad)
        border = gray[by0:by1, bx0:bx1]
        mean_border = float(border.mean()) if border.size else mean_in
        score = mean_in - mean_border
        candidates.append((score, area, (x0, y0, x1 - x0, y1 - y0)))

    if not candidates:
        return img, (0, 0, w0, h0)

    candidates_sorted = sorted(candidates, key=lambda x: (x[0], x[1]), reverse=True)
    best = candidates_sorted[0]
    bx, by, bw, bh = best[2]
    crop = img[by:by + bh, bx:bx + bw]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if float(crop_gray.mean()) < 160:
        return img, (0, 0, w0, h0)
    # shrink a small margin inside the detected bbox to avoid including frame
    shrink_x = max(2, int(0.01 * bw))
    shrink_y = max(2, int(0.01 * bh))
    bx2 = bx + shrink_x
    by2 = by + shrink_y
    bw2 = max(1, bx + bw - shrink_x - bx2)
    bh2 = max(1, by + bh - shrink_y - by2)
    crop2 = img[by2:by2 + bh2, bx2:bx2 + bw2]
    if crop2.size == 0:
        return crop, (bx, by, bw, bh)
    return crop2, (bx2, by2, bw2, bh2)


def _find_horizontal_separator(gray: np.ndarray) -> int:
    """Return the y-position of a full-width horizontal separator if one exists.

    This is used for 2-line plates where the top and bottom rows are separated by a
    dark, nearly horizontal bar spanning the plate width.
    """
    h, w = gray.shape
    if h <= 0 or w <= 0:
        return h // 2

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark_mask = (blur < max(20, int((255 - thr).mean())))
    # Keep the central band only to avoid border artefacts.
    y0, y1 = int(0.15 * h), int(0.85 * h)
    band = dark_mask[y0:y1, :]
    if band.size == 0:
        return h // 2

    row_ratio = np.sum(band, axis=1) / float(band.shape[1])
    idxs = np.where(row_ratio > 0.55)[0]
    if idxs.size == 0:
        return -1

    groups = np.split(idxs, np.where(np.diff(idxs) != 1)[0] + 1)
    best_group = None
    best_len = 0
    for g in groups:
        if g.size < max(2, int(0.004 * h)):
            continue
        if g.size > best_len:
            best_group = g
            best_len = g.size

    if best_group is None:
        return -1

    y = y0 + int((best_group[0] + best_group[-1]) / 2.0)
    return max(0, min(h - 1, y))


def _find_top_vertical_separator(gray: np.ndarray) -> int:
    """Find the one separator between letter and region on a two-line plate.

    Unlike the one-line format, the top row has *one* internal separator.  The
    generic three-zone helper deliberately searches for two separators, so
    using it here can make the region crop start at a synthetic second split.
    """
    h, w = gray.shape
    if h <= 0 or w <= 0:
        return max(1, w // 2)

    candidates = [x for x in _detect_vertical_bars(gray) if 0.18 * w < x < 0.82 * w]
    if not candidates:
        candidates = [x for x, _ in _detect_vertical_lines_hough(gray) if 0.18 * w < x < 0.82 * w]

    if candidates:
        # The divider is normally near the centre; favour it over glyph strokes
        # that happen to survive the full-height filters.
        return min(candidates, key=lambda x: abs(x - (w / 2.0)))

    # A deterministic safe fallback keeps both fields present when the divider
    # is faint or broken by glare.
    return max(1, min(w - 1, int(w * 0.5)))


def detect_plate_layout(image: np.ndarray) -> str:
    """Detect whether the plate is a 1-line or 2-line layout.

    Returns:
        "1_ligne" or "2_lignes"
    """
    img = image.copy()
    plate, _ = _detect_plate_contour(img)
    if plate is None or plate.size == 0:
        plate = img

    if plate.ndim == 3:
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate

    h, w = gray.shape
    if h <= 0 or w <= 0:
        return "1_ligne"

    split_y = _find_horizontal_separator(gray)
    if split_y < 0:
        return "1_ligne"
    # A 2-line plate has a central dark separator band. A 1-line plate does not.
    if 0.3 * h <= split_y <= 0.85 * h and abs(split_y - (h / 2.0)) <= 0.5 * h:
        return "2_lignes"
    return "1_ligne"


def segment_plate_2lines(image: np.ndarray, margin: int = 6, debug_dir: t.Optional[str] = None) -> t.List[np.ndarray]:
    """Segment a 2-line plate into [series, letter, region] zones.

    Intended layout: top = letter + region (two columns), bottom = series.
    The horizontal separator splits top from bottom; the vertical separator splits
    letter from region inside the top block.
    """
    img = image.copy()
    plate, bbox = _detect_plate_contour(img)
    px, py, pw, ph = bbox
    if plate is None or plate.size == 0:
        plate = img
        pw = img.shape[1]
        ph = img.shape[0]

    if plate.ndim == 3:
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate

    h, w = gray.shape
    if w <= 0:
        raise ValueError("invalid image width")

    split_y = _find_horizontal_separator(gray)
    if split_y == h // 2:
        split_y = max(1, int(h * 0.5))

    top = plate[:split_y, :]
    bottom = plate[split_y:, :]

    top_gray = cv2.cvtColor(top, cv2.COLOR_BGR2GRAY) if top.ndim == 3 else top
    separator_x = _find_top_vertical_separator(top_gray)
    edge_margin = max(1, int(margin / 3))
    left_end = max(1, separator_x - edge_margin)
    right_start = min(top_gray.shape[1] - 1, separator_x + edge_margin)

    series_zone = bottom
    letter_zone = top[:, :left_end]
    region_zone = top[:, right_start:]

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "separators.txt"), "w", encoding="utf-8") as f:
            f.write(f"layout=2_lignes split_y={split_y} top_separator={separator_x}\n")
        cv2.imwrite(os.path.join(debug_dir, "plate.png"), plate)
        cv2.imwrite(os.path.join(debug_dir, "left.png"), series_zone)
        cv2.imwrite(os.path.join(debug_dir, "letter.png"), letter_zone)
        cv2.imwrite(os.path.join(debug_dir, "right.png"), region_zone)
        vis = img.copy()
        cv2.line(vis, (px, py + split_y), (px + pw - 1, py + split_y), (0, 255, 0), 2)
        cv2.line(vis, (px + separator_x, py), (px + separator_x, py + split_y - 1), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "overlay.png"), vis)

    return [series_zone, letter_zone, region_zone]


def segment_plate_by_layout(image: np.ndarray, margin: int = 6, debug_dir: t.Optional[str] = None) -> t.List[np.ndarray]:
    """Route to the proper segmenter based on the detected plate layout."""
    layout = detect_plate_layout(image)
    if layout == "2_lignes":
        return segment_plate_2lines(image, margin=margin, debug_dir=debug_dir)
    return segment_plate(image, margin=margin, debug_dir=debug_dir)


def segment_plate(image: np.ndarray, margin: int = 6, debug_dir: t.Optional[str] = None) -> t.List[np.ndarray]:
    """Adaptive segmentation into left / letter / right using image content.

    Steps:
      - detect plate rectangle in full image (robust to background)
      - run separator detection on the plate crop
      - cut three zones between the two detected separators
    """
    img = image.copy()
    # if the input is a full photo, detect plate region first
    plate, bbox = _detect_plate_contour(img)
    px, py, pw, ph = bbox
    if plate is None or plate.size == 0:
        plate = img
        pw = img.shape[1]
        ph = img.shape[0]

    if plate.ndim == 3:
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate

    h, w = gray.shape
    if w <= 0:
        raise ValueError("invalid image width")

    # choose separators adaptively on the plate crop
    seps = _choose_separators(gray, debug_dir=debug_dir)
    if not seps or len(seps) < 2:
        # try to find separators from narrow full-height dark groups
        bars = _detect_vertical_bars(gray)
        seps = bars if bars else [int(w * 0.33), int(w * 0.66)]
    left_x, right_x = sorted(seps[:2])

    # ensure ordering and boundaries
    left_x = max(1, min(left_x, w - 2))
    right_x = max(left_x + 1, min(right_x, w - 1))

    # Edge margin: slightly smaller margin to avoid including the bar itself in the crop
    edge_margin = max(1, int(margin / 3))

    # compute crop boundaries excluding the vertical bars themselves
    left_end = max(0, left_x - edge_margin)
    middle_start = min(w, left_x + edge_margin)
    middle_end = max(middle_start + 1, right_x - edge_margin)
    right_start = min(w, right_x + edge_margin)

    # Robust fallback when the detected bars are too close or produce a tiny middle region
    if middle_end <= middle_start + 4 or (right_start - middle_start) < max(12, int(0.12 * w)):
        left_end = max(1, int(w * 0.30))
        middle_start = max(left_end + 1, int(w * 0.38))
        middle_end = min(w - 1, int(w * 0.62))
        right_start = max(middle_end + 1, int(w * 0.70))

    # final crops (on plate crop)
    left = plate[:, :left_end]
    middle = plate[:, middle_start:middle_end]
    right = plate[:, right_start:]

    # debug logging: save plate crop + crops and separator positions (global coords)
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        with open(os.path.join(debug_dir, "separators.txt"), "w", encoding="utf-8") as f:
            f.write(f"separators_found={len(seps)} positions={sorted(seps)}\n")
            f.write(f"plate_bbox={bbox} plate_size={(pw,ph)} computed_bounds: left_end={left_end}, middle_start={middle_start}, middle_end={middle_end}, right_start={right_start}\n")
        cv2.imwrite(os.path.join(debug_dir, "plate.png"), plate)
        cv2.imwrite(os.path.join(debug_dir, "left.png"), left)
        cv2.imwrite(os.path.join(debug_dir, "letter.png"), middle)
        cv2.imwrite(os.path.join(debug_dir, "right.png"), right)
        # overlay in full-image coordinates
        vis = img.copy()
        # lines in plate coords -> translate to full image
        cv2.line(vis, (px + left_x, py), (px + left_x, py + ph - 1), (0, 255, 0), 2)
        cv2.line(vis, (px + right_x, py), (px + right_x, py + ph - 1), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_dir, "overlay.png"), vis)

    return [left, middle, right]


def segment_plate_from_path(image_path: str, debug_dir: t.Optional[str] = None) -> t.List[np.ndarray]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    return segment_plate(img, debug_dir=debug_dir)

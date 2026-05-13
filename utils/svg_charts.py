import base64
import numpy as np
from utils.formatting import _da_num, format_big_number

def _make_sparkline_data_url(values, ref_value=None, width=120, height=30):
    """Generér SVG-sparkline med grønne/røde segmenter omkring en reference-linje.

    Hvis ref_value er givet (typisk forrige regular-close), tegnes en stiplet
    sort baseline ved den værdi, og linjen farves grøn over og rød under.
    Zero-crossings interpoleres så farveskift sker præcis på baseline.
    """
    def _empty():
        empty = '<svg xmlns="http://www.w3.org/2000/svg" width="120" height="30"></svg>'
        return f"data:image/svg+xml;base64,{base64.b64encode(empty.encode('utf-8')).decode('ascii')}"

    if not values or len(values) < 2:
        return _empty()

    # Y-range inkluderer ref_value så baselinen altid ligger indenfor plottet
    all_vals = list(values)
    if ref_value is not None:
        all_vals.append(ref_value)
    vmin, vmax = min(all_vals), max(all_vals)
    pad = 1.5
    plot_h = height - 2 * pad
    plot_w = width - 2 * pad
    n = len(values)

    def y_at(v):
        if vmax == vmin:
            return pad + plot_h / 2
        return pad + plot_h - (v - vmin) / (vmax - vmin) * plot_h

    def x_at(idx_frac):
        return pad + idx_frac * plot_w / (n - 1)

    # Hvis ingen ref_value, fallback til simpel grå linje
    if ref_value is None:
        pts = " L ".join(f"{x_at(i):.1f},{y_at(v):.1f}" for i, v in enumerate(values))
        svg = (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
            f'<path d="M {pts}" stroke="#999" stroke-width="1.5" fill="none" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
            f'</svg>'
        )
        return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode('utf-8')).decode('ascii')}"

    ref = float(ref_value)
    ref_y = y_at(ref)

    # Indsæt interpolerede zero-crossings (relativt til ref) så segmenter mødes på baseline
    ex_idx = []   # fraktionel position i values-array
    ey_vals = []  # værdier (inkl. ref ved crossings)
    for i in range(n):
        ex_idx.append(float(i))
        ey_vals.append(float(values[i]))
        if i + 1 < n:
            v0 = values[i] - ref
            v1 = values[i + 1] - ref
            if (v0 > 0 and v1 < 0) or (v0 < 0 and v1 > 0):
                frac = v0 / (v0 - v1)
                ex_idx.append(i + frac)
                ey_vals.append(ref)

    coords = [(x_at(idx), y_at(v)) for idx, v in zip(ex_idx, ey_vals)]

    # Split i kontinuerlige sign-segmenter (over/under ref)
    segments = []
    cur_sign, cur_seg = None, []
    for (x, y), v in zip(coords, ey_vals):
        delta = v - ref
        if delta > 1e-9:
            new_sign = "pos"
        elif delta < -1e-9:
            new_sign = "neg"
        else:
            new_sign = None  # præcis på baseline
        if new_sign is None:
            if cur_seg:
                cur_seg.append((x, y))
                segments.append((cur_sign, cur_seg))
            cur_seg = [(x, y)]
            cur_sign = None
        elif cur_sign is None or cur_sign == new_sign:
            cur_seg.append((x, y))
            cur_sign = new_sign
        else:
            segments.append((cur_sign, cur_seg))
            cur_seg = [(x, y)]
            cur_sign = new_sign
    if cur_seg:
        segments.append((cur_sign, cur_seg))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    ]
    # Først tint-fill, så stroke ovenpå, så stiplet baseline øverst
    for sign, seg in segments:
        if len(seg) < 2 or sign is None:
            continue
        stroke = "#2e7d32" if sign == "pos" else "#d32f2f"
        fill = "rgba(46,125,50,0.15)" if sign == "pos" else "rgba(211,47,47,0.15)"
        # Fill-polygon: line-points + luk til baseline og tilbage
        poly = (
            "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in seg)
            + f" L {seg[-1][0]:.1f},{ref_y:.1f}"
            + f" L {seg[0][0]:.1f},{ref_y:.1f} Z"
        )
        parts.append(f'<path d="{poly}" fill="{fill}" stroke="none"/>')
        line_d = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in seg)
        parts.append(
            f'<path d="{line_d}" stroke="{stroke}" stroke-width="1.5" fill="none" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
    # Stiplet baseline øverst
    parts.append(
        f'<line x1="{pad:.1f}" y1="{ref_y:.1f}" x2="{width - pad:.1f}" y2="{ref_y:.1f}" '
        f'stroke="#000" stroke-width="0.8" stroke-dasharray="2,2" opacity="0.6"/>'
    )
    parts.append('</svg>')
    svg = "".join(parts)
    return f"data:image/svg+xml;base64,{base64.b64encode(svg.encode('utf-8')).decode('ascii')}"


def _make_volume_bar_html(volume, avg_volume, width_px=520):
    """Vandret bar med 'X% VS AVG'-callout, MarketWatch-stil.

    Baren viser dagens volumen som procent af 65d-gennemsnittet (cap'et ved
    150%, men callout-procenten er den faktiske). Returnerer HTML der kan
    sendes til st.markdown(unsafe_allow_html=True)."""
    if not volume or not avg_volume or avg_volume <= 0:
        return (
            "<div style='color:#888; font-size:13px; padding:8px 0;'>"
            "Volumen-data ikke tilgængelig</div>"
        )
    pct = volume / avg_volume * 100
    fill_pct = min(pct, 150) / 150 * 100  # bar er 0..150% visuelt
    callout_left = max(2, min(fill_pct - 6, 88))
    return (
        f"<div style='width:100%; max-width:{width_px}px; padding:8px 0;'>"
        f"  <div style='position:relative; height:38px;'>"
        f"    <div style='position:absolute; top:18px; left:0; width:100%; "
        f"                height:8px; background:#e0e0e0; border-radius:4px;'>"
        f"      <div style='height:100%; width:{fill_pct:.1f}%; "
        f"                  background:#666; border-radius:4px;'></div>"
        f"    </div>"
        f"    <div style='position:absolute; top:0; left:{callout_left:.1f}%; "
        f"                background:#000; color:#fff; padding:2px 8px; "
        f"                border-radius:3px; font-size:11px; font-weight:600;'>"
        f"      {_da_num(pct, decimals=0)}% VS AVG"
        f"    </div>"
        f"  </div>"
        f"  <div style='display:flex; justify-content:space-between; "
        f"              margin-top:6px; font-size:11px; color:#555;'>"
        f"    <span><strong>VOLUMEN:</strong> {format_big_number(volume)}</span>"
        f"    <span>↑ 65d-snit: <strong>{format_big_number(avg_volume)}</strong></span>"
        f"  </div>"
        f"</div>"
    )


def _make_range_bar_html(low, high, marker_low=None, marker_high=None,
                        marker_low_label="OPEN", marker_high_label="LAST",
                        bottom_label="DAY LOW/HIGH",
                        currency_symbol="$", width_px=520, segment_fill=False):
    """Vandret range-bar med to markører på en min..max skala.

    Hvis segment_fill=True: et farvet segment fra marker_low til marker_high
    (bruges til 52w-range hvor 'DAY RANGE' fremhæves som et segment).
    Ellers: to separate callout-tags på baren ved hver markør-position."""
    if low is None or high is None or low >= high:
        return (
            "<div style='color:#888; font-size:13px; padding:8px 0;'>"
            f"{bottom_label}: ikke tilgængelig</div>"
        )

    span = high - low

    def _pos(v):
        if v is None:
            return None
        return max(0.0, min(100.0, (v - low) / span * 100))

    pos_lo = _pos(marker_low)
    pos_hi = _pos(marker_high)

    if segment_fill and pos_lo is not None and pos_hi is not None:
        seg_left = min(pos_lo, pos_hi)
        seg_width = abs(pos_hi - pos_lo)
        segment_html = (
            f"<div style='position:absolute; top:18px; left:{seg_left:.1f}%; "
            f"            width:{max(seg_width, 1.5):.1f}%; height:8px; "
            f"            background:#c62828; border-radius:2px;'></div>"
        )
        callouts_html = (
            f"<div style='position:absolute; top:0; "
            f"            left:{min(max(seg_left + seg_width / 2 - 6, 2), 88):.1f}%; "
            f"            background:#000; color:#fff; padding:2px 8px; "
            f"            border-radius:3px; font-size:11px; font-weight:600;'>"
            f"  {bottom_label.upper()}"
            f"</div>"
        )
    else:
        segment_html = ""
        callouts = []
        if pos_lo is not None and marker_low is not None:
            cl = max(2, min(pos_lo - 6, 88))
            callouts.append(
                f"<div style='position:absolute; top:0; left:{cl:.1f}%; "
                f"            background:#1976d2; color:#fff; padding:2px 8px; "
                f"            border-radius:3px; font-size:11px; font-weight:600; "
                f"            white-space:nowrap;'>"
                f"  {marker_low_label}: {currency_symbol}{_da_num(marker_low)}"
                f"</div>"
                f"<div style='position:absolute; top:14px; left:{pos_lo:.1f}%; "
                f"            width:2px; height:16px; background:#1976d2;'></div>"
            )
        if pos_hi is not None and marker_high is not None:
            cl = max(2, min(pos_hi - 6, 88))
            callouts.append(
                f"<div style='position:absolute; top:0; left:{cl:.1f}%; "
                f"            background:#000; color:#fff; padding:2px 8px; "
                f"            border-radius:3px; font-size:11px; font-weight:600; "
                f"            white-space:nowrap;'>"
                f"  {marker_high_label}: {currency_symbol}{_da_num(marker_high)}"
                f"</div>"
                f"<div style='position:absolute; top:14px; left:{pos_hi:.1f}%; "
                f"            width:2px; height:16px; background:#000;'></div>"
            )
        callouts_html = "".join(callouts)

    return (
        f"<div style='width:100%; max-width:{width_px}px; padding:14px 0 8px;'>"
        f"  <div style='position:relative; height:38px;'>"
        f"    <div style='position:absolute; top:18px; left:0; width:100%; "
        f"                height:8px; background:#e8e8e8; border-radius:4px;'></div>"
        f"    {segment_html}"
        f"    {callouts_html}"
        f"  </div>"
        f"  <div style='display:flex; justify-content:space-between; "
        f"              margin-top:6px; font-size:11px; color:#555;'>"
        f"    <span>{currency_symbol}{_da_num(low)}</span>"
        f"    <span style='color:#888;'>{bottom_label}</span>"
        f"    <span>{currency_symbol}{_da_num(high)}</span>"
        f"  </div>"
        f"</div>"
    )
import base64
import datetime
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


def _callout_with_pointer(box_left, pointer_pct, text, bg_color,
                          box_top=0, tri_top=21, point_down=True):
    """MarketWatch-stil callout-boks med trekant der peger på baren.

    box_left: boksens venstre kant i % (clampes af kalderen).
    pointer_pct: trekantens præcise position i %.
    point_down=True: boks over baren, trekant peger nedad; ellers boks
    under baren, trekant peger opad."""
    if point_down:
        tri_borders = (
            "border-left:6px solid transparent; "
            "border-right:6px solid transparent; "
            f"border-top:6px solid {bg_color};"
        )
    else:
        tri_borders = (
            "border-left:6px solid transparent; "
            "border-right:6px solid transparent; "
            f"border-bottom:6px solid {bg_color};"
        )
    return (
        f"<div style='position:absolute; top:{box_top}px; left:{box_left:.1f}%; "
        f"            background:{bg_color}; color:#fff; padding:3px 9px; "
        f"            border-radius:3px; font-size:12px; font-weight:700; "
        f"            white-space:nowrap;'>{text}</div>"
        f"<div style='position:absolute; top:{tri_top}px; left:{pointer_pct:.1f}%; "
        f"            width:0; height:0; transform:translateX(-6px); "
        f"            {tri_borders}'></div>"
    )


def _make_volume_bar_html(volume, avg_volume):
    """Vandret volumen-bar med 'X% VS AVG'-callout, MarketWatch-stil.

    Baren viser dagens volumen som procent af 65d-gennemsnittet (cap'et ved
    150% visuelt, men callout-procenten er den faktiske). Returnerer HTML der
    kan sendes til st.markdown(unsafe_allow_html=True)."""
    if not volume or not avg_volume or avg_volume <= 0:
        return (
            "<div style='color:#888; font-size:13px; padding:8px 0;'>"
            "Volumen-data ikke tilgængelig</div>"
        )
    pct = volume / avg_volume * 100
    fill_pct = min(pct, 150) / 150 * 100  # bar er 0..150% visuelt
    box_left = max(0.0, min(fill_pct - 7.0, 82.0))
    callout = _callout_with_pointer(
        box_left, fill_pct, f"{_da_num(pct, decimals=0)}% VS AVG", "#1a1a1a"
    )
    return (
        "<div style='width:100%; padding:10px 0;'>"
        "  <div style='position:relative; height:48px;'>"
        f"    {callout}"
        "    <div style='position:absolute; top:28px; left:0; width:100%; "
        "                height:14px; background:#e6e6e6; border-radius:7px;'>"
        f"      <div style='height:100%; width:{fill_pct:.1f}%; "
        f"                  background:#7d7d7d; border-radius:7px;'></div>"
        "    </div>"
        "  </div>"
        "  <div style='display:flex; justify-content:space-between; "
        "              margin-top:8px; font-size:12px; color:#444;'>"
        f"    <span><strong>VOLUMEN:</strong> {format_big_number(volume)}</span>"
        f"    <span>↑ 65d-snit: <strong>{format_big_number(avg_volume)}</strong></span>"
        "  </div>"
        "</div>"
    )


def _make_range_bar_html(low, high, marker_low=None, marker_high=None,
                         marker_low_label="OPEN", marker_high_label="LAST",
                         bottom_label="DAY LOW/HIGH", segment_label=None,
                         currency_symbol="$", segment_fill=False,
                         track_color="#e8e8e8"):
    """Vandret range-bar med markører på en min..max skala, MarketWatch-stil.

    segment_fill=True: en rød markør fra marker_low til marker_high med en
    enkelt callout over baren (52w-baren; segment_label er callout-teksten).
    Ellers (day-baren): marker_low som callout OVER baren og marker_high
    UNDER baren, så de aldrig overlapper når kurserne ligger tæt."""
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

    def _clamp(p):
        return max(0.0, min(p - 7.0, 82.0))

    pos_lo = _pos(marker_low)
    pos_hi = _pos(marker_high)

    if segment_fill and pos_lo is not None and pos_hi is not None:
        # 52w-bar: rød day-range-markør + enkelt callout over baren
        seg_left = min(pos_lo, pos_hi)
        seg_width = max(abs(pos_hi - pos_lo), 1.2)
        segment_html = (
            f"<div style='position:absolute; top:24px; left:{seg_left:.1f}%; "
            f"            width:{seg_width:.1f}%; height:22px; "
            f"            background:#c62828; border-radius:2px;'></div>"
        )
        seg_center = seg_left + seg_width / 2
        seg_text = segment_label if segment_label is not None else bottom_label.upper()
        callouts_html = _callout_with_pointer(
            _clamp(seg_center), seg_center, seg_text, "#1a1a1a",
        )
        container_height = 50
        track_top = 28
    else:
        # day-bar: marker_low over baren, marker_high under baren
        segment_html = ""
        callouts = []
        if pos_lo is not None and marker_low is not None:
            callouts.append(_callout_with_pointer(
                _clamp(pos_lo), pos_lo,
                f"{marker_low_label}: {currency_symbol}{_da_num(marker_low)}",
                "#1976d2", box_top=0, tri_top=21, point_down=True,
            ))
        if pos_hi is not None and marker_high is not None:
            callouts.append(_callout_with_pointer(
                _clamp(pos_hi), pos_hi,
                f"{marker_high_label}: {currency_symbol}{_da_num(marker_high)}",
                "#1a1a1a", box_top=51, tri_top=45, point_down=False,
            ))
        callouts_html = "".join(callouts)
        container_height = 74
        track_top = 30

    return (
        "<div style='width:100%; padding:14px 0 8px;'>"
        f"  <div style='position:relative; height:{container_height}px;'>"
        f"    <div style='position:absolute; top:{track_top}px; left:0; "
        f"                width:100%; height:14px; background:{track_color}; "
        f"                border-radius:7px;'></div>"
        f"    {segment_html}"
        f"    {callouts_html}"
        "  </div>"
        "  <div style='display:flex; justify-content:space-between; "
        "              margin-top:6px; font-size:12px; color:#444;'>"
        f"    <span>{currency_symbol}{_da_num(low)}</span>"
        f"    <span style='color:#888;'>{bottom_label}</span>"
        f"    <span>{currency_symbol}{_da_num(high)}</span>"
        "  </div>"
        "</div>"
    )


def _make_performance_bars_html(rows):
    """MarketWatch-stil performance-liste: label, procent og en vandret bar
    pr. periode. rows: liste af (label, pct) — pct kan være None ("—").
    Bar-længde er relativ til den største |pct| i listen; grøn for positiv,
    rød for negativ."""
    vals = [abs(p) for _, p in rows if p is not None]
    max_abs = max(vals) if vals else 1.0
    if max_abs <= 0:
        max_abs = 1.0

    out = ["<div style='width:100%;'>"]
    for label, pct in rows:
        if pct is None:
            value_cell = "<span style='color:#888;'>—</span>"
            bar_fill = ""
        else:
            color = "#2e7d32" if pct >= 0 else "#d32f2f"
            width = min(abs(pct) / max_abs * 100, 100)
            value_cell = (
                f"<span style='color:{color}; font-weight:700;'>"
                f"{_da_num(pct, signed=True)}%</span>"
            )
            bar_fill = (
                f"<div style='height:100%; width:{width:.1f}%; "
                f"background:{color}; border-radius:3px;'></div>"
            )
        out.append(
            f"<div style='display:flex; align-items:center; gap:10px; "
            f"            padding:9px 0; border-bottom:1px solid #eee;'>"
            f"  <span style='flex:1; font-size:12px; font-weight:700; "
            f"               color:#555; letter-spacing:0.4px;'>{label}</span>"
            f"  <span style='flex:0 0 78px; font-size:14px; "
            f"               text-align:right;'>{value_cell}</span>"
            f"  <div style='flex:0 0 42%; height:12px; background:#f0f0f0; "
            f"              border-radius:3px;'>{bar_fill}</div>"
            f"</div>"
        )
    out.append("</div>")
    return "".join(out)


# Danske rec-labels — bemærk: yfinance's kategori 'sell' svarer til Yahoos
# UI-label "Underperform", og 'strongSell' til "Sell" (samme som Yahoo).
_REC_LABELS = {
    "strongBuy": "Stærkt køb",
    "buy": "Køb",
    "hold": "Hold",
    "sell": "Underpræstation",
    "strongSell": "Sælg",
}
_REC_COLORS = {
    "strongBuy": "#1b5e20",
    "buy": "#66bb6a",
    "hold": "#ffc107",
    "sell": "#ef6c00",
    "strongSell": "#d32f2f",
}


def _make_analyst_price_target_html(low, avg, high, current,
                                    currency_symbol="$", analyst_count=0,
                                    recommendation_key=""):
    """Vandret analytiker-kursmål-bar (Yahoo Finance-stil).

    Track fra low til high; en blå 'Gennemsnit'-callout over baren, en
    mørk 'Aktuel'-callout under baren, og et blåt segment mellem de to
    markører. Returnerer tom streng hvis kritiske tal mangler eller
    low >= high (så kalderen kan skjule blokken)."""
    if low is None or avg is None or high is None or current is None:
        return ""
    if low >= high:
        return ""

    span = high - low

    def _pos(v):
        return max(0.0, min(100.0, (v - low) / span * 100))

    def _clamp(p):
        return max(0.0, min(p - 7.0, 82.0))

    pos_cur = _pos(current)
    pos_avg = _pos(avg)
    seg_left = min(pos_cur, pos_avg)
    seg_width = max(abs(pos_avg - pos_cur), 0.6)

    rec_key = (recommendation_key or "").strip().lower().replace("_", "")
    rec_label = {
        "strongbuy": "Stærkt køb", "buy": "Køb", "hold": "Hold",
        "underperform": "Underpræstation", "sell": "Sælg",
        "strongsell": "Stærkt salg",
    }.get(rec_key, "")

    avg_callout = _callout_with_pointer(
        _clamp(pos_avg), pos_avg,
        f"{currency_symbol}{_da_num(avg)}"
        f"<span style='font-weight:400; font-size:10px; "
        f"display:block;'>Gennemsnit</span>",
        "#1976d2", box_top=0, tri_top=38, point_down=True,
    )
    cur_callout = _callout_with_pointer(
        _clamp(pos_cur), pos_cur,
        f"{currency_symbol}{_da_num(current)}"
        f"<span style='font-weight:400; font-size:10px; "
        f"display:block;'>Aktuel</span>",
        "#333333", box_top=59, tri_top=53, point_down=False,
    )

    caption_bits = []
    if analyst_count:
        caption_bits.append(f"{analyst_count} analytikere")
    if rec_label:
        caption_bits.append(rec_label)
    caption = "  ·  ".join(caption_bits)

    # Linje under baren: gennemsnitligt kursmål + afstand til aktuel kurs.
    if current and current > 0:
        _upside = (avg - current) / current * 100
        _up_color = "#2e7d32" if _upside >= 0 else "#d32f2f"
        avg_line = (
            "<div style='font-size:15px; color:#222; margin-top:12px; "
            "text-align:center; font-weight:600;'>"
            f"Gns. kursmål: <strong>{currency_symbol}{_da_num(avg)}</strong> "
            f"<span style='color:{_up_color}; font-weight:700;'>"
            f"({_da_num(_upside, signed=True)}%)</span>"
            "</div>"
        )
    else:
        avg_line = ""

    return (
        "<div style='width:100%; padding:14px 0 8px;'>"
        "  <div style='font-size:13px; font-weight:700; color:#888; "
        "              letter-spacing:0.5px;'>ANALYTIKERNES KURSMÅL</div>"
        "  <div style='position:relative; height:104px; margin-top:6px;'>"
        "    <div style='position:absolute; top:44px; left:0; width:100%; "
        "                height:8px; background:#d6e4f0; "
        "                border-radius:4px;'></div>"
        f"    <div style='position:absolute; top:44px; left:{seg_left:.1f}%; "
        f"                width:{seg_width:.1f}%; height:8px; "
        f"                background:#1976d2; opacity:0.4; "
        f"                border-radius:4px;'></div>"
        f"    <div style='position:absolute; top:48px; left:{pos_avg:.1f}%; "
        f"                transform:translate(-50%,-50%); width:13px; "
        f"                height:13px; border-radius:50%; background:#fff; "
        f"                border:2.5px solid #1976d2;'></div>"
        f"    <div style='position:absolute; top:48px; left:{pos_cur:.1f}%; "
        f"                transform:translate(-50%,-50%); width:12px; "
        f"                height:12px; border-radius:50%; "
        f"                background:#333;'></div>"
        f"    {avg_callout}"
        f"    {cur_callout}"
        "  </div>"
        "  <div style='display:flex; justify-content:space-between; "
        "              font-size:12px; color:#444;'>"
        f"    <span>{currency_symbol}{_da_num(low)}</span>"
        f"    <span style='color:#888; font-size:11px;'>{caption}</span>"
        f"    <span>{currency_symbol}{_da_num(high)}</span>"
        "  </div>"
        f"  {avg_line}"
        "</div>"
    )


def _make_analyst_rec_bars_html(rec_df, max_months=4):
    """Stablet søjlediagram over analytiker-anbefalinger pr. måned
    (Yahoo Finance-stil).

    rec_df: DataFrame fra fetch_analyst_recommendations med kolonnerne
    period, strongBuy, buy, hold, sell, strongSell. Viser de seneste
    max_months perioder med ældste måned til venstre. Returnerer tom
    streng hvis data mangler eller skemaet er uventet."""
    cats = ["strongBuy", "buy", "hold", "sell", "strongSell"]
    if rec_df is None or getattr(rec_df, "empty", True):
        return ""
    if not all(c in rec_df.columns for c in cats + ["period"]):
        return ""

    da_months = ["Jan", "Feb", "Mar", "Apr", "Maj", "Jun",
                 "Jul", "Aug", "Sep", "Okt", "Nov", "Dec"]
    cur_month = datetime.date.today().month  # 1-12

    def _offset(period_str):
        try:
            return int(str(period_str).strip().rstrip("m") or 0)
        except (TypeError, ValueError):
            return 0

    def _count(v):
        try:
            return int(v) if v == v else 0  # v != v er True for NaN
        except (TypeError, ValueError):
            return 0

    # yfinance er ikke garanteret sorteret; sortér på period-offset så
    # ældste måned havner til venstre uanset kildens rækkefølge.
    recs = sorted(
        ((_offset(r.get("period")), r) for _, r in rec_df.iterrows()),
        key=lambda x: x[0],
    )[-max_months:]
    if not recs:
        return ""

    bars = []
    for offset, row in recs:
        counts = {c: _count(row.get(c)) for c in cats}
        total = sum(counts.values())
        layers = []
        # column-reverse: første div nederst → strongSell i bunden,
        # strongBuy i toppen (matcher Yahoos stabling).
        for c in ["strongSell", "sell", "hold", "buy", "strongBuy"]:
            if counts[c] <= 0 or total <= 0:
                continue
            h = counts[c] / total * 100
            bg = _REC_COLORS[c]
            lbl = _REC_LABELS[c]
            # Antallet vises kun hvis segmentet er højt nok til at rumme det.
            txt = str(counts[c]) if h >= 12 else ""
            txt_color = "#333333" if c == "hold" else "#ffffff"
            layers.append(
                f"<div style='width:100%; height:{h:.2f}%; background:{bg}; "
                f"display:flex; align-items:center; justify-content:center; "
                f"color:{txt_color}; font-size:11px; font-weight:600;' "
                f"title='{lbl}: {counts[c]}'>{txt}</div>"
            )
        bar_inner = "".join(layers) if layers else (
            "<div style='width:100%; height:100%; "
            "background:#eeeeee;'></div>"
        )
        month = da_months[(cur_month - 1 + offset) % 12]
        bars.append(
            "<div style='display:flex; flex-direction:column; "
            "            align-items:center; flex:1;'>"
            f"  <span style='font-size:11px; color:#555; "
            f"               margin-bottom:4px;'>{total}</span>"
            "  <div style='width:100%; max-width:46px; height:120px; "
            "              display:flex; flex-direction:column-reverse; "
            "              border-radius:3px; overflow:hidden; "
            f"              background:#f0f0f0;'>{bar_inner}</div>"
            f"  <span style='font-size:11px; color:#666; "
            f"               margin-top:4px;'>{month}</span>"
            "</div>"
        )

    # Procentfordeling for seneste måned (recs er sorteret ældste→nyeste).
    _, _latest_row = recs[-1]
    _latest_counts = {c: _count(_latest_row.get(c)) for c in cats}
    _latest_total = sum(_latest_counts.values())

    def _legend_pct(c):
        if _latest_total <= 0:
            return ""
        pct = _latest_counts[c] / _latest_total * 100
        return f" ({_da_num(pct, decimals=0)}%)"

    legend = "".join(
        "<div style='display:flex; align-items:center; gap:5px; "
        "            font-size:11px; color:#444; white-space:nowrap;'>"
        f"  <div style='width:10px; height:10px; border-radius:2px; "
        f"              background:{_REC_COLORS[c]};'></div>"
        f"  {_REC_LABELS[c]}{_legend_pct(c)}</div>"
        for c in cats
    )

    return (
        "<div style='width:100%; padding:14px 0 8px;'>"
        "  <div style='font-size:13px; font-weight:700; color:#888; "
        "              letter-spacing:0.5px; "
        "              padding-bottom:8px;'>ANALYTIKERNES ANBEFALINGER</div>"
        "  <div style='display:flex; align-items:flex-end; gap:14px;'>"
        "    <div style='flex:1; display:flex; align-items:flex-end; "
        f"                gap:10px;'>{''.join(bars)}</div>"
        "    <div style='display:flex; flex-direction:column; "
        f"                gap:5px;'>{legend}</div>"
        "  </div>"
        "</div>"
    )
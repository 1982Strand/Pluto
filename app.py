"""
Pluto Portefølje — Streamlit app
Korrekt afkastberegning efter Plutos princip:
  • Tidsvægtet afkast (TWR) — industri-standard performance-måling
  • Korrigeret for ind- og udbetalinger
  • Omregnet til DKK
  • Gebyrer automatisk inkluderet (de er allerede fratrukket i cashflows)
"""
import base64
import json
import os
from datetime import date, time, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# -------------------- KONFIGURATION --------------------
st.set_page_config(layout="wide", page_title="Pluto Portefølje", page_icon="📈")
from config import *
from styles import inject_styles
inject_styles()

# -------------------- HJÆLPEFUNKTIONER --------------------
from utils.formatting import _safe_float, _da_num, format_currency, format_big_number, format_quantity, _flatten_html, _CCY_SYMBOLS
from data.deposits import load_deposit_times, save_deposit_times, _deposit_key, _time_to_frac, _time_to_frac
from data.fetch import fetch_price_history, fetch_price_history_intraday, fetch_live_quotes, fetch_live_fx_rates, fetch_ticker_meta


def _time_to_frac(hhmm):
    """Konverter 'HH:MM' eller datetime.time til fraktion af døgn (0..1)."""
    if isinstance(hhmm, time):
        return (hhmm.hour * 60 + hhmm.minute) / (24 * 60)
    if isinstance(hhmm, str) and ":" in hhmm:
        try:
            h, m = hhmm.split(":")[:2]
            return (int(h) * 60 + int(m)) / (24 * 60)
        except (ValueError, IndexError):
            pass
    return 9 / 24  # default 09:00


def _lighten_rgb(rgb_str, factor=0.45):
    """Bland 'rgb(r, g, b)' med hvid for at lave en lysere variant til sunburst-yderring."""
    if not isinstance(rgb_str, str) or not rgb_str.startswith("rgb"):
        return rgb_str
    try:
        nums = rgb_str.replace("rgb(", "").replace(")", "").split(",")
        r, g, b = (int(n.strip()) for n in nums[:3])
    except Exception:
        return rgb_str
    return (
        f"rgb({int(r + (255 - r) * factor)},"
        f"{int(g + (255 - g) * factor)},"
        f"{int(b + (255 - b) * factor)})"
    )


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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_intraday_sparklines(tickers_tuple):
    """For hver ticker: liste af 5-min closes for seneste regular session
    (09:30 - 16:00 ET) — afsluttet eller igangværende. Pre/after-hours
    filtreres væk så sparklinen matcher 'Sidste luk' som er regular-
    session-close. I pre-market falder target tilbage til gårsdagens
    session, da dagens dato endnu ikke har regular-hours-bars."""
    if not tickers_tuple:
        return {}
    out = {t: [] for t in tickers_tuple}
    try:
        prices_i = fetch_price_history_intraday(tickers_tuple, "5d", "5m")
    except Exception:
        return out
    if prices_i is None or prices_i.empty:
        return out
    idx = pd.DatetimeIndex(prices_i.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_et = idx.tz_convert("America/New_York")
    et_dates = idx_et.normalize()
    # Find seneste ET-dato der HAR regular-hours-bars. I pre-market vil
    # dagens dato ikke have nogen regular-bars endnu, så target falder
    # naturligt tilbage til seneste afsluttede handelsdag (typisk gårsdag,
    # eller fredag hvis vi er mandag-morgen).
    et_minutes = idx_et.hour * 60 + idx_et.minute
    regular_mask = (et_minutes >= 9 * 60 + 30) & (et_minutes < 16 * 60)
    weekday_mask = et_dates.weekday < 5
    candidate_dates = et_dates[regular_mask & weekday_mask]
    if len(candidate_dates) == 0:
        return out
    target = candidate_dates.max()
    mask = (et_dates == target) & regular_mask
    sliced = prices_i.loc[mask]
    for t in tickers_tuple:
        if t in sliced.columns:
            series = sliced[t].dropna()
            if len(series) >= 2:
                out[t] = [float(v) for v in series.tolist()]
    return out


def render_breakdown(rows, color_map, *, count_word="positioner", chart_height=320):
    """Render donut + farvet HTML-tabel for en bucket-fordeling.

    rows: liste af dicts med: name, count, value_dkk, invested_dkk, gain_dkk,
          gain_pct, alloc_pct, optional emoji
    color_map: dict[name -> rgb-streng]
    """
    col_chart, col_list = st.columns([1, 2])
    with col_chart:
        fig = go.Figure(go.Pie(
            labels=[r["name"] for r in rows],
            values=[r["value_dkk"] for r in rows],
            hole=0.6,
            marker=dict(
                colors=[color_map[r["name"]] for r in rows],
                line=dict(color="white", width=2),
            ),
            textinfo="none",
            hovertemplate="<b>%{label}</b><br>%{value:,.2f} DKK (%{percent})<extra></extra>",
        ))
        fig.update_layout(
            height=chart_height, showlegend=False,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="white", plot_bgcolor="white",
            separators=".,",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_list:
        rows_html = ""
        for r in rows:
            c = color_map[r["name"]]
            gain_color = "#2e7d32" if r["gain_dkk"] >= 0 else "#d32f2f"
            arrow = "▲" if r["gain_dkk"] >= 0 else "▼"
            if abs(r["gain_dkk"]) < 0.01 and r.get("hide_zero_gain"):
                gain_html = "<small style='color:#888;'>—</small>"
            else:
                gain_html = (
                    f"{_da_num(r['gain_dkk'], signed=True)}<br>"
                    f"<small>{arrow} {_da_num(r['gain_pct'], signed=True)}%</small>"
                )
            emoji = (r.get("emoji", "") + " ") if r.get("emoji") else ""
            rows_html += f"""
            <tr>
              <td style='border-left:4px solid {c}; padding:10px 0 10px 14px;'>
                <strong>{emoji}{r['name']}</strong><br>
                <small style='color:#888;'>{r['count']} {count_word}</small>
              </td>
              <td style='padding:10px;'>
                <strong>{_da_num(r['value_dkk'])}</strong><br>
                <small style='color:#888;'>{_da_num(r['invested_dkk'])}</small>
              </td>
              <td style='padding:10px; color:{gain_color};'>
                {gain_html}
              </td>
              <td style='padding:10px; text-align:right;'>
                <strong>{_da_num(r['alloc_pct'])}%</strong>
              </td>
            </tr>
            """
        table_html = f"""
        <table style='width:100%; border-collapse:collapse; font-size:14px;'>
          <thead>
            <tr style='border-bottom:1px solid #ddd; color:#666;'>
              <th style='text-align:left; padding:8px 0 8px 14px;'>Navn</th>
              <th style='text-align:left; padding:8px;'>Værdi/Investeret</th>
              <th style='text-align:left; padding:8px;'>Gevinst</th>
              <th style='text-align:right; padding:8px;'>Allokering</th>
            </tr>
          </thead>
          <tbody>{rows_html}</tbody>
        </table>
        """
        st.markdown(_flatten_html(table_html), unsafe_allow_html=True)


def render_drilldown(rows, buckets_dict, color_map, *, count_word="positioner",
                     bucket_value_total_key="value_dkk"):
    """Render expander pr. bucket med tabel over underliggende positioner."""
    st.write("")
    st.caption(f"Klik for at se de underliggende {count_word}")
    for r in rows:
        bucket = buckets_dict.get(r["name"])
        if not bucket or not bucket.get("positions"):
            continue
        positions = sorted(bucket["positions"], key=lambda p: p["value_dkk"], reverse=True)
        bucket_total = bucket.get(bucket_value_total_key, 0) or 0
        color = color_map.get(r["name"], "rgb(150,150,150)")
        emoji = (r.get("emoji", "") + " ") if r.get("emoji") else ""
        label = (
            f"{emoji}{r['name']} — {r['count']} {count_word}  •  "
            f"{_da_num(r['value_dkk'])} DKK  •  "
            f"{_da_num(r['alloc_pct'])}%"
        )
        with st.expander(label):
            pos_rows_html = ""
            for p in positions:
                gc = "#2e7d32" if p["gain_dkk"] >= 0 else "#d32f2f"
                share = (p["value_dkk"] / bucket_total * 100) if bucket_total else 0
                qty_str = format_quantity(p.get("qty"))
                if abs(p["gain_dkk"]) < 0.01 and p.get("hide_zero_gain"):
                    gain_cell = "<small style='color:#888;'>—</small>"
                else:
                    gain_cell = (
                        f"{_da_num(p['gain_dkk'], signed=True)}<br>"
                        f"<small>{_da_num(p.get('pos_pct', 0), signed=True)}%</small>"
                    )
                pos_rows_html += f"""
                <tr>
                  <td style='border-left:4px solid {color}; padding:8px 0 8px 14px;'>
                    <strong>{p['name'][:40]}</strong><br>
                    <small style='color:#888;'>{p['ticker']} • {qty_str} stk.</small>
                  </td>
                  <td style='padding:8px;'>
                    <strong>{_da_num(p['value_dkk'])}</strong><br>
                    <small style='color:#888;'>{_da_num(p['invested_dkk'])}</small>
                  </td>
                  <td style='padding:8px; color:{gc};'>
                    {gain_cell}
                  </td>
                  <td style='padding:8px; text-align:right;'>
                    <strong>{_da_num(share)}%</strong>
                  </td>
                </tr>
                """
            pos_table_html = f"""
            <table style='width:100%; border-collapse:collapse; font-size:13px;'>
              <thead>
                <tr style='border-bottom:1px solid #ddd; color:#666;'>
                  <th style='text-align:left; padding:6px 0 6px 14px;'>Position</th>
                  <th style='text-align:left; padding:6px;'>Værdi/Investeret</th>
                  <th style='text-align:left; padding:6px;'>Gevinst</th>
                  <th style='text-align:right; padding:6px;'>Andel af gruppe</th>
                </tr>
              </thead>
              <tbody>{pos_rows_html}</tbody>
            </table>
            """
            st.markdown(_flatten_html(pos_table_html), unsafe_allow_html=True)


def get_us_market_status():
    """
    Returnerer (kode, label, emoji, baggrundsfarve) baseret på aktuel ET-tid.
    Bemærk: Tjekker kun weekender — ikke amerikanske helligdage.
    """
    try:
        et_now = pd.Timestamp.now(tz="America/New_York")
    except Exception:
        # Fallback hvis tz-data ikke er tilgængelig: antag dansk tid - 6 timer
        et_now = pd.Timestamp.now() - pd.Timedelta(hours=6)

    # Weekend
    if et_now.weekday() >= 5:
        return ("weekend", "Lukket (weekend)", "🌙", "#f0f2f6")

    minutes = et_now.hour * 60 + et_now.minute

    if minutes < 4 * 60:                     # 00:00 - 04:00 ET
        return ("overnight", "Overnight (lukket)", "🌙", "#f0f2f6")
    elif minutes < 9 * 60 + 30:              # 04:00 - 09:30 ET
        return ("pre", "Pre-market", "🌅", "#fff3e0")
    elif minutes < 16 * 60:                  # 09:30 - 16:00 ET
        return ("regular", "Marked åbent", "🟢", "#e8f5e9")
    elif minutes < 20 * 60:                  # 16:00 - 20:00 ET
        return ("post", "After-hours", "🌆", "#fff3e0")
    else:                                    # 20:00 - 24:00 ET
        return ("overnight", "Overnight (lukket)", "🌙", "#f0f2f6")


def get_eu_market_status():
    """
    Forenklet status for europæiske børser (Nasdaq Copenhagen, Xetra, Euronext).
    Bruger Europe/Copenhagen-tid; primær handelstid 09:00-17:00 CET/CEST.
    Ingen pre-/after-market — disse børser har stort set ikke detailhandel udenfor.
    """
    try:
        cph_now = pd.Timestamp.now(tz="Europe/Copenhagen")
    except Exception:
        cph_now = pd.Timestamp.now()
    if cph_now.weekday() >= 5:
        return ("weekend", "Lukket (weekend)", "🌙", "#f0f2f6")
    minutes = cph_now.hour * 60 + cph_now.minute
    if 9 * 60 <= minutes < 17 * 60:
        return ("regular", "Marked åbent", "🟢", "#e8f5e9")
    return ("closed", "Lukket", "🌙", "#f0f2f6")


def get_market_status_for_currency(asset_ccy):
    """Vælger session-funktion ud fra aktivets noteringsvaluta."""
    if asset_ccy == "USD":
        return get_us_market_status()
    return get_eu_market_status()


def cashflow_timeline(tx_df, date_range):
    """Akkumuleret saldo for en transaktionsliste, reindekseret til daglig frekvens."""
    if tx_df.empty:
        return pd.Series(0.0, index=date_range)
    s = tx_df.copy()
    s["DateNorm"] = pd.to_datetime(s["Date"], dayfirst=True, errors="coerce").dt.normalize()
    daily = s.groupby("DateNorm")["Amount"].sum()
    return daily.cumsum().reindex(date_range, method="ffill").fillna(0)


def _to_cph_dates(dt_index):
    """Returnerer Index af date-objekter i Europe/Copenhagen for et datetime-index."""
    idx = pd.DatetimeIndex(dt_index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert("Europe/Copenhagen")
    return pd.Index(idx.date, name="Date")


def compute_portfolio_value_series_intraday(orders, dkk_tx, usd_tx, eur_tx, period_key):
    """Intraday porteføljeværdi i DKK til 1D/1U/1M med periode-specifik opløsning."""
    orders = orders.copy()

    if period_key == "1D":
        # 1-min-bars (max yfinance-opløsning). period="5d" er max for 1m-data.
        # Bemærk: 1m har ikke data for alle tickers og kan have gaps i pre/post-market.
        yf_period = "5d"
        yf_interval = "1m"
        resample_rule = None
    elif period_key == "1U":
        yf_period = "5d"
        yf_interval = "1h"
        resample_rule = None
    elif period_key == "1M":
        yf_period = "1mo"
        yf_interval = "1h"
        resample_rule = "4h"  # Pluto: 4 timer mellem punkter
    else:
        raise ValueError("Ugyldig intraday-periode")

    tickers = sorted(orders["Ticker"].dropna().unique().tolist())
    if not tickers:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    prices_i = fetch_price_history_intraday(tuple(tickers), yf_period, yf_interval)
    if prices_i.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    # Resample hvis ønsket (1M: 4h)
    if resample_rule is not None:
        prices_i = prices_i.resample(resample_rule).last()

    # 1D: vis seneste US-handelsdag (ET-dato), ikke CET-kalenderdag.
    # Pga. tidszone-forskydning falder US-after-hours (16:00-20:00 ET) ind i
    # næste CET-kalenderdag. Filtrering på ET-dato sikrer at hele fredagens
    # session (pre-market + regular + after-hours) vises samlet, også når
    # appen åbnes weekenden efter.
    if period_key == "1D":
        idx = pd.DatetimeIndex(prices_i.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        if len(idx) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        idx_et = idx.tz_convert("America/New_York")
        et_dates = idx_et.normalize()
        # Filtrér til hverdage (mandag-fredag) — US-børser handler ikke weekend
        weekday_et_dates = et_dates[et_dates.weekday < 5]
        if len(weekday_et_dates) == 0:
            return pd.Series(dtype=float), pd.Series(dtype=float)
        target_et_date = weekday_et_dates.max()
        mask = et_dates == target_et_date
        prices_i = prices_i.loc[mask]
        if prices_i.empty:
            return pd.Series(dtype=float), pd.Series(dtype=float)

    t_index = prices_i.index
    date_index = _to_cph_dates(t_index)

    # Holdings (daglig) -> map til intraday
    orders["Date"] = pd.to_datetime(orders["Date"], dayfirst=True, errors="coerce")
    orders["TradeDate"] = orders["Date"].dt.normalize()
    orders["Qty_Adj"] = np.where(orders["Side"] == "BUY", orders["Quantity"], -orders["Quantity"])

    start_d = pd.Timestamp(min(date_index)).normalize()
    end_d = pd.Timestamp(max(date_index)).normalize()
    daily_range = pd.date_range(start_d, end_d, freq="D")

    holdings_daily = pd.DataFrame(0.0, index=daily_range, columns=tickers)
    for t in tickers:
        sub = orders[orders["Ticker"] == t]
        daily_change = sub.groupby("TradeDate")["Qty_Adj"].sum()
        holdings_daily[t] = daily_change.cumsum().reindex(daily_range, method="ffill").fillna(0)

    holdings_daily.index = pd.Index(holdings_daily.index.date, name="Date")
    holdings_i = holdings_daily.reindex(date_index, method="ffill").fillna(0)
    holdings_i.index = t_index

    # FX intraday
    usd_dkk_i = prices_i.get("USDDKK=X")
    eur_dkk_i = prices_i.get("EURDKK=X")
    if usd_dkk_i is None:
        usd_dkk_i = pd.Series(6.85, index=t_index)
    else:
        usd_dkk_i = usd_dkk_i.ffill().bfill().fillna(6.85)
    if eur_dkk_i is None:
        eur_dkk_i = pd.Series(7.46, index=t_index)
    else:
        eur_dkk_i = eur_dkk_i.ffill().bfill().fillna(7.46)

    asset_ccy = orders.drop_duplicates("Ticker").set_index("Ticker")["Asset currency"].to_dict()

    stock_value = pd.Series(0.0, index=t_index)
    for t in tickers:
        if t not in prices_i.columns:
            continue
        p = prices_i[t].ffill().bfill().fillna(0)
        q = holdings_i[t]
        ccy = asset_ccy.get(t, "USD")
        if ccy == "USD":
            rate = usd_dkk_i
        elif ccy == "EUR":
            rate = eur_dkk_i
        else:
            rate = 1.0
        stock_value = stock_value + q * p * rate

    # Kontant (daglig saldo) -> map til intraday
    def _daily_balance(tx_df, daily_range_local):
        if tx_df.empty:
            return pd.Series(0.0, index=daily_range_local)
        s = tx_df.copy()
        s["DateNorm"] = pd.to_datetime(s["Date"], dayfirst=True, errors="coerce").dt.normalize()
        daily = s.groupby("DateNorm")["Amount"].sum()
        return daily.cumsum().reindex(daily_range_local, method="ffill").fillna(0)

    cash_dkk_d = _daily_balance(dkk_tx, daily_range)
    cash_usd_d = _daily_balance(usd_tx, daily_range)
    cash_eur_d = _daily_balance(eur_tx, daily_range)

    cash_daily = pd.DataFrame({
        "DKK": cash_dkk_d,
        "USD": cash_usd_d,
        "EUR": cash_eur_d,
    })
    cash_daily.index = pd.Index(cash_daily.index.date, name="Date")
    cash_i = cash_daily.reindex(date_index, method="ffill").fillna(0)
    cash_i.index = t_index

    cash_value = cash_i["DKK"] + cash_i["USD"] * usd_dkk_i + cash_i["EUR"] * eur_dkk_i

    total_value = (stock_value + cash_value).fillna(0)

    # Cashflows til TWR: daglig netto ind/udbetaling -> læg på dagens første intraday punkt
    dates_unique = pd.Index(sorted(set(date_index)), name="Date")
    usd_first = pd.Series(usd_dkk_i.groupby(date_index).first()).reindex(dates_unique)
    eur_first = pd.Series(eur_dkk_i.groupby(date_index).first()).reindex(dates_unique)

    daily_deposits, _daily_fracs = compute_deposits_dkk(dkk_tx, usd_tx, eur_tx, usd_first, eur_first, pd.to_datetime(dates_unique))

    cf_i = pd.Series(0.0, index=t_index)
    first_ts = pd.Series(t_index).groupby(date_index).first()
    for d, ts in first_ts.items():
        try:
            cf_val = float(daily_deposits.loc[pd.Timestamp(d)])
        except Exception:
            cf_val = 0.0
        cf_i.loc[ts] = cf_val

    return total_value, cf_i






@st.cache_data(ttl=3600, show_spinner=False)
def fetch_ticker_quote_info(ticker):
    """yfinance Ticker.info-snapshot — kun de felter detalje-siden bruger.

    Cached i 1 time. Returnerer dict med nøgler:
    volume, average_volume, day_low, day_high, open, fifty_two_week_low,
    fifty_two_week_high, market_cap, trailing_pe, trailing_eps, beta,
    exchange, isin, long_name, website.
    """
    out = {
        "volume": None, "average_volume": None,
        "day_low": None, "day_high": None, "open": None,
        "fifty_two_week_low": None, "fifty_two_week_high": None,
        "market_cap": None, "trailing_pe": None, "trailing_eps": None,
        "beta": None, "exchange": None, "isin": None,
        "long_name": None, "website": None,
    }
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return out
    out["volume"] = _safe_float(info.get("volume") or info.get("regularMarketVolume"))
    out["average_volume"] = _safe_float(
        info.get("averageVolume") or info.get("averageVolume10days")
    )
    out["day_low"] = _safe_float(info.get("dayLow") or info.get("regularMarketDayLow"))
    out["day_high"] = _safe_float(info.get("dayHigh") or info.get("regularMarketDayHigh"))
    out["open"] = _safe_float(info.get("open") or info.get("regularMarketOpen"))
    out["fifty_two_week_low"] = _safe_float(info.get("fiftyTwoWeekLow"))
    out["fifty_two_week_high"] = _safe_float(info.get("fiftyTwoWeekHigh"))
    out["market_cap"] = _safe_float(info.get("marketCap"))
    out["trailing_pe"] = _safe_float(info.get("trailingPE"))
    out["trailing_eps"] = _safe_float(info.get("trailingEps"))
    out["beta"] = _safe_float(info.get("beta"))
    out["exchange"] = info.get("fullExchangeName") or info.get("exchange") or None
    out["isin"] = info.get("isin") or None
    out["long_name"] = info.get("longName") or info.get("shortName") or None
    out["website"] = info.get("website") or None
    return out


def build_holdings_matrix(orders, date_range):
    """DataFrame: index=dato, kolonner=ticker, værdier=beholdning end-of-day."""
    orders = orders.copy()
    orders["Qty_Adj"] = np.where(orders["Side"] == "BUY", orders["Quantity"], -orders["Quantity"])
    tickers = orders["Ticker"].unique()
    holdings = pd.DataFrame(0.0, index=date_range, columns=tickers)
    for t in tickers:
        sub = orders[orders["Ticker"] == t]
        daily_change = sub.groupby("TradeDate")["Qty_Adj"].sum()
        holdings[t] = daily_change.cumsum().reindex(date_range, method="ffill").fillna(0)
    return holdings, list(tickers)


def compute_portfolio_value_series(orders, dkk_tx, usd_tx, eur_tx, date_range):
    """Returnerer en pd.Series med daglig porteføljeværdi i DKK."""
    holdings, tickers = build_holdings_matrix(orders, date_range)

    start_str = (date_range[0] - timedelta(days=7)).strftime("%Y-%m-%d")
    end_str = (date_range[-1] + timedelta(days=2)).strftime("%Y-%m-%d")
    prices_raw = fetch_price_history(tuple(tickers), start_str, end_str)
    prices = prices_raw.reindex(date_range).ffill().bfill()

    if "USDDKK=X" in prices.columns:
        usd_dkk = prices["USDDKK=X"].ffill().bfill().fillna(6.85)
    else:
        usd_dkk = pd.Series(6.85, index=date_range)
    if "EURDKK=X" in prices.columns:
        eur_dkk = prices["EURDKK=X"].ffill().bfill().fillna(7.46)
    else:
        eur_dkk = pd.Series(7.46, index=date_range)

    asset_ccy = orders.drop_duplicates("Ticker").set_index("Ticker")["Asset currency"].to_dict()

    stock_value_dkk = pd.Series(0.0, index=date_range)
    missing = []
    for t in tickers:
        if t not in prices.columns:
            missing.append(t)
            continue
        price_t = prices[t]
        if price_t.isna().all():
            missing.append(t)
            continue
        price_t = price_t.ffill().bfill().fillna(0)
        ccy = asset_ccy.get(t, "USD")
        if ccy == "USD":
            rate = usd_dkk
        elif ccy == "EUR":
            rate = eur_dkk
        else:
            rate = pd.Series(1.0, index=date_range)
        stock_value_dkk = stock_value_dkk + holdings[t] * price_t * rate

    cash_dkk = cashflow_timeline(dkk_tx, date_range)
    cash_usd = cashflow_timeline(usd_tx, date_range)
    cash_eur = cashflow_timeline(eur_tx, date_range)
    cash_value_dkk = cash_dkk + cash_usd * usd_dkk + cash_eur * eur_dkk

    total = (stock_value_dkk + cash_value_dkk).fillna(0)
    return total, stock_value_dkk.fillna(0), cash_value_dkk.fillna(0), holdings, prices, usd_dkk, missing


def compute_deposits_dkk(dkk_tx, usd_tx, eur_tx, usd_dkk_series, eur_dkk_series, date_range):
    """Daglig nettoindskud i DKK. Kun rene ind/udbetalinger — ikke køb/salg.

    Returnerer (cf_series, fracs_series) hvor fracs_series[d] er det vægtede
    gennemsnit af tidsfraktionen (0..1) for deposits på dag d, baseret på enten
    deposit_times.json eller default DEFAULT_DEPOSIT_TIME (09:00 CET).
    """
    cf = pd.Series(0.0, index=date_range)
    # Akkumulér numerator (sum CF*frac) og denominator (sum |CF|) pr. dag for vægtet gennemsnit
    weighted_frac_num = pd.Series(0.0, index=date_range)
    weighted_frac_den = pd.Series(0.0, index=date_range)
    pattern = r"deposit|withdraw|indbetal|udbetal"
    deposit_times = load_deposit_times()
    default_frac = _time_to_frac(DEFAULT_DEPOSIT_TIME)

    for tx_df, rate_series, ccy in [
        (dkk_tx, None, "DKK"),
        (usd_tx, usd_dkk_series, "USD"),
        (eur_tx, eur_dkk_series, "EUR"),
    ]:
        if tx_df.empty:
            continue
        mask = tx_df["Description"].str.contains(pattern, case=False, na=False, regex=True)
        if not mask.any():
            continue
        sub = tx_df[mask].copy()
        sub["DateNorm"] = pd.to_datetime(sub["Date"], dayfirst=True, errors="coerce").dt.normalize()

        # Per-row: slå tidspunkt op og beregn DKK-værdi
        for _, row in sub.iterrows():
            dnorm = row["DateNorm"]
            if pd.isna(dnorm) or dnorm not in date_range:
                continue
            amount_local = float(row["Amount"])
            # Konverter til DKK
            if rate_series is not None:
                rate = float(rate_series.loc[dnorm]) if dnorm in rate_series.index else None
                if rate is None or rate == 0:
                    continue
                amount_dkk = amount_local * rate
            else:
                amount_dkk = amount_local

            # Slå tid op (override eller default)
            date_str = dnorm.strftime("%Y-%m-%d")
            key = _deposit_key(date_str, amount_local, ccy)
            time_str = deposit_times.get(key, DEFAULT_DEPOSIT_TIME)
            frac = _time_to_frac(time_str)

            cf.loc[dnorm] += amount_dkk
            # Vægtet med |amount| så beløbsstørrelse afgør det vægtede gennemsnit
            abs_dkk = abs(amount_dkk)
            weighted_frac_num.loc[dnorm] += frac * abs_dkk
            weighted_frac_den.loc[dnorm] += abs_dkk

    # Beregn vægtet gennemsnits-frac pr. dag; default for dage uden deposits
    fracs = pd.Series(default_frac, index=date_range)
    has_dep = weighted_frac_den > 0
    fracs.loc[has_dep] = weighted_frac_num.loc[has_dep] / weighted_frac_den.loc[has_dep]
    return cf, fracs


def cumulative_return_series(values, cashflows, dates, cashflow_fracs=None):
    """
    Tidsvægtet afkast (TWR) — single-period Modified Dietz, beregnet på hvert
    punkt i tidsserien.

    For hvert punkt d (efter første aktivitet):
        r(d) = (V(d) - V(start) - sum(CFs)) / (V(start) + sum(CF_i × w_i))
        w_i = (d - i - frac_i) / period_length

    hvor period_length = d - first_active - first_active_frac (effektiv investe-
    ringsperiode i fraktionerede dage). frac_i ∈ [0, 1) er andel af dag i
    forløbet før cashflow ankom (fx 09:00 = 9/24 = 0.375). Normaliseringen sikrer
    at en cashflow ved first_active-tidspunktet får vægt 1.0 uanset PORTFOLIO_START.

    cashflow_fracs: optional array eller pd.Series med fracs per dag. Hvis None,
    behandles alle som start-of-day (frac = 0) — bagud-kompatibel med tidligere kode.

    Optimering: O(n) via inkrementelle running sums.
    """
    n = len(values)
    if n == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    values = np.asarray(values, dtype=float)
    cashflows = np.asarray(cashflows, dtype=float)
    if cashflow_fracs is None:
        fracs = np.zeros(n)
    else:
        fracs = np.asarray(cashflow_fracs, dtype=float)
        if len(fracs) != n:
            fracs = np.zeros(n)
    V_start = values[0]
    cum_cf = np.cumsum(cashflows) - cashflows[0]
    return_dkk = values - V_start - cum_cf

    # Find første aktive dag (første ikke-nul cashflow eller positiv porteføljeværdi).
    first_active = 0
    first_active_frac = 0.0
    for i in range(n):
        if cashflows[i] != 0 or values[i] > 1e-9:
            first_active = i
            first_active_frac = fracs[i] if cashflows[i] != 0 else 0.0
            break

    twr_pct = np.zeros(n)
    total_cf_so_far = 0.0
    weighted_index_sum = 0.0  # sum((i + frac_i) × CF_i)
    for d in range(n):
        total_cf_so_far += cashflows[d]
        weighted_index_sum += (d + fracs[d]) * cashflows[d]
        if d <= first_active:
            continue
        # period_length i fraktionerede dage: fra (first_active + first_active_frac) til d
        period_length = d - first_active - first_active_frac
        if period_length <= 0:
            continue
        # weighted_cf = sum(CF_i × (d - i - frac_i)) / period_length
        # numerator = d × total_cf - sum((i + frac_i) × CF_i) = d × total_cf - weighted_index_sum
        weighted_cf = (d * total_cf_so_far - weighted_index_sum) / period_length
        denom = V_start + weighted_cf
        if denom > 1e-9:
            twr_pct[d] = return_dkk[d] / denom * 100

    return pd.Series(return_dkk, index=dates), pd.Series(twr_pct, index=dates)


def slice_period(value_series, cf_series, period_key):
    """Slicer tidsserien til den valgte periode."""
    end_date = value_series.index[-1]
    if period_key == "1D":
        start = value_series.index[-2] if len(value_series) >= 2 else value_series.index[0]
    elif period_key == "1U":
        start = end_date - timedelta(days=7)
    elif period_key == "1M":
        start = end_date - timedelta(days=30)
    elif period_key == "3M":
        start = end_date - timedelta(days=90)
    elif period_key == "6M":
        start = end_date - timedelta(days=180)
    elif period_key == "YTD":
        start = pd.Timestamp(year=end_date.year, month=1, day=1)
    elif period_key == "1Å":
        start = end_date - timedelta(days=365)
    elif period_key == "3Å":
        start = end_date - timedelta(days=3 * 365)
    elif period_key == "5Å":
        start = end_date - timedelta(days=5 * 365)
    else:
        start = value_series.index[0]

    start = max(start, value_series.index[0])
    mask = value_series.index >= start
    return value_series[mask], cf_series[mask], value_series.index[mask]


def color_change_str(val):
    """Farv strenge der starter med + (grøn) eller - (rød)."""
    if not isinstance(val, str):
        return ""
    if val.startswith("+"):
        return "color: #2e7d32; font-weight: 600"
    if val.startswith("-"):
        return "color: #d32f2f; font-weight: 600"
    return ""


# -------------------- AKTIV-DETALJE-SIDE --------------------
def _fetch_asset_history(ticker, period_key, asset_ccy, include_extended=False):
    """Hent pris- og volumen-historik for én ticker til detalje-grafens periode.

    Returnerer (prices_series, volume_series) med datetime-indeks i UTC.
    Bruger intraday-interval for 1D/1U/1M, daglig ellers. include_extended
    styrer om pre/post-market bars hentes (kun relevant for intraday — for
    1M ændrer det yfinance-fetchet, for 1D/1U er prepost altid True og
    filtreres efterfølgende)."""
    try:
        tk = yf.Ticker(ticker)
        if period_key == "1D":
            hist = tk.history(period="2d", interval="1m", prepost=True, auto_adjust=False)
        elif period_key == "1U":
            hist = tk.history(period="5d", interval="5m", prepost=True, auto_adjust=False)
        elif period_key == "1M":
            hist = tk.history(
                period="1mo", interval="1h",
                prepost=include_extended, auto_adjust=False,
            )
        elif period_key == "3M":
            hist = tk.history(period="3mo", interval="1d", prepost=False, auto_adjust=False)
        elif period_key == "6M":
            hist = tk.history(period="6mo", interval="1d", prepost=False, auto_adjust=False)
        elif period_key == "YTD":
            hist = tk.history(period="ytd", interval="1d", prepost=False, auto_adjust=False)
        elif period_key == "1Å":
            hist = tk.history(period="1y", interval="1d", prepost=False, auto_adjust=False)
        elif period_key == "5Å":
            hist = tk.history(period="5y", interval="1wk", prepost=False, auto_adjust=False)
        else:  # Maks
            hist = tk.history(period="max", interval="1mo", prepost=False, auto_adjust=False)
    except Exception:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    if hist is None or hist.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    prices = hist["Close"].dropna() if "Close" in hist.columns else pd.Series(dtype=float)
    volumes = hist["Volume"].dropna() if "Volume" in hist.columns else pd.Series(dtype=float)
    return prices, volumes


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_period_reference_price(ticker, period_key):
    """Yahoo-stil referencepris: close FØR periode-start.

    Bruges til at beregne periode-ændring sådan at den er uafhængig af om
    extended hours er på/af i grafen — reference er altid en regular session
    close fra dagen før perioden startede.

    Returnerer None for 1D (caller bruger prev_close) og Maks (caller bruger
    første pris i grafen). For øvrige perioder returneres close ved eller
    lige før target-datoen.
    """
    if period_key in ("1D", "Maks"):
        return None

    fetch_period_map = {
        "1U": "1mo", "1M": "3mo", "3M": "6mo", "6M": "1y",
        "YTD": "2y", "1Å": "2y", "5Å": "6y",
    }
    fetch_period = fetch_period_map.get(period_key, "1mo")

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(
            period=fetch_period, interval="1d",
            prepost=False, auto_adjust=False,
        )
        if hist.empty or "Close" not in hist.columns:
            return None
    except Exception:
        return None
    closes = hist["Close"].dropna()
    if closes.empty:
        return None

    today = pd.Timestamp.now().normalize()
    if period_key == "1U":
        target = today - pd.Timedelta(days=7)
    elif period_key == "1M":
        target = today - pd.DateOffset(months=1)
    elif period_key == "3M":
        target = today - pd.DateOffset(months=3)
    elif period_key == "6M":
        target = today - pd.DateOffset(months=6)
    elif period_key == "YTD":
        target = pd.Timestamp(year=today.year, month=1, day=1)
    elif period_key == "1Å":
        target = today - pd.DateOffset(years=1)
    elif period_key == "5Å":
        target = today - pd.DateOffset(years=5)
    else:
        return None

    target = pd.Timestamp(target).normalize()
    idx = pd.DatetimeIndex(closes.index)
    if idx.tz is not None:
        idx_naive = idx.tz_localize(None).normalize()
    else:
        idx_naive = idx.normalize()

    # Sidste close STRIKT FØR target-datoen — det er Yahoo's reference,
    # dvs. closen fra dagen FØR periode-grafens første bar (ikke closen
    # PÅ samme dag som første bar, som ville være periodens åbnings-close).
    mask = idx_naive < target
    if mask.any():
        return float(closes[mask].iloc[-1])
    # Edge case: target er før vores ældste data → brug ældste close
    return float(closes.iloc[0])


def _ticker_prev_close(ticker):
    """Henter forrige regular-session close (= 'Sidste luk') for en ticker."""
    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period="10d", interval="1d", prepost=False, auto_adjust=False)
        if hist.empty:
            return None
        # Drop dagens igangværende bar — samme logik som fetch_live_quotes
        et_today = pd.Timestamp.now(tz="America/New_York").date()
        last_ts = pd.Timestamp(hist.index[-1])
        last_date_et = (last_ts.tz_convert("America/New_York").date()
                        if last_ts.tz is not None else last_ts.date())
        if last_date_et >= et_today:
            hist = hist.iloc[:-1]
        closes = hist["Close"].dropna()
        return float(closes.iloc[-1]) if len(closes) else None
    except Exception:
        return None


def _slice_intraday_to_regular(prices, volumes):
    """For 1D-grafen: behold kun bars i regular hours (09:30-16:00 ET) på
    seneste handelsdag. Falder tilbage til seneste afsluttede session i
    pre-market — samme logik som fetch_intraday_sparklines."""
    if prices.empty:
        return prices, volumes
    idx = pd.DatetimeIndex(prices.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_et = idx.tz_convert("America/New_York")
    et_dates = idx_et.normalize()
    et_minutes = idx_et.hour * 60 + idx_et.minute
    regular_mask = (et_minutes >= 9 * 60 + 30) & (et_minutes < 16 * 60)
    weekday_mask = et_dates.weekday < 5
    candidate_dates = et_dates[regular_mask & weekday_mask]
    if len(candidate_dates) == 0:
        return prices.iloc[0:0], volumes.iloc[0:0] if not volumes.empty else volumes
    target = candidate_dates.max()
    mask = (et_dates == target) & regular_mask
    keep = pd.Series(mask, index=prices.index)
    return prices[keep], (volumes[keep[volumes.index]] if not volumes.empty else volumes)


def _segment_by_sign(xs, ys, baseline=0.0):
    """Split serie i kontinuerlige segmenter ift. baseline. Returnerer liste
    af (sign, [(x, y), ...]) hvor sign er 'pos'/'neg'/None. Interpolerer
    zero-crossings så grøn/rød segmenter mødes præcis på baseline."""
    if len(xs) == 0:
        return []
    ex_xs, ex_ys = [], []
    for i in range(len(xs)):
        ex_xs.append(xs[i])
        ex_ys.append(ys[i])
        if i + 1 < len(xs):
            y0 = ys[i] - baseline
            y1 = ys[i + 1] - baseline
            if (y0 > 0 and y1 < 0) or (y0 < 0 and y1 > 0):
                t0 = pd.Timestamp(xs[i]).value
                t1 = pd.Timestamp(xs[i + 1]).value
                frac = y0 / (y0 - y1)
                ex_xs.append(pd.Timestamp(int(t0 + frac * (t1 - t0))).to_numpy())
                ex_ys.append(baseline)
    segments = []
    cur_sign, cur_seg = None, []
    for x, y in zip(ex_xs, ex_ys):
        delta = y - baseline
        if delta > 1e-9:
            new_sign = "pos"
        elif delta < -1e-9:
            new_sign = "neg"
        else:
            new_sign = None
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
    return segments


def _annualized_volatility(daily_returns):
    """Annualiseret σ × √252 i procent."""
    r = pd.Series(daily_returns).dropna()
    if len(r) < 2:
        return None
    return float(r.std(ddof=1) * np.sqrt(252) * 100)


def _max_drawdown(price_series):
    """Returnerer (drawdown_pct, peak_date, trough_date) for en prisserie."""
    s = pd.Series(price_series).dropna()
    if len(s) < 2:
        return None, None, None
    cummax = s.cummax()
    dd = (s - cummax) / cummax
    trough_idx = dd.idxmin()
    peak_idx = s.loc[:trough_idx].idxmax()
    return float(dd.min() * 100), peak_idx, trough_idx


def _period_return_pct(price_series, period_key):
    """Procentvis ændring i en periode af en daglig prisserie."""
    s = pd.Series(price_series).dropna()
    if len(s) < 2:
        return None
    end_date = s.index[-1]
    if period_key == "1D":
        start = s.index[-2]
    elif period_key == "1U":
        start_dt = end_date - timedelta(days=7)
        idx = s.index[s.index >= start_dt]
        start = idx[0] if len(idx) else s.index[0]
    elif period_key == "1M":
        start_dt = end_date - timedelta(days=30)
        idx = s.index[s.index >= start_dt]
        start = idx[0] if len(idx) else s.index[0]
    elif period_key == "YTD":
        ytd = pd.Timestamp(year=pd.Timestamp(end_date).year, month=1, day=1, tz=getattr(end_date, "tz", None))
        idx = s.index[s.index >= ytd]
        start = idx[0] if len(idx) else s.index[0]
    else:  # Maks
        start = s.index[0]
    v0 = float(s.loc[start])
    v1 = float(s.iloc[-1])
    if v0 == 0:
        return None
    return (v1 - v0) / v0 * 100


def render_asset_detail(ticker, orders_df, positions_df, cash_df, prices,
                        usd_dkk, eur_dkk, total_value, cashflows, cashflow_fracs,
                        grand_total, total_v, all_active_tickers):
    """Render detalje-side for én ticker. Kaldes når ?ticker=XXX er sat."""
    # Validering
    if ticker not in orders_df["Ticker"].values:
        st.error(f"Ukendt ticker: {ticker}")
        st.markdown('<a href="?" style="font-size:16px;">← Tilbage til oversigt</a>',
                    unsafe_allow_html=True)
        return

    # --- Aggregér min position ---
    sub = orders_df[orders_df["Ticker"] == ticker].copy()
    sub["Qty_Adj"] = np.where(sub["Side"] == "BUY", sub["Quantity"], -sub["Quantity"])
    sub["DKK_Adj"] = np.where(sub["Side"] == "BUY", sub["Notional, DKK"], -sub["Notional, DKK"])
    qty_total = float(sub["Qty_Adj"].sum())
    cost_dkk = float(sub["DKK_Adj"].sum())
    asset_ccy = sub["Asset currency"].iloc[0]
    name = sub["Name"].iloc[0]
    first_buy = sub[sub["Side"] == "BUY"]["TradeDate"].min()
    holding_days = (pd.Timestamp.now().normalize() - pd.Timestamp(first_buy).normalize()).days if pd.notnull(first_buy) else 0

    # --- Hent live + meta + info (samlet fetch for alle aktive — hits cache) ---
    all_quotes = fetch_live_quotes(tuple(all_active_tickers))
    quote = all_quotes.get(ticker, {})
    meta = fetch_ticker_meta((ticker,)).get(ticker, {})
    info = fetch_ticker_quote_info(ticker)
    live_fx = fetch_live_fx_rates()
    usd_dkk_now = live_fx.get("USDDKK") or (float(usd_dkk.iloc[-1]) if len(usd_dkk) else 6.85)
    eur_dkk_now = live_fx.get("EURDKK") or (float(eur_dkk.iloc[-1]) if len(eur_dkk) else 7.46)
    rate_now = usd_dkk_now if asset_ccy == "USD" else (eur_dkk_now if asset_ccy == "EUR" else 1.0)

    prev_close = quote.get("prev_close")
    prev_prev = quote.get("prev_prev_close")
    live = quote.get("live")
    # Fallback til prices-DataFrame hvis live-fetch fejlede
    if prev_close is None and ticker in prices.columns:
        ser = prices[ticker].dropna()
        if len(ser):
            prev_close = float(ser.iloc[-1])
    if live is None:
        live = prev_close

    ccy_sym = _CCY_SYMBOLS.get(asset_ccy, asset_ccy)

    # --- Beregn afledte tal ---
    aktuel_dkk_value = qty_total * (live or 0) * rate_now if live else 0.0
    unrealized_dkk = aktuel_dkk_value - cost_dkk
    unrealized_pct = (unrealized_dkk / cost_dkk * 100) if cost_dkk else 0
    gak_valuta = (cost_dkk / qty_total) / rate_now if (qty_total > 0 and rate_now > 0) else 0
    delta_dkk = (live - prev_close) if (live is not None and prev_close is not None) else 0
    delta_pct = (delta_dkk / prev_close * 100) if prev_close else 0

    # =====================================================================
    # SEKTION 1: HEADER
    # =====================================================================
    st.markdown(
        '<a href="?" style="font-size:14px; color:#1976d2; text-decoration:none;">'
        '← Tilbage til oversigt</a>',
        unsafe_allow_html=True,
    )
    st.write("")

    # Session-badge
    sess_code, sess_label, sess_emoji, sess_bg = get_market_status_for_currency(asset_ccy)

    # Chips
    exchange = info.get("exchange") or "—"
    chips = [
        ("Sektor", meta.get("sector", "Andet")),
        ("Region", meta.get("region", "Andet")),
        ("Land", meta.get("country", "Andet")),
        ("Aktivklasse", meta.get("asset_class", "Andet")),
        ("Valuta", asset_ccy),
        ("Børs", exchange),
    ]
    chips_html = "".join(
        f"<span style='display:inline-block; padding:3px 10px; "
        f"background:#f0f2f6; border-radius:12px; font-size:12px; "
        f"margin:0 6px 4px 0; color:#444;'>"
        f"<span style='color:#888;'>{label}:</span> <strong>{value}</strong>"
        f"</span>"
        for label, value in chips
    )

    header_html = (
        f"<div style='display:flex; justify-content:space-between; align-items:flex-start;'>"
        f"  <div>"
        f"    <h1 style='margin:0 0 4px;'>{name}</h1>"
        f"    <span style='display:inline-block; padding:4px 12px; background:#1976d2; "
        f"                 color:#fff; border-radius:6px; font-weight:600; font-size:14px;'>"
        f"      {ticker}"
        f"    </span>"
        f"  </div>"
        f"  <div style='background:{sess_bg}; padding:8px 14px; border-radius:12px; "
        f"              font-size:13px;'>"
        f"    <strong>{sess_emoji} {sess_label}</strong>"
        f"  </div>"
        f"</div>"
        f"<div style='margin-top:10px;'>{chips_html}</div>"
    )
    st.markdown(_flatten_html(header_html), unsafe_allow_html=True)
    st.write("---")

    # =====================================================================
    # SEKTION 2: PRIS-HERO (Yahoo-stil dual pris)
    # =====================================================================
    today_close_val = quote.get("today_close")

    # Bestem hvilke priser der skal vises baseret på markeds-session
    def _price_box_html(price, ref, context_label):
        """Byg én pris-blok (label + stor pris + Δ vs ref)."""
        if price is None:
            return ""
        if ref is not None and ref != 0:
            d_val = price - ref
            d_pct = d_val / ref * 100
            d_color = "#2e7d32" if d_val >= 0 else "#d32f2f"
            d_arrow = "▲" if d_val >= 0 else "▼"
            delta_part = (
                f"<div style='font-size:18px; color:{d_color}; "
                f"font-weight:600; margin-top:6px;'>"
                f"{d_arrow} {_da_num(d_val, signed=True)} "
                f"({_da_num(d_pct, signed=True)}%)"
                f"</div>"
            )
        else:
            delta_part = ""
        return (
            f"<div>"
            f"  <div style='font-size:13px; color:#888; margin-bottom:4px;'>{context_label}</div>"
            f"  <div style='font-size:40px; font-weight:700; line-height:1;'>"
            f"    {ccy_sym} {_da_num(price)}"
            f"  </div>"
            f"  {delta_part}"
            f"</div>"
        )

    if sess_code == "regular":
        # Regular session: én pris (live), Δ vs forrige lukkepris
        main_box = _price_box_html(live, prev_close, "Aktuel pris (markedet åbent)")
        ext_box = ""
    elif today_close_val is not None:
        # Efter dagens regular close (samme dag): venstre = dagens close,
        # højre = nuværende extended-hours live
        main_box = _price_box_html(today_close_val, prev_close, "Dagens lukkepris (16:00 ET)")
        if live is not None and abs(live - today_close_val) > 0.001:
            ext_box = _price_box_html(live, today_close_val, sess_label)
        else:
            ext_box = ""
    else:
        # Pre-market / overnight (efter midnat) / weekend:
        # venstre = forrige lukkepris, højre = nuværende live (hvis afviger)
        main_box = _price_box_html(prev_close, prev_prev, "Forrige lukkepris")
        if (live is not None and prev_close is not None
                and abs(live - prev_close) > 0.001):
            ext_box = _price_box_html(live, prev_close, sess_label)
        else:
            ext_box = ""

    if ext_box:
        hero_inner = (
            f"<div style='display:flex; gap:48px; align-items:flex-start;'>"
            f"  {main_box}"
            f"  <div style='width:1px; background:#ddd; align-self:stretch;'></div>"
            f"  {ext_box}"
            f"</div>"
        )
    else:
        hero_inner = main_box

    hero_html = (
        f"{hero_inner}"
        f"<div style='font-size:13px; color:#888; margin-top:12px;'>"
        f"  ≙ <strong>{_da_num(aktuel_dkk_value)} DKK</strong> i min beholdning "
        f"  ({format_quantity(qty_total)} stk.)"
        f"</div>"
    )
    st.markdown(_flatten_html(hero_html), unsafe_allow_html=True)
    st.write("---")

    # =====================================================================
    # SEKTION 3: PRIS-GRAF + VOLUMEN
    # =====================================================================
    st.subheader("Pris over tid")

    gc_col1, gc_col2 = st.columns([4, 1])
    with gc_col1:
        det_period = st.radio(
            "Periode",
            ["1D", "1U", "1M", "3M", "6M", "YTD", "1Å", "5Å", "Maks"],
            horizontal=True, label_visibility="collapsed",
            index=0, key=f"detail_period_{ticker}",
        )
    with gc_col2:
        include_extended = st.toggle(
            "Inkl. udvidet åbningstid",
            value=False,
            key=f"detail_ext_{ticker}",
            help=(
                "Inkluder pre-market (04:00-09:30 ET) og after-hours "
                "(16:00-20:00 ET) bars i intraday-graferne. Virker kun for "
                "1D/1U/1M."
            ),
        )
    det_y_mode = "$"  # Y-aksen er altid pris i aktivets valuta nu

    with st.spinner("Henter prishistorik..."):
        det_prices, det_volumes = _fetch_asset_history(
            ticker, det_period, asset_ccy, include_extended=include_extended
        )

    # Filtrér intraday-data til de relevante markedstimer (på hverdage). For
    # 1D: kun seneste handelsdag. For 1U/1M: alle hverdage i perioden. Det
    # giver en ren linje uden diagonaler hen over overnight/weekend.
    # Bemærk: include_extended=True udvider vinduet til 04:00-20:00 ET
    # (pre-market + regular + after-hours).
    if det_period in ("1D", "1U", "1M") and not det_prices.empty:
        _idx = pd.DatetimeIndex(det_prices.index)
        if _idx.tz is None:
            _idx = _idx.tz_localize("UTC")
        _idx_et = _idx.tz_convert("America/New_York")
        _et_min = _idx_et.hour * 60 + _idx_et.minute
        if include_extended:
            _hours_mask = (_et_min >= 4 * 60) & (_et_min < 20 * 60)
        else:
            _hours_mask = (_et_min >= 9 * 60 + 30) & (_et_min < 16 * 60)
        _weekday = _idx_et.weekday < 5
        _keep = _hours_mask & _weekday
        if det_period == "1D":
            _et_dates = _idx_et.normalize()
            _cand = _et_dates[_keep]
            if len(_cand) > 0:
                _keep = _keep & (_et_dates == _cand.max())
        _keep_s = pd.Series(_keep, index=det_prices.index)
        det_prices = det_prices[_keep_s]
        if not det_volumes.empty:
            det_volumes = det_volumes[
                _keep_s.reindex(det_volumes.index, fill_value=False)
            ]

    if det_prices.empty:
        st.warning("Ingen prishistorik tilgængelig for denne periode.")
    else:
        # Periode-ændring (Yahoo-stil): brug en FAST referencepris baseret på
        # regular session close FØR perioden startede — det gør procenten
        # uafhængig af om extended hours er på/af i grafen.
        #
        #  - 1D: prev_close (= gårsdagens close)
        #  - 1U / 1M / 3M / 6M / YTD / 1Å / 5Å: daglig close ved eller lige
        #    før target-datoen (hentet separat via _fetch_period_reference_price)
        #  - Maks: chart-data'ens første punkt (ingen ekstern reference)
        #
        # End-prisen er live (seneste tick fra fetch_live_quotes), så også
        # uafhængig af toggle. Begge dele giver samme procent som Yahoo/MSN.
        if det_period == "1D":
            ref_price = prev_close
        elif det_period == "Maks":
            ref_price = None  # Brug første pris i grafen
        else:
            ref_price = _fetch_period_reference_price(ticker, det_period)

        if ref_price is not None and ref_price > 0:
            period_start_price = float(ref_price)
        else:
            period_start_price = float(det_prices.iloc[0])

        # End-pris: Yahoo bruger dagens regular close som end (ikke after-hours).
        # I regular hours er today_close None og vi bruger live (= dagens
        # igangværende intraday-pris). Efter close bruges today_close ($55,15
        # = 16:00 ET), så efter-hours-bevægelser ikke påvirker procenten.
        # Det matcher Yahoo's tal for alle perioder og for 1D specifikt sikrer
        # det at periode-ændringen = 'Dagens lukkepris'-Δ i hero-sektionen.
        if today_close_val is not None:
            period_end_price = float(today_close_val)
        elif live is not None:
            period_end_price = float(live)
        else:
            period_end_price = float(det_prices.iloc[-1])
        period_change_local = period_end_price - period_start_price
        period_change_pct = (
            period_change_local / period_start_price * 100
        ) if period_start_price else 0
        period_is_up = period_change_local >= 0

        # Periode-tekst over grafen (Google Finance-stil)
        arrow = "↑" if period_is_up else "↓"
        pc_color = "#2e7d32" if period_is_up else "#d32f2f"
        st.markdown(
            f"<div style='font-size:18px; color:{pc_color}; font-weight:600; "
            f"margin:4px 0 10px;'>"
            f"{arrow} {_da_num(period_change_local, signed=True)} {ccy_sym} "
            f"({_da_num(period_change_pct, signed=True)}%)"
            f"<span style='color:#888; font-weight:400; font-size:13px; "
            f"margin-left:8px;'>i {det_period}-perioden</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Vælg om vi viser stiplet baseline. 1D = sidste luk, Maks = GAK.
        # For øvrige perioder vises ingen stiplet linje.
        if det_period == "Maks":
            baseline = gak_valuta if gak_valuta > 0 else period_start_price
            baseline_label = f"GAK {ccy_sym} {_da_num(baseline)}"
            show_baseline_line = True
        elif det_period == "1D":
            baseline = prev_close if prev_close else period_start_price
            baseline_label = f"Sidste luk {ccy_sym} {_da_num(baseline)}"
            show_baseline_line = True
        else:
            baseline = period_start_price
            baseline_label = ""
            show_baseline_line = False

        # Y-værdier afhænger af $/% toggle
        if det_y_mode == "%":
            ys = ((det_prices.values - period_start_price) / period_start_price) * 100 \
                 if period_start_price else np.zeros(len(det_prices))
            y_title = "Afkast i perioden (%)"
            y_tickformat = ".2f"
            hover_fmt = "<b>%{x|%d. %b %Y %H:%M}</b><br>Afkast: %{y:.2f}%<extra></extra>"
            # Baseline-y i %-mode = afkast af baseline-prisen i forhold til periode-start
            y_baseline = ((baseline - period_start_price) / period_start_price * 100) \
                         if period_start_price else 0
        else:
            ys = det_prices.values
            y_title = f"Pris ({asset_ccy})"
            y_tickformat = ",.2f"
            hover_fmt = f"<b>%{{x|%d. %b %Y %H:%M}}</b><br>Pris: {ccy_sym} %{{y:,.2f}}<extra></extra>"
            y_baseline = baseline

        # Konvertér x-akse til ET tz-naiv for intraday-perioder så rangebreaks
        # virker korrekt (skjul weekender + overnight 16:00-09:30 ET).
        is_intraday = det_period in ("1D", "1U", "1M")
        price_idx_orig = pd.DatetimeIndex(det_prices.index)
        if is_intraday and price_idx_orig.tz is not None:
            price_x = price_idx_orig.tz_convert("America/New_York").tz_localize(None)
        else:
            price_x = price_idx_orig

        if not det_volumes.empty:
            vol_idx_orig = pd.DatetimeIndex(det_volumes.index)
            if is_intraday and vol_idx_orig.tz is not None:
                vol_x = vol_idx_orig.tz_convert("America/New_York").tz_localize(None)
            else:
                vol_x = vol_idx_orig
        else:
            vol_x = det_volumes.index

        # Bygges som subplot: pris øverst, volumen nederst
        from plotly.subplots import make_subplots
        fig_det = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            row_heights=[0.75, 0.25], vertical_spacing=0.05,
        )

        # Én linje-trace med fill ned til chart-bunden (tozeroy). Farven
        # bestemmes af periode-retningen — grøn hvis prisen samlet steg,
        # rød hvis den faldt. Y-aksen begrænses manuelt til data-range,
        # så det usynlige fill-areal under bundlinjen ikke trækker chartet ud.
        line_color = "#2e7d32" if period_is_up else "#d32f2f"
        fill_rgba = "rgba(46,125,50,0.15)" if period_is_up else "rgba(211,47,47,0.15)"

        fig_det.add_trace(
            go.Scatter(
                x=price_x, y=ys,
                mode="lines",
                line=dict(color=line_color, width=2),
                fill="tozeroy",
                fillcolor=fill_rgba,
                hovertemplate=hover_fmt,
                showlegend=False,
            ),
            row=1, col=1,
        )

        # Stiplet baseline med label (kun for 1D og Maks)
        if show_baseline_line:
            fig_det.add_hline(
                y=y_baseline, line_dash="dash", line_color="rgba(0,0,0,0.5)",
                annotation_text=baseline_label, annotation_position="top right",
                annotation_font_size=11, row=1, col=1,
            )

        # Manuel y-akse range så fill="tozeroy" ikke presser y=0 ind i visningen.
        # Inkludér også baseline-værdien hvis vi viser den stiplede linje.
        y_min_data = float(np.min(ys))
        y_max_data = float(np.max(ys))
        if show_baseline_line:
            y_min_data = min(y_min_data, y_baseline)
            y_max_data = max(y_max_data, y_baseline)
        y_span = y_max_data - y_min_data
        y_pad = max(y_span * 0.05, 0.01)
        fig_det.update_yaxes(
            range=[y_min_data - y_pad, y_max_data + y_pad],
            row=1, col=1,
        )

        # Volumen-søjler
        if not det_volumes.empty:
            avg_vol = float(det_volumes.mean()) if len(det_volumes) else 0
            fig_det.add_trace(
                go.Bar(
                    x=vol_x, y=det_volumes.values,
                    marker_color="#bdbdbd",
                    hovertemplate=(
                        "<b>%{x|%d. %b %Y}</b><br>Volumen: %{y:,.0f}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=2, col=1,
            )
            if avg_vol > 0:
                fig_det.add_hline(
                    y=avg_vol, line_dash="dot", line_color="rgba(0,0,0,0.4)",
                    annotation_text=f"Snit: {format_big_number(avg_vol)}",
                    annotation_position="top right",
                    annotation_font_size=10, row=2, col=1,
                )

        fig_det.update_layout(
            height=480, margin=dict(l=0, r=0, t=20, b=0),
            hovermode="x unified",
            plot_bgcolor="white", separators=".,",
        )
        fig_det.update_yaxes(title=y_title, tickformat=y_tickformat, row=1, col=1)
        fig_det.update_yaxes(title="Volumen", tickformat=",.0s", row=2, col=1)

        # Rangebreaks: skjul weekender og ikke-handelstimer på intraday-grafer,
        # så de små overnight-gaps ikke bliver til lange diagonale linjer.
        # - Regular only: skjul 16:00-09:30 ET
        # - Extended (pre/regular/post): skjul 20:00-04:00 ET
        # Virker kun korrekt fordi x-aksen er konverteret til ET tz-naiv.
        if is_intraday:
            if include_extended:
                rangebreaks = [
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[20, 4], pattern="hour"),
                ]
            else:
                rangebreaks = [
                    dict(bounds=["sat", "mon"]),
                    dict(bounds=[16, 9.5], pattern="hour"),
                ]
            fig_det.update_xaxes(rangebreaks=rangebreaks, row=1, col=1)
            fig_det.update_xaxes(rangebreaks=rangebreaks, row=2, col=1)

        st.plotly_chart(fig_det, use_container_width=True)

    st.write("---")

    # =====================================================================
    # SEKTION 4: MARKEDSDATA (range-bars)
    # =====================================================================
    st.subheader("Markedsdata")

    col_vol, col_day = st.columns(2)
    with col_vol:
        st.markdown(
            _flatten_html(_make_volume_bar_html(info["volume"], info["average_volume"])),
            unsafe_allow_html=True,
        )
    with col_day:
        st.markdown(
            _flatten_html(_make_range_bar_html(
                info["day_low"], info["day_high"],
                marker_low=info["open"], marker_high=live,
                marker_low_label="OPEN", marker_high_label="LAST",
                bottom_label="DAY LOW/HIGH",
                currency_symbol=ccy_sym, segment_fill=False,
            )),
            unsafe_allow_html=True,
        )

    st.markdown(
        _flatten_html(_make_range_bar_html(
            info["fifty_two_week_low"], info["fifty_two_week_high"],
            marker_low=info["day_low"], marker_high=info["day_high"],
            bottom_label="52 WEEK LOW/HIGH",
            currency_symbol=ccy_sym, segment_fill=True,
        )),
        unsafe_allow_html=True,
    )

    st.write("")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("Market cap", format_big_number(info["market_cap"]) if info["market_cap"] else "—")
    mc2.metric("P/E (TTM)", _da_num(info["trailing_pe"]) if info["trailing_pe"] else "—")
    mc3.metric("EPS (TTM)", f"{ccy_sym} {_da_num(info['trailing_eps'])}" if info["trailing_eps"] else "—")
    st.write("---")

    # =====================================================================
    # SEKTION 5: MIN POSITION
    # =====================================================================
    st.subheader("Min position")

    p1, p2, p3, p4 = st.columns(4)
    p1.metric("Antal", format_quantity(qty_total))
    p2.metric("GAK", f"{ccy_sym} {_da_num(gak_valuta)}")
    p3.metric("Kostbasis", f"{_da_num(cost_dkk)} DKK")
    p4.metric("Aktuel værdi", f"{_da_num(aktuel_dkk_value)} DKK")

    q1, q2, q3, q4 = st.columns(4)
    q1.metric(
        "Urealiseret afkast",
        f"{_da_num(unrealized_dkk)} DKK",
        delta=f"{_da_num(unrealized_pct, signed=True)}%",
    )
    pct_of_total = (aktuel_dkk_value / grand_total * 100) if grand_total else 0
    pct_of_aktier = (aktuel_dkk_value / total_v * 100) if total_v else 0
    q2.metric("Andel af aktier", f"{_da_num(pct_of_aktier)}%")
    q3.metric("Andel af portefølje", f"{_da_num(pct_of_total)}%")
    q4.metric("Holdt i", f"{holding_days} dage")

    # Realiseret afkast (simpel FIFO)
    realized_dkk = 0.0
    realized_qty = float(sub[sub["Side"] == "SELL"]["Quantity"].sum())
    if realized_qty > 0:
        # FIFO: par hver SELL med tidligere BUYs
        sub_sorted = sub.sort_values("TradeDate")
        buys_queue = []  # liste af [resterende_qty, kostbasis_per_stk_DKK]
        for _, r in sub_sorted.iterrows():
            if r["Side"] == "BUY":
                cb = float(r["Notional, DKK"]) / float(r["Quantity"]) if r["Quantity"] else 0
                buys_queue.append([float(r["Quantity"]), cb])
            else:  # SELL
                qty_remaining = float(r["Quantity"])
                sell_price_dkk = float(r["Notional, DKK"]) / qty_remaining if qty_remaining else 0
                while qty_remaining > 0 and buys_queue:
                    q_avail = buys_queue[0][0]
                    cb = buys_queue[0][1]
                    matched = min(qty_remaining, q_avail)
                    realized_dkk += matched * (sell_price_dkk - cb)
                    buys_queue[0][0] -= matched
                    qty_remaining -= matched
                    if buys_queue[0][0] <= 1e-9:
                        buys_queue.pop(0)
        st.caption(f"Realiseret afkast siden køb: **{_da_num(realized_dkk, signed=True)} DKK** "
                   f"({_da_num(realized_qty)} stk. solgt)")

    st.write("---")

    # =====================================================================
    # SEKTION 6: PERFORMANCE VS. PORTEFØLJE
    # =====================================================================
    st.subheader("Performance vs. portefølje")

    perf_periods = ["1D", "1U", "1M", "YTD", "Maks"]
    asset_returns = {}
    port_returns = {}

    # Asset-afkast pr. periode (brug daglig prices-DataFrame som er allerede loaded)
    if ticker in prices.columns:
        asset_daily = prices[ticker].dropna()
        for p in perf_periods:
            asset_returns[p] = _period_return_pct(asset_daily, p)

    # Portefølje-TWR pr. periode
    for p in perf_periods:
        try:
            v_p, cf_p, d_p = slice_period(total_value, cashflows, p)
            fracs_p = cashflow_fracs.reindex(d_p).values if cashflow_fracs is not None else None
            _, twr_s = cumulative_return_series(v_p.values, cf_p.values, d_p,
                                                cashflow_fracs=fracs_p)
            port_returns[p] = float(twr_s.iloc[-1]) if len(twr_s) else None
        except Exception:
            port_returns[p] = None

    def _fmt_pct_signed(v):
        if v is None:
            return "<span style='color:#aaa;'>—</span>"
        color = "#2e7d32" if v >= 0 else "#d32f2f"
        return f"<span style='color:{color}; font-weight:600;'>{_da_num(v, signed=True)}%</span>"

    def _fmt_diff(a, p):
        if a is None or p is None:
            return "<span style='color:#aaa;'>—</span>"
        d = a - p
        color = "#2e7d32" if d >= 0 else "#d32f2f"
        return (f"<span style='color:{color}; font-weight:700;'>"
                f"{_da_num(d, signed=True)} pp</span>")

    header_cells = "".join(f"<th style='padding:8px; text-align:right;'>{p}</th>" for p in perf_periods)
    asset_cells = "".join(f"<td style='padding:8px; text-align:right;'>{_fmt_pct_signed(asset_returns.get(p))}</td>" for p in perf_periods)
    port_cells = "".join(f"<td style='padding:8px; text-align:right;'>{_fmt_pct_signed(port_returns.get(p))}</td>" for p in perf_periods)
    diff_cells = "".join(f"<td style='padding:8px; text-align:right;'>{_fmt_diff(asset_returns.get(p), port_returns.get(p))}</td>" for p in perf_periods)

    perf_table_html = f"""
    <table style='width:100%; border-collapse:collapse; font-size:14px;'>
      <thead>
        <tr style='border-bottom:1px solid #ddd; color:#666;'>
          <th style='text-align:left; padding:8px;'>Periode</th>
          {header_cells}
        </tr>
      </thead>
      <tbody>
        <tr style='border-bottom:1px solid #f0f0f0;'>
          <td style='padding:8px;'><strong>{ticker}</strong></td>
          {asset_cells}
        </tr>
        <tr style='border-bottom:1px solid #f0f0f0;'>
          <td style='padding:8px;'>Portefølje (TWR)</td>
          {port_cells}
        </tr>
        <tr>
          <td style='padding:8px;'>Differential</td>
          {diff_cells}
        </tr>
      </tbody>
    </table>
    """
    st.markdown(_flatten_html(perf_table_html), unsafe_allow_html=True)
    st.write("---")

    # =====================================================================
    # SEKTION 7: RISIKO / VOLATILITET
    # =====================================================================
    st.subheader("Risiko og volatilitet")

    # Brug daglig prices fra holding-period (PORTFOLIO_START → i dag)
    if ticker in prices.columns:
        hold_prices = prices[ticker].dropna()
        if pd.notnull(first_buy):
            hold_prices = hold_prices[hold_prices.index >= pd.Timestamp(first_buy)]
        daily_rets = hold_prices.pct_change().dropna()
        vol = _annualized_volatility(daily_rets)
        dd_pct, dd_peak, dd_trough = _max_drawdown(hold_prices)
        if len(daily_rets):
            best_day_pct = float(daily_rets.max() * 100)
            best_day_date = daily_rets.idxmax()
            worst_day_pct = float(daily_rets.min() * 100)
            worst_day_date = daily_rets.idxmin()
        else:
            best_day_pct = worst_day_pct = None
            best_day_date = worst_day_date = None
    else:
        vol = dd_pct = best_day_pct = worst_day_pct = None
        dd_peak = dd_trough = best_day_date = worst_day_date = None

    r1, r2, r3, r4 = st.columns(4)
    r1.metric(
        "Volatilitet (ann.)",
        f"{_da_num(vol)}%" if vol is not None else "—",
        help="Annualiseret standardafvigelse af daglige afkast (σ × √252).",
    )
    r2.metric(
        "Maks drawdown",
        f"{_da_num(dd_pct)}%" if dd_pct is not None else "—",
        help="Største pct-fald fra et tidligere højdepunkt i ejertiden.",
    )
    if best_day_pct is not None:
        r3.metric(
            "Bedste dag",
            f"{_da_num(best_day_pct, signed=True)}%",
            delta=pd.Timestamp(best_day_date).strftime("%d. %b %Y"),
            delta_color="off",
        )
    else:
        r3.metric("Bedste dag", "—")
    if worst_day_pct is not None:
        r4.metric(
            "Værste dag",
            f"{_da_num(worst_day_pct, signed=True)}%",
            delta=pd.Timestamp(worst_day_date).strftime("%d. %b %Y"),
            delta_color="off",
        )
    else:
        r4.metric("Værste dag", "—")

    b1, _ = st.columns([1, 3])
    b1.metric("Beta (S&P 500)", _da_num(info["beta"]) if info["beta"] is not None else "—",
              help="Aktiens følsomhed over for det brede marked. <1 = mindre volatil, >1 = mere.")

    if dd_pct is not None and dd_peak is not None and dd_trough is not None:
        st.caption(
            f"Maks drawdown gik fra {pd.Timestamp(dd_peak).strftime('%d. %b %Y')} "
            f"til {pd.Timestamp(dd_trough).strftime('%d. %b %Y')}."
        )

    st.write("---")

    # =====================================================================
    # SEKTION 8: BIDRAG OG RANG
    # =====================================================================
    st.subheader("Bidrag til porteføljen")

    contrib_dkk = unrealized_dkk + realized_dkk
    deposits_total = cashflows.cumsum().iloc[-1] if len(cashflows) else 0
    contrib_pp = (contrib_dkk / deposits_total * 100) if deposits_total else 0

    # Rang blandt aktier baseret på Maks-afkast (pos_pct) — genbruger all_quotes
    asset_ranks = []
    for at in all_active_tickers:
        sub_at = orders_df[orders_df["Ticker"] == at].copy()
        sub_at["Qty_Adj"] = np.where(sub_at["Side"] == "BUY", sub_at["Quantity"], -sub_at["Quantity"])
        sub_at["DKK_Adj"] = np.where(sub_at["Side"] == "BUY", sub_at["Notional, DKK"], -sub_at["Notional, DKK"])
        q_at = float(sub_at["Qty_Adj"].sum())
        c_at = float(sub_at["DKK_Adj"].sum())
        if q_at <= 0.001 or c_at <= 0:
            continue
        ac_at = sub_at["Asset currency"].iloc[0]
        rate_at = usd_dkk_now if ac_at == "USD" else (eur_dkk_now if ac_at == "EUR" else 1.0)
        live_at = all_quotes.get(at, {}).get("live")
        if live_at is None and at in prices.columns:
            ser_at = prices[at].dropna()
            live_at = float(ser_at.iloc[-1]) if len(ser_at) else None
        if live_at is None:
            continue
        v_at = q_at * live_at * rate_at
        ret_pct_at = (v_at - c_at) / c_at * 100
        asset_ranks.append({"ticker": at, "ret_pct": ret_pct_at})

    asset_ranks.sort(key=lambda r: r["ret_pct"], reverse=True)
    rank_idx = next((i for i, r in enumerate(asset_ranks) if r["ticker"] == ticker), -1)
    rank_str = f"{rank_idx + 1}. af {len(asset_ranks)}" if rank_idx >= 0 else "—"

    b1, b2, b3 = st.columns(3)
    b1.metric("Bidrag (DKK)", f"{_da_num(contrib_dkk, signed=True)}")
    b2.metric("Bidrag (%-pt af indskud)", f"{_da_num(contrib_pp, signed=True)}")
    b3.metric("Rang (Maks-afkast)", rank_str)

    # Bar chart
    if asset_ranks:
        bar_colors = ["#1976d2" if r["ticker"] == ticker else "#cccccc" for r in asset_ranks]
        fig_rank = go.Figure(go.Bar(
            x=[r["ret_pct"] for r in asset_ranks],
            y=[r["ticker"] for r in asset_ranks],
            orientation="h",
            marker_color=bar_colors,
            hovertemplate="<b>%{y}</b><br>%{x:.2f}%<extra></extra>",
        ))
        fig_rank.update_layout(
            height=max(220, 28 * len(asset_ranks)),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(title="Maks-afkast (%)", tickformat=".1f"),
            yaxis=dict(autorange="reversed"),
            plot_bgcolor="white", separators=".,",
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    st.write("---")

    # =====================================================================
    # SEKTION 9: HANDELSHISTORIK
    # =====================================================================
    st.subheader("Handelshistorik")
    trades_rows = []
    for _, r in sub.sort_values("Date", ascending=False).iterrows():
        d_str = r["Date"].strftime("%d. %b %Y %H:%M") if pd.notnull(r["Date"]) else "—"
        px_local = r["Notional (account currency)"] / r["Quantity"] if r["Quantity"] else 0
        comm = _safe_float(r.get("Commission (account currency)")) or 0
        ac = r.get("Account currency")
        rate_t = usd_dkk_now if ac == "USD" else (eur_dkk_now if ac == "EUR" else 1.0)
        comm_dkk = comm * rate_t
        trades_rows.append({
            "Dato": d_str,
            "Side": r["Side"],
            "Antal": format_quantity(r["Quantity"]),
            f"Pris ({asset_ccy})": f"{ccy_sym} {_da_num(px_local)}",
            "Notional (DKK)": _da_num(r["Notional, DKK"]),
            "Kurtage (DKK)": _da_num(comm_dkk),
        })
    if trades_rows:
        trades_df_disp = pd.DataFrame(trades_rows)
        st.dataframe(trades_df_disp, use_container_width=True, hide_index=True)

    st.write("---")

    # =====================================================================
    # SEKTION 10: FOOTER / EKSTERNE LINKS
    # =====================================================================
    st.subheader("Mere info")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown(
            f"[📊 Yahoo Finance](https://finance.yahoo.com/quote/{ticker})  \n"
            f"[📈 MarketWatch](https://www.marketwatch.com/investing/stock/{ticker.lower()})"
        )
    with f2:
        meta_lines = []
        if info["isin"]:
            meta_lines.append(f"**ISIN:** {info['isin']}")
        if info["long_name"] and info["long_name"] != name:
            meta_lines.append(f"**Officielt navn:** {info['long_name']}")
        if info["website"]:
            meta_lines.append(f"**Website:** [{info['website']}]({info['website']})")
        if meta_lines:
            st.markdown("  \n".join(meta_lines))


# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("Kontrolpanel")
    uploaded_file = st.file_uploader("Upload kontoudtog (XLSX)", type="xlsx")
    if uploaded_file:
        with open(PERSISTENT_FILE, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("Fil opdateret!")
        st.cache_data.clear()
    st.caption(f"Porteføljestart: {PORTFOLIO_START.date().strftime('%d. %b %Y')}")


# -------------------- HOVEDFLOW --------------------
if not os.path.exists(PERSISTENT_FILE):
    st.info("Upload venligst din Pluto-fil for at se oversigten.")
    st.stop()

try:
    orders_df = pd.read_excel(PERSISTENT_FILE, sheet_name="Orders")
    cash_df = pd.read_excel(PERSISTENT_FILE, sheet_name="Cash overview")
    dkk_tx = pd.read_excel(PERSISTENT_FILE, sheet_name="Transactions - DKK account")
    try:
        usd_tx = pd.read_excel(PERSISTENT_FILE, sheet_name="Transactions - USD account")
    except Exception:
        usd_tx = pd.DataFrame(columns=["Date", "Description", "Amount"])
    try:
        eur_tx = pd.read_excel(PERSISTENT_FILE, sheet_name="Transactions - EUR account")
    except Exception:
        eur_tx = pd.DataFrame(columns=["Date", "Description", "Amount"])
    try:
        positions_df = pd.read_excel(PERSISTENT_FILE, sheet_name="Positions, Ultimo")
        positions_df.columns = positions_df.columns.str.strip()
    except Exception:
        positions_df = pd.DataFrame(columns=["Ticker", "Average entry price (asset currency)"])

    for df in (orders_df, cash_df, dkk_tx, usd_tx, eur_tx):
        df.columns = df.columns.str.strip()
    orders_df["Date"] = pd.to_datetime(orders_df["Date"], dayfirst=True, errors="coerce")
    orders_df["TradeDate"] = orders_df["Date"].dt.normalize()

    today = pd.Timestamp.today().normalize()
    last_data_date = orders_df["TradeDate"].max() if not orders_df.empty else PORTFOLIO_START
    end_date = max(today, last_data_date)
    date_range = pd.date_range(PORTFOLIO_START, end_date, freq="D")

    with st.spinner("Henter kurser og bygger porteføljehistorik..."):
        total_value, stock_value, cash_value_total, holdings, prices, usd_dkk, missing = \
            compute_portfolio_value_series(orders_df, dkk_tx, usd_tx, eur_tx, date_range)
        eur_dkk = prices.get("EURDKK=X", pd.Series(7.46, index=date_range)).reindex(date_range, method="ffill").bfill()
        cashflows, cashflow_fracs = compute_deposits_dkk(dkk_tx, usd_tx, eur_tx, usd_dkk, eur_dkk, date_range)

    # Override seneste punkt i total_value/stock_value med live-priser, så Total
    # porteføljeværdi (top), TWR-afkast og Aktieværdi (DKK, live) alle bruger
    # samme priskilde (live tick incl. pre/post-market) — i stedet for daglig
    # close vs. live tick på forskellige steder.
    try:
        _orders_pre = orders_df.copy()
        _orders_pre["Qty_Adj"] = np.where(
            _orders_pre["Side"] == "BUY", _orders_pre["Quantity"], -_orders_pre["Quantity"]
        )
        _active_pre = (
            _orders_pre.groupby(["Ticker", "Asset currency"])
            .agg(Qty_Adj=("Qty_Adj", "sum")).reset_index()
        )
        _active_pre = _active_pre[_active_pre["Qty_Adj"] > 0.001]
        if not _active_pre.empty:
            _live_top = fetch_live_quotes(tuple(_active_pre["Ticker"].tolist()))
            _live_fx = fetch_live_fx_rates()
            _usd_now = _live_fx.get("USDDKK") or (float(usd_dkk.iloc[-1]) if len(usd_dkk) else 6.85)
            _eur_now = _live_fx.get("EURDKK") or (float(eur_dkk.iloc[-1]) if len(eur_dkk) else 7.46)
            _live_aktier_dkk = 0.0
            for _, _row in _active_pre.iterrows():
                _q = _live_top.get(_row["Ticker"], {})
                _p = _q.get("live")
                if _p is None and _row["Ticker"] in prices.columns:
                    _ser = prices[_row["Ticker"]].dropna()
                    if len(_ser):
                        _p = float(_ser.iloc[-1])
                if _p is None:
                    continue
                _ccy_row = _row["Asset currency"]
                _rate = _usd_now if _ccy_row == "USD" else (_eur_now if _ccy_row == "EUR" else 1.0)
                _live_aktier_dkk += float(_row["Qty_Adj"]) * _p * _rate

            _cash_now_dkk = float(cash_value_total.iloc[-1]) if len(cash_value_total) else 0.0
            total_value = total_value.copy()
            stock_value = stock_value.copy()
            total_value.iloc[-1] = _live_aktier_dkk + _cash_now_dkk
            stock_value.iloc[-1] = _live_aktier_dkk
    except Exception:
        # Hvis live-fetch fejler, fall back til daglige slut-værdier
        pass

    if missing:
        st.warning(f"Kunne ikke hente kurser for: {', '.join(missing)}. Disse positioner indgår ikke i værdiberegningen.")

    # --- Routing: ?ticker=XXX åbner detalje-side ---
    _detail_ticker = st.query_params.get("ticker")
    if _detail_ticker:
        # Beregn live grand_total + total_v til detalje-sidens andels-tal
        _orders_rt = orders_df.copy()
        _orders_rt["Qty_Adj"] = np.where(
            _orders_rt["Side"] == "BUY", _orders_rt["Quantity"], -_orders_rt["Quantity"]
        )
        _orders_rt["DKK_Adj"] = np.where(
            _orders_rt["Side"] == "BUY", _orders_rt["Notional, DKK"], -_orders_rt["Notional, DKK"]
        )
        _portfolio_rt = (
            _orders_rt.groupby(["Ticker", "Asset currency"])
            .agg(Qty_Adj=("Qty_Adj", "sum"), DKK_Adj=("DKK_Adj", "sum"))
            .reset_index()
        )
        _active_rt = _portfolio_rt[_portfolio_rt["Qty_Adj"] > 0.001]
        _live_fx_rt = fetch_live_fx_rates()
        _usd_rt = _live_fx_rt.get("USDDKK") or (float(usd_dkk.iloc[-1]) if len(usd_dkk) else 6.85)
        _eur_rt = _live_fx_rt.get("EURDKK") or (float(eur_dkk.iloc[-1]) if len(eur_dkk) else 7.46)
        _live_q_rt = fetch_live_quotes(tuple(_active_rt["Ticker"].tolist()))
        _total_v_rt = 0.0
        for _, _r_rt in _active_rt.iterrows():
            _p_rt = _live_q_rt.get(_r_rt["Ticker"], {}).get("live")
            if _p_rt is None and _r_rt["Ticker"] in prices.columns:
                _ser_rt = prices[_r_rt["Ticker"]].dropna()
                _p_rt = float(_ser_rt.iloc[-1]) if len(_ser_rt) else None
            if _p_rt is None:
                continue
            _rate_rt = (_usd_rt if _r_rt["Asset currency"] == "USD"
                        else (_eur_rt if _r_rt["Asset currency"] == "EUR" else 1.0))
            _total_v_rt += float(_r_rt["Qty_Adj"]) * _p_rt * _rate_rt
        _cash_rt = (
            float(cash_df[cash_df["Currency"] == "DKK"]["End cash balance"].sum())
            + float(cash_df[cash_df["Currency"] == "USD"]["End cash balance"].sum()) * _usd_rt
            + float(cash_df[cash_df["Currency"] == "EUR"]["End cash balance"].sum()) * _eur_rt
        )
        _grand_total_rt = _total_v_rt + _cash_rt
        render_asset_detail(
            _detail_ticker, orders_df, positions_df, cash_df, prices,
            usd_dkk, eur_dkk, total_value, cashflows, cashflow_fracs,
            _grand_total_rt, _total_v_rt, list(_active_rt["Ticker"]),
        )
        st.stop()

    # 2 tabs i stedet for 3 — Portefølje + Aktier samlet, Pung separat
    tab_main, tab_wallet = st.tabs(["📈 Portefølje", "👛 Pung"])

    # ===== TAB 1: PORTEFØLJE (overblik + graf + aktier) =====
    with tab_main:
        # --- Markedsstatus øverst til højre ---
        ms_code, ms_label, ms_emoji, ms_bg = get_us_market_status()

        # Beregn CET-tider for hver US-session, robust mod sommertid
        try:
            _et_now = pd.Timestamp.now(tz="America/New_York")
            _cph_now = pd.Timestamp.now(tz="Europe/Copenhagen")
            _offset_hours = int(round(
                (_cph_now.utcoffset() - _et_now.utcoffset()).total_seconds() / 3600
            ))
        except Exception:
            _offset_hours = 6  # fallback til standard CET-ET-forskel

        def _et_to_cet(h, m=0):
            total_min = h * 60 + m + _offset_hours * 60
            return f"{(total_min // 60) % 24:02d}:{total_min % 60:02d}"

        _sessions = [
            ("pre",       "Pre-market",  "🌅", _et_to_cet(4),     _et_to_cet(9, 30)),
            ("regular",   "Regular",     "🟢", _et_to_cet(9, 30), _et_to_cet(16)),
            ("post",      "After-hours", "🌆", _et_to_cet(16),    _et_to_cet(20)),
            ("overnight", "Overnight",   "🌙", _et_to_cet(20),    _et_to_cet(4)),
        ]

        _session_rows = ""
        for _code, _label, _emoji, _start, _end in _sessions:
            _is_active = (_code == ms_code)
            _row_style = (
                "padding:3px 8px; border-radius:4px; "
                + ("font-weight:600; background:rgba(0,0,0,0.06);" if _is_active else "color:#555;")
            )
            _session_rows += (
                f"<div style='{_row_style}'>{_emoji} {_label}: {_start} – {_end}</div>"
            )

        _card_html = (
            f"<div style='text-align:right;'>"
            f"<div style='display:inline-block; background:{ms_bg}; padding:8px 14px;"
            f" border-radius:12px; font-size:13px; text-align:left; min-width:220px;'>"
            f"<div style='font-weight:600; margin-bottom:6px;'>"
            f"{ms_emoji} US-marked: {ms_label}"
            f"</div>"
            f"<div style='font-size:12px;'>{_session_rows}</div>"
            f"</div></div>"
        )

        # --- Periode-vælger + åbningstider i samme række ---
        col_period, col_hours = st.columns([3, 1])
        with col_period:
            period = st.radio(
                "Periode", ["1D", "1U", "1M", "3M", "6M", "YTD", "1Å", "3Å", "5Å", "Maks"],
                horizontal=True, index=9, label_visibility="collapsed"
            )
        with col_hours:
            st.markdown(_flatten_html(_card_html), unsafe_allow_html=True)
        # Periode-serier: intraday for 1D/1U/1M, ellers daglig
        fracs_per = None
        if period in ["1D", "1U", "1M"]:
            v_intraday, cf_intraday = compute_portfolio_value_series_intraday(orders_df, dkk_tx, usd_tx, eur_tx, period)
            if len(v_intraday) >= 2:
                v_per = v_intraday
                cf_per = cf_intraday.reindex(v_intraday.index).fillna(0)
                d_per = v_intraday.index
            else:
                v_per, cf_per, d_per = slice_period(total_value, cashflows, period)
                fracs_per = cashflow_fracs.reindex(d_per).values if cashflow_fracs is not None else None
        else:
            v_per, cf_per, d_per = slice_period(total_value, cashflows, period)
            fracs_per = cashflow_fracs.reindex(d_per).values if cashflow_fracs is not None else None

        ret_dkk_series, ret_pct_series = cumulative_return_series(
            v_per.values, cf_per.values, d_per, cashflow_fracs=fracs_per
        )
        ret_dkk_now = ret_dkk_series.iloc[-1] if len(ret_dkk_series) else 0
        ret_pct_now = ret_pct_series.iloc[-1] if len(ret_pct_series) else 0

        # --- Hovedtal ---
        st.caption("Total porteføljeværdi")
        st.markdown(
            f"<p class='big-value' style='font-size: 2.25rem; font-weight: 700;'>"
            f"{_da_num(total_value.iloc[-1])} DKK</p>",
            unsafe_allow_html=True,
        )
        cls = "return-pos" if ret_dkk_now >= 0 else "return-neg"
        st.markdown(
            f"<p class='{cls}'>{_da_num(ret_dkk_now, signed=True)} DKK "
            f"({_da_num(ret_pct_now, signed=True)}%)</p>",
            unsafe_allow_html=True,
        )
        st.caption(f"Afkast for perioden: {period} • Tidsvægtet afkast (TWR, Modified Dietz), korrigeret for ind- og udbetalinger")

        # Nøgletal beregnes her — bruges både i højre kolonne og senere sektioner
        total_deposits = cashflows.cumsum().iloc[-1]

        # --- Graf + nøgletal side om side ---
        col_chart, col_metrics = st.columns([4, 1])

        with col_chart:
            # --- Graf: Afkast over tid ---
            # Grøn for positive segmenter, rød for negative. Vi bygger én trace per
            # kontinuerlig segment (split ved hver zero-crossing) for at undgå at
            # Plotlys fill="tozeroy" bleder mellem segmenter.
            st.write("")
            _y_arr = np.asarray(ret_pct_series.values, dtype=float)
            _x_arr = pd.DatetimeIndex(d_per).to_numpy()

            # Byg ekspanderede serier med interpolerede zero-crossings
            _ex_list = []
            _ey_list = []
            for _i in range(len(_y_arr)):
                _ex_list.append(_x_arr[_i])
                _ey_list.append(_y_arr[_i])
                if _i + 1 < len(_y_arr):
                    _y0, _y1 = _y_arr[_i], _y_arr[_i + 1]
                    if (_y0 > 0 and _y1 < 0) or (_y0 < 0 and _y1 > 0):
                        _t0 = pd.Timestamp(_x_arr[_i]).value
                        _t1 = pd.Timestamp(_x_arr[_i + 1]).value
                        _frac = _y0 / (_y0 - _y1)
                        _ex_list.append(pd.Timestamp(int(_t0 + _frac * (_t1 - _t0))).to_numpy())
                        _ey_list.append(0.0)

            # Split i kontinuerlige sign-segmenter
            # Hver segment er en liste af (x, y) der enten er alle ≥0 eller alle ≤0
            _segments = []  # liste af (sign, [(x, y), ...])
            if len(_ex_list) > 0:
                _cur_sign = None  # "pos", "neg", eller None
                _cur_seg = []
                for _x, _y in zip(_ex_list, _ey_list):
                    if _y > 0:
                        _new_sign = "pos"
                    elif _y < 0:
                        _new_sign = "neg"
                    else:
                        _new_sign = None  # zero — ambivalent, tilhører begge
                    # Zero punkter (crossings og første-punkt) tilføjes til både slutning
                    # af forrige segment og start af næste
                    if _new_sign is None:
                        if _cur_seg:
                            _cur_seg.append((_x, _y))
                            _segments.append((_cur_sign, _cur_seg))
                        # Start nyt segment med dette zero-point — sign bliver afgjort af næste punkt
                        _cur_seg = [(_x, _y)]
                        _cur_sign = None
                    elif _cur_sign is None or _cur_sign == _new_sign:
                        _cur_seg.append((_x, _y))
                        _cur_sign = _new_sign
                    else:
                        # Sign-skift uden zero (burde ikke ske efter interpolation, men safety)
                        _segments.append((_cur_sign, _cur_seg))
                        _cur_seg = [(_x, _y)]
                        _cur_sign = _new_sign
                if _cur_seg:
                    _segments.append((_cur_sign, _cur_seg))

            fig = go.Figure()
            for _sign, _seg in _segments:
                if len(_seg) < 2:
                    continue
                _xs = [p[0] for p in _seg]
                _ys = [p[1] for p in _seg]
                if _sign == "neg":
                    _color = "#d32f2f"
                    _fill = "rgba(211,47,47,0.12)"
                else:
                    _color = "#2e7d32"
                    _fill = "rgba(46,125,50,0.12)"
                fig.add_trace(go.Scatter(
                    x=_xs, y=_ys,
                    mode="lines",
                    line=dict(color=_color, width=2.5),
                    fill="tozeroy",
                    fillcolor=_fill,
                    hovertemplate="<b>%{x|%d. %b %Y %H:%M}</b><br>Afkast: %{y:.2f}%<extra></extra>",
                    showlegend=False,
                ))
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.3)")
            fig.update_layout(
                height=380,
                margin=dict(l=0, r=0, t=10, b=0),
                yaxis=dict(title="Afkast (%)", tickformat=".2f"),
                xaxis=dict(title=""),
                # "closest" i stedet for "x unified": multi-trace-tilgangen (én trace
                # pr. sign-segment) gjorde at zero-crossings viste to tooltips på
                # samme tid. "closest" snapper til nærmeste punkt og viser kun ét.
                hovermode="closest",
                plot_bgcolor="white",
                separators=".,",
            )
            st.plotly_chart(fig, use_container_width=True)

            with st.expander("Vis porteføljeværdi (DKK) over tid"):
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(
                    x=d_per, y=v_per.values,
                    mode="lines", line=dict(color="#1976d2", width=2),
                    name="Porteføljeværdi",
                    hovertemplate="<b>%{x|%d. %b %Y}</b><br>Værdi: %{y:,.2f} DKK<extra></extra>",
                ))
                cum_cf_series = (cf_per.cumsum() - cf_per.iloc[0])
                fig2.add_trace(go.Scatter(
                    x=d_per,
                    y=v_per.iloc[0] + cum_cf_series,
                    mode="lines", line=dict(color="rgba(0,0,0,0.4)", width=1.5, dash="dash"),
                    name="Investeret kapital",
                    hovertemplate="<b>%{x|%d. %b %Y}</b><br>Investeret: %{y:,.2f} DKK<extra></extra>",
                ))
                fig2.update_layout(
                    height=320, margin=dict(l=0, r=0, t=10, b=0),
                    yaxis=dict(title="DKK", tickformat=",.0f"),
                    hovermode="x unified", plot_bgcolor="white",
                    separators=".,",
                )
                st.plotly_chart(fig2, use_container_width=True)

        with col_metrics:
            st.metric("Aktier (DKK)", _da_num(stock_value.iloc[-1]))
            st.metric("Kontant (DKK)", _da_num(cash_value_total.iloc[-1]))
            st.metric("Total nettoindskud", f"{_da_num(total_deposits)} DKK")
            st.metric("Akkum. afkast (Maks)", f"{_da_num(total_value.iloc[-1] - total_deposits)} DKK")

        # --- Aktier-sektion ---
        st.write("")
        st.write("---")
        st.subheader("Mine Aktier")

        afk_col_label, afk_col_radio = st.columns([1, 3])
        with afk_col_label:
            st.caption("Afkast-periode")
        with afk_col_radio:
            afkast_periode = st.radio(
                "Afkast-periode for tabel-kolonnerne",
                ["1D", "YTD", "Maks"],
                horizontal=True,
                label_visibility="collapsed",
                index=2,
                key="afkast_periode",
            )
        ytd_start = pd.Timestamp(year=pd.Timestamp.now().year, month=1, day=1)

        orders_df["Qty_Adj"] = np.where(orders_df["Side"] == "BUY", orders_df["Quantity"], -orders_df["Quantity"])
        orders_df["DKK_Adj"] = np.where(orders_df["Side"] == "BUY", orders_df["Notional, DKK"], -orders_df["Notional, DKK"])

        portfolio = (
            orders_df.groupby(["Ticker", "Name", "Asset currency"])
            .agg(Qty_Adj=("Qty_Adj", "sum"), DKK_Adj=("DKK_Adj", "sum"))
            .reset_index()
        )
        active = portfolio[portfolio["Qty_Adj"] > 0.001].copy()

        avg_entry_by_ticker = {}
        if "Ticker" in positions_df.columns and "Average entry price (asset currency)" in positions_df.columns:
            for _, row in positions_df.iterrows():
                avg_entry_by_ticker[row["Ticker"]] = _safe_float(row["Average entry price (asset currency)"])

        if active.empty:
            st.info("Ingen aktive positioner.")
        else:
            _live_fx_main = fetch_live_fx_rates()
            usd_dkk_now = _live_fx_main.get("USDDKK") or float(usd_dkk.iloc[-1])
            eur_dkk_now = _live_fx_main.get("EURDKK") or float(eur_dkk.iloc[-1])

            with st.spinner("Henter live-kurser..."):
                live_quotes = fetch_live_quotes(tuple(active["Ticker"].tolist()))

            with st.spinner("Henter sektor- og aktivklasse-info..."):
                ticker_meta = fetch_ticker_meta(tuple(active["Ticker"].tolist()))

            sparklines_map = fetch_intraday_sparklines(tuple(active["Ticker"].tolist()))

            session_short_map = {
                "weekend":   "Weekend",
                "overnight": "Overnight",
                "pre":       "Pre-market",
                "regular":   "Åbent",
                "post":      "After-hours",
                "closed":    "Lukket",
            }

            def session_cell_for(asset_ccy):
                code, _label, emoji, _bg = get_market_status_for_currency(asset_ccy)
                return f"{emoji} {session_short_map.get(code, '—')}"

            rows = []
            total_v = 0.0
            per_sector = {}
            per_asset_class = {}
            per_currency = {}
            per_region = {}
            per_country = {}
            all_holdings = []
            for _, s in active.iterrows():
                t = s["Ticker"]
                q = live_quotes.get(t, {})
                ccy = s["Asset currency"]

                fallback_close = None
                if t in prices.columns:
                    ser = prices[t].dropna()
                    if len(ser) >= 1:
                        fallback_close = float(ser.iloc[-1])

                # 'Sidste luk' = den seneste KOMPLETTE regular session close.
                # I regular hours (today_close = None): = gårsdagens close (prev_close)
                # I after-hours/overnight/pre-market: = dagens close (today_close).
                # 'Δ sidste luk' beregnes mod den foregående close, så Δ altid
                # afspejler den daglige ændring fra Sidste luk vs. dagen før.
                # Matcher Yahoo's adfærd hvor "previous close"-info skifter til
                # dagens close efter regular session er afsluttet.
                _today_c = q.get("today_close")
                _prev_c = q.get("prev_close")
                _prev_prev_c = q.get("prev_prev_close")
                if _today_c is not None:
                    sidste_luk = _today_c
                    forrige_luk = _prev_c
                else:
                    sidste_luk = _prev_c
                    forrige_luk = _prev_prev_c
                live = q.get("live")

                if sidste_luk is None:
                    sidste_luk = fallback_close
                if live is None:
                    live = sidste_luk

                if sidste_luk is None or live is None:
                    continue

                if forrige_luk is not None and forrige_luk != 0:
                    d_yest = sidste_luk - forrige_luk
                    d_yest_pct = d_yest / forrige_luk * 100
                    d_yest_str = f"{_da_num(d_yest, signed=True)} ({_da_num(d_yest_pct, signed=True)}%)"
                else:
                    d_yest_str = "—"

                if sidste_luk:
                    d_now = live - sidste_luk
                    d_now_pct = d_now / sidste_luk * 100
                    d_now_str = f"{_da_num(d_now, signed=True)} ({_da_num(d_now_pct, signed=True)}%)"
                else:
                    d_now_str = "—"

                if ccy == "USD":
                    rate = usd_dkk_now
                elif ccy == "EUR":
                    rate = eur_dkk_now
                else:
                    rate = 1.0

                gak_pluto = avg_entry_by_ticker.get(t)
                if gak_pluto is not None:
                    gak_valuta = gak_pluto
                else:
                    gak_valuta = (s["DKK_Adj"] / s["Qty_Adj"]) / rate if rate else 0
                vaerdi_dkk = s["Qty_Adj"] * live * rate
                pos_pct = (vaerdi_dkk - s["DKK_Adj"]) / s["DKK_Adj"] * 100 if s["DKK_Adj"] else 0
                total_v += vaerdi_dkk

                # Periodisk afkast (kr. + %) afhængigt af afkast_periode-vælger
                if afkast_periode == "1D":
                    if sidste_luk and live is not None and sidste_luk > 0:
                        pnl_pos_pct = (live - sidste_luk) / sidste_luk * 100
                        pnl_pos_dkk = float(s["Qty_Adj"]) * (live - sidste_luk) * rate
                    else:
                        pnl_pos_pct = 0.0
                        pnl_pos_dkk = 0.0
                elif afkast_periode == "YTD":
                    first_buy = orders_df[orders_df["Ticker"] == t]["TradeDate"].min()
                    used_jan1 = False
                    if pd.notnull(first_buy) and first_buy < ytd_start and t in prices.columns:
                        jan1_close = prices[t].asof(ytd_start)
                        if pd.notnull(jan1_close) and float(jan1_close) > 0 and live is not None:
                            jan1_close_f = float(jan1_close)
                            pnl_pos_pct = (live - jan1_close_f) / jan1_close_f * 100
                            pnl_pos_dkk = float(s["Qty_Adj"]) * (live - jan1_close_f) * rate
                            used_jan1 = True
                    if not used_jan1:
                        # Position købt efter 1. jan eller manglende jan1-pris → fallback til "siden køb"
                        pnl_pos_dkk = vaerdi_dkk - float(s["DKK_Adj"])
                        pnl_pos_pct = pos_pct
                else:  # Maks
                    pnl_pos_dkk = vaerdi_dkk - float(s["DKK_Adj"])
                    pnl_pos_pct = pos_pct

                pos_entry = {
                    "ticker": t,
                    "name": s["Name"],
                    "qty": float(s["Qty_Adj"]),
                    "value_dkk": vaerdi_dkk,
                    "invested_dkk": float(s["DKK_Adj"]),
                    "gain_dkk": vaerdi_dkk - float(s["DKK_Adj"]),
                    "pos_pct": pos_pct,
                }
                all_holdings.append(pos_entry)

                meta = ticker_meta.get(t, {})

                def _add_to(bucket_dict, key):
                    b = bucket_dict.setdefault(
                        key, {"count": 0, "value_dkk": 0.0, "invested_dkk": 0.0, "positions": []}
                    )
                    b["count"] += 1
                    b["value_dkk"] += vaerdi_dkk
                    b["invested_dkk"] += float(s["DKK_Adj"])
                    b["positions"].append(pos_entry)

                _add_to(per_sector, meta.get("sector", "Andet"))
                _add_to(per_asset_class, meta.get("asset_class", "Andet"))
                _add_to(per_currency, ccy)
                _add_to(per_region, meta.get("region", "Andet"))
                _add_to(per_country, meta.get("country", "Andet"))

                afkast_kr_label = "Afkast (kr.)" if afkast_periode == "Maks" else f"Afkast {afkast_periode} (kr.)"
                afkast_pct_label = "Pos. afkast" if afkast_periode == "Maks" else f"Pos. afkast {afkast_periode}"

                # Sparkline: regular-hours intraday-graf for seneste handelsdag,
                # med stiplet baseline ved forrige close. Linjen farves grøn over
                # baseline og rød under, splittet ved hver krydsning.
                _spark_vals = sparklines_map.get(t, [])
                _spark_url = _make_sparkline_data_url(_spark_vals, ref_value=forrige_luk)

                rows.append({
                    "Navn": s["Name"][:32],
                    "Ticker": t,
                    "🔍": f"?ticker={t}",
                    "1D": _spark_url,
                    "Antal": format_quantity(s["Qty_Adj"]),
                    "GAK": format_currency(gak_valuta, ccy),
                    "Sidste luk": format_currency(sidste_luk, ccy),
                    "Δ sidste luk": d_yest_str,
                    "Aktuel": format_currency(live, ccy),
                    "Δ aktuel": f"{d_now_str} {session_cell_for(ccy)}",
                    "Værdi (DKK)": format_currency(vaerdi_dkk, "DKK"),
                    afkast_kr_label: f"{_da_num(pnl_pos_dkk, signed=True)} kr.",
                    afkast_pct_label: f"{_da_num(pnl_pos_pct, signed=True)}%",
                })

            df_disp = pd.DataFrame(rows)
            afkast_kr_label = "Afkast (kr.)" if afkast_periode == "Maks" else f"Afkast {afkast_periode} (kr.)"
            afkast_pct_label = "Pos. afkast" if afkast_periode == "Maks" else f"Pos. afkast {afkast_periode}"
            styled = df_disp.style.map(
                color_change_str,
                subset=["Δ sidste luk", "Δ aktuel", afkast_kr_label, afkast_pct_label],
            )
            # Beregn tabel-højde så alle rækker vises uden indre scrollbar.
            # ~35 px pr. række + ~38 px til header. Justér op hvis Streamlit
            # ændrer rækkehøjde i en fremtidig version.
            _table_height = 35 * max(1, len(df_disp)) + 38
            st.dataframe(
                styled,
                use_container_width=True,
                hide_index=True,
                height=_table_height,
                column_config={
                    "1D": st.column_config.ImageColumn(
                        "1D",
                        help="Intraday-bevægelse for seneste US-handelsdag (5-min bars). Grøn hvis dagen var op, rød hvis ned.",
                        width="small",
                    ),
                    "🔍": st.column_config.LinkColumn(
                        "🔍",
                        display_text="Åbn",
                        help="Klik for detaljer. Ctrl/Cmd-klik åbner i ny fane.",
                        width="small",
                    ),
                },
                column_order=[
                    "Navn", "Ticker", "🔍", "1D", "Antal", "GAK",
                    "Sidste luk", "Δ sidste luk", "Aktuel", "Δ aktuel",
                    "Værdi (DKK)", afkast_kr_label, afkast_pct_label,
                ],
            )

            periode_forklaring = {
                "1D": "afspejler dagens ændring (sidste luk → live).",
                "YTD": "afspejler årets ændring (1. januar → live; for positioner købt efter 1. jan: siden køb).",
                "Maks": "er kostbasis-afkast pr. aktie (siden køb) — det officielle TWR-afkast vises øverst på siden.",
            }
            st.caption(
                "Live-priser opdateres hvert 60. sek. *Sidste luk* er den seneste officielle slutkurs. "
                "*Δ sidste luk* = ændring fra forrige handelsdag til seneste luk. "
                "*Δ aktuel* = ændring fra sidste luk til den aktuelle pris, efterfulgt af session-indikator "
                "for det marked aktivet handles på (US- eller EU-børs ud fra noteringsvaluta). "
                f"*{afkast_kr_label}* og *{afkast_pct_label}* {periode_forklaring[afkast_periode]}"
            )
            total_i = active["DKK_Adj"].sum()

            # ----- Beregn omkostninger (kurtage + FX-spread) og realiseret afkast -----
            # Pluto's prismodel er fast: kurtage 0,10% direkte fra XLSX "Commission",
            # FX-spread 0,15% af "Notional, DKK" på USD/EUR-handler. Verificeret mod
            # AEHR-handel 1/5/2026: Pluto-rate 6,377302 vs mid 6,3678 = 0,15% spread.
            total_commission_dkk = 0.0
            total_fx_spread_dkk = 0.0
            for _, _ord in orders_df.iterrows():
                _comm = _safe_float(_ord.get("Commission (account currency)"))
                if _comm and _comm != 0:
                    _acc = _ord.get("Account currency")
                    if _acc == "USD":
                        _r = usd_dkk_now or 6.85
                    elif _acc == "EUR":
                        _r = eur_dkk_now or 7.46
                    else:
                        _r = 1.0
                    total_commission_dkk += _comm * _r

                # FX-spread: 0,15% af Notional, DKK for ikke-DKK handler (BUYS+SELLS)
                _ac = _ord.get("Asset currency")
                if _ac in ("USD", "EUR"):
                    _n_dkk = _safe_float(_ord.get("Notional, DKK"))
                    if _n_dkk:
                        total_fx_spread_dkk += abs(_n_dkk) * PLUTO_FX_SPREAD_RATE

            total_costs_dkk = total_commission_dkk + total_fx_spread_dkk

            # Cash til realiseret-beregning
            _cash_real_now = float(cash_value_total.iloc[-1]) if len(cash_value_total) else 0.0
            unrealized_pnl = total_v - total_i
            realized_all_in = total_i + _cash_real_now - total_deposits
            # Pure realized = realized minus costs (så omkostninger vises separat)
            realized_pnl = realized_all_in + total_costs_dkk
            # Reelt investeret = nettoindskud minus betalte omkostninger
            reelt_investeret = total_deposits - total_costs_dkk

            unrealized_pct = (unrealized_pnl / total_i * 100) if total_i else 0
            realized_pct = (realized_pnl / total_deposits * 100) if total_deposits else 0

            st.write("")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric(
                "Reelt investeret",
                f"{_da_num(reelt_investeret)} DKK",
                delta=f"af {_da_num(total_deposits)} DKK indskud",
                delta_color="off",
            )
            k2.metric(
                "Samlede omkostninger",
                f"{_da_num(total_costs_dkk)} DKK",
                delta=f"kurtage {_da_num(total_commission_dkk)} + FX {_da_num(total_fx_spread_dkk)}",
                delta_color="off",
            )
            k3.metric(
                "Urealiseret afkast",
                f"{_da_num(unrealized_pnl)} DKK",
                delta=f"{_da_num(unrealized_pct, signed=True)}%" if total_i else "—",
            )
            k4.metric(
                "Realiseret afkast",
                f"{_da_num(realized_pnl)} DKK",
                delta=f"{_da_num(realized_pct, signed=True)}%" if total_deposits else "—",
            )
            st.caption(
                "*Reelt investeret* = nettoindskud minus samlede omkostninger (det beløb der reelt nåede markedet). "
                "*Samlede omkostninger* = sum af kurtage (fra XLSX) + FX-spread (0,15% af handelsbeløb i USD/EUR). "
                "*Urealiseret afkast* = aktuel værdi minus kostbasis af aktuelle positioner (relativ til kostbasis). "
                "*Realiseret afkast* = trading-P/L fra lukkede/delvist solgte positioner — eksklusiv omkostninger (relativ til nettoindskud)."
            )

            # ----- Cash-håndtering: tilføj cash til relevante buckets -----
            usd_dkk_rate = float(usd_dkk.iloc[-1]) if len(usd_dkk) else 6.85
            eur_dkk_rate = float(eur_dkk.iloc[-1]) if len(eur_dkk) else 7.46
            cash_currency_meta = [
                ("DKK", "Danske Kroner", 1.0),
                ("USD", "US Dollar", usd_dkk_rate),
                ("EUR", "Euro", eur_dkk_rate),
            ]
            cash_entries = []
            for cur, cur_name, rate in cash_currency_meta:
                bal = cash_df[cash_df["Currency"] == cur]["End cash balance"].sum()
                bal_dkk = float(bal) * rate
                if bal_dkk > 0.01:
                    cash_entries.append({
                        "ticker": cur,
                        "name": cur_name,
                        "qty": float(bal),
                        "value_dkk": bal_dkk,
                        "invested_dkk": bal_dkk,  # cash = no gain
                        "gain_dkk": 0.0,
                        "pos_pct": 0.0,
                        "hide_zero_gain": True,
                    })

            # Tilføj cash til Aktivklasser (én samlet "Kontanter"-bucket), Valutaer (per ccy)
            # og all_holdings (én entry pr. ccy). Udeladt fra Sektorer/Regioner/Lande.
            if cash_entries:
                cash_total = sum(c["value_dkk"] for c in cash_entries)
                per_asset_class["Kontanter"] = {
                    "count": len(cash_entries),
                    "value_dkk": cash_total,
                    "invested_dkk": cash_total,
                    "positions": cash_entries[:],
                }
                for ce in cash_entries:
                    cur = ce["ticker"]
                    cb = per_currency.setdefault(
                        cur, {"count": 0, "value_dkk": 0.0, "invested_dkk": 0.0, "positions": []}
                    )
                    cb["count"] += 1
                    cb["value_dkk"] += ce["value_dkk"]
                    cb["invested_dkk"] += ce["invested_dkk"]
                    cb["positions"].append(ce)
                    all_holdings.append(ce)

            grand_total = total_v + sum(c["value_dkk"] for c in cash_entries)

            # ----- Alle beholdninger (donut + flad liste) -----
            if all_holdings:
                all_sorted = sorted(all_holdings, key=lambda h: h["value_dkk"], reverse=True)
                hold_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Pastel
                hold_color_map = {h["ticker"]: hold_palette[i % len(hold_palette)]
                                  for i, h in enumerate(all_sorted)}

                st.write("")
                st.write("---")
                st.subheader("Alle beholdninger")

                ah_col_chart, ah_col_list = st.columns([1, 2])

                with ah_col_chart:
                    fig_ah = go.Figure(go.Pie(
                        labels=[h["ticker"] for h in all_sorted],
                        values=[h["value_dkk"] for h in all_sorted],
                        hole=0.55,
                        marker=dict(
                            colors=[hold_color_map[h["ticker"]] for h in all_sorted],
                            line=dict(color="white", width=2),
                        ),
                        textinfo="percent",
                        textposition="outside",
                        texttemplate="%{percent:.1%}",
                        hovertemplate="<b>%{label}</b><br>%{customdata}<br>%{value:,.2f} DKK (%{percent})<extra></extra>",
                        customdata=[h["name"] for h in all_sorted],
                    ))
                    fig_ah.update_layout(
                        height=420, showlegend=False,
                        margin=dict(l=20, r=20, t=10, b=10),
                        paper_bgcolor="white", plot_bgcolor="white",
                        separators=".,",
                    )
                    st.plotly_chart(fig_ah, use_container_width=True)

                with ah_col_list:
                    ah_html = ""
                    for h in all_sorted:
                        c = hold_color_map[h["ticker"]]
                        alloc = (h["value_dkk"] / grand_total * 100) if grand_total else 0
                        ah_html += f"""
                        <tr>
                          <td style='padding:8px 12px; width:64px;'>
                            <span style='display:inline-block; width:48px; padding:3px 6px;
                                         border-radius:4px; background:{c}; color:white;
                                         font-size:11px; font-weight:600; text-align:center;'>
                              {h['ticker']}
                            </span>
                          </td>
                          <td style='padding:8px;'>{h['name']}</td>
                          <td style='padding:8px; text-align:right;'>
                            <strong>{_da_num(alloc)}%</strong>
                          </td>
                        </tr>
                        """
                    ah_table_html = f"""
                    <table style='width:100%; border-collapse:collapse; font-size:14px;'>
                      <tbody>{ah_html}</tbody>
                    </table>
                    """
                    st.markdown(_flatten_html(ah_table_html), unsafe_allow_html=True)

            # ----- Fordelinger (tabbed) -----
            def _build_rows(buckets_dict, total_for_alloc, *, emoji_map=None):
                """Konverterer en bucket-dict til en sorteret liste til render_breakdown."""
                rows_out = []
                for name, d in buckets_dict.items():
                    rows_out.append({
                        "name": name,
                        "emoji": (emoji_map or {}).get(name, ""),
                        "count": d["count"],
                        "value_dkk": d["value_dkk"],
                        "invested_dkk": d["invested_dkk"],
                        "gain_dkk": d["value_dkk"] - d["invested_dkk"],
                        "gain_pct": ((d["value_dkk"] - d["invested_dkk"]) / d["invested_dkk"] * 100)
                                    if d["invested_dkk"] else 0,
                        "alloc_pct": (d["value_dkk"] / total_for_alloc * 100) if total_for_alloc else 0,
                        "hide_zero_gain": name in ("Kontanter", "USD", "EUR", "DKK"),
                    })
                rows_out.sort(key=lambda r: r["value_dkk"], reverse=True)
                return rows_out

            def _make_color_map(rows_list, palette):
                return {r["name"]: palette[i % len(palette)] for i, r in enumerate(rows_list)}

            if per_sector or per_asset_class or per_currency or per_region or per_country:
                st.write("")
                st.write("---")
                st.subheader("Fordelinger")

                tab_sec, tab_ac, tab_ccy, tab_reg, tab_country = st.tabs(
                    ["Sektorer", "Aktivklasser", "Valutaer", "Regioner", "Lande"]
                )

                # === SEKTORER ===
                with tab_sec:
                    if per_sector:
                        sector_rows = _build_rows(per_sector, total_v)
                        sec_color_map = _make_color_map(sector_rows, px.colors.qualitative.Set2)

                        col_chart, col_list = st.columns([1, 2])
                        with col_chart:
                            chart_type = st.radio(
                                "Visning",
                                ["Sektor", "Sektor + positioner"],
                                horizontal=True,
                                label_visibility="collapsed",
                                key="sector_chart_type",
                            )
                            if chart_type == "Sektor + positioner":
                                sb_labels, sb_parents, sb_values, sb_ids, sb_colors = [], [], [], [], []
                                for r in sector_rows:
                                    sb_labels.append(r["name"])
                                    sb_parents.append("")
                                    sb_values.append(r["value_dkk"])
                                    sb_ids.append(r["name"])
                                    sb_colors.append(sec_color_map[r["name"]])
                                for r in sector_rows:
                                    light = _lighten_rgb(sec_color_map[r["name"]])
                                    for p in sorted(per_sector[r["name"]]["positions"],
                                                    key=lambda x: x["value_dkk"], reverse=True):
                                        sb_labels.append(p["ticker"])
                                        sb_parents.append(r["name"])
                                        sb_values.append(p["value_dkk"])
                                        sb_ids.append(f"{r['name']}::{p['ticker']}")
                                        sb_colors.append(light)
                                fig_chart = go.Figure(go.Sunburst(
                                    labels=sb_labels, parents=sb_parents, values=sb_values,
                                    ids=sb_ids, branchvalues="total",
                                    marker=dict(colors=sb_colors, line=dict(color="white", width=2)),
                                    hovertemplate="<b>%{label}</b><br>%{value:,.2f} DKK<extra></extra>",
                                    insidetextfont=dict(size=11),
                                ))
                            else:
                                fig_chart = go.Figure(go.Pie(
                                    labels=[r["name"] for r in sector_rows],
                                    values=[r["value_dkk"] for r in sector_rows],
                                    hole=0.6,
                                    marker=dict(
                                        colors=[sec_color_map[r["name"]] for r in sector_rows],
                                        line=dict(color="white", width=2),
                                    ),
                                    textinfo="none",
                                    hovertemplate="<b>%{label}</b><br>%{value:,.2f} DKK (%{percent})<extra></extra>",
                                ))
                            fig_chart.update_layout(
                                height=380, showlegend=False,
                                margin=dict(l=0, r=0, t=0, b=0),
                                paper_bgcolor="white", plot_bgcolor="white",
                                separators=".,",
                            )
                            st.plotly_chart(fig_chart, use_container_width=True)

                        with col_list:
                            sec_html = ""
                            for r in sector_rows:
                                c = sec_color_map[r["name"]]
                                gc = "#2e7d32" if r["gain_dkk"] >= 0 else "#d32f2f"
                                arrow = "▲" if r["gain_dkk"] >= 0 else "▼"
                                sec_html += f"""
                                <tr>
                                  <td style='border-left:4px solid {c}; padding:10px 0 10px 14px;'>
                                    <strong>{r['name']}</strong><br>
                                    <small style='color:#888;'>{r['count']} positioner</small>
                                  </td>
                                  <td style='padding:10px;'>
                                    <strong>{_da_num(r['value_dkk'])}</strong><br>
                                    <small style='color:#888;'>{_da_num(r['invested_dkk'])}</small>
                                  </td>
                                  <td style='padding:10px; color:{gc};'>
                                    {_da_num(r['gain_dkk'], signed=True)}<br>
                                    <small>{arrow} {_da_num(r['gain_pct'], signed=True)}%</small>
                                  </td>
                                  <td style='padding:10px; text-align:right;'>
                                    <strong>{_da_num(r['alloc_pct'])}%</strong>
                                  </td>
                                </tr>
                                """
                            sec_table_html = f"""
                            <table style='width:100%; border-collapse:collapse; font-size:14px;'>
                              <thead>
                                <tr style='border-bottom:1px solid #ddd; color:#666;'>
                                  <th style='text-align:left; padding:8px 0 8px 14px;'>Navn</th>
                                  <th style='text-align:left; padding:8px;'>Værdi/Investeret</th>
                                  <th style='text-align:left; padding:8px;'>Gevinst</th>
                                  <th style='text-align:right; padding:8px;'>Allokering</th>
                                </tr>
                              </thead>
                              <tbody>{sec_html}</tbody>
                            </table>
                            """
                            st.markdown(_flatten_html(sec_table_html), unsafe_allow_html=True)

                        render_drilldown(sector_rows, per_sector, sec_color_map)
                    else:
                        st.info("Ingen aktiepositioner at vise sektorfordeling for.")

                # === AKTIVKLASSER ===
                with tab_ac:
                    if per_asset_class:
                        ac_emoji = {
                            "Aktier": "📈", "Fonde / ETF'er": "📊",
                            "Krypto": "🪙", "Kontanter": "💰", "Andet": "❔",
                        }
                        ac_rows = _build_rows(per_asset_class, grand_total, emoji_map=ac_emoji)
                        ac_color_map = _make_color_map(ac_rows, px.colors.qualitative.Plotly)
                        render_breakdown(ac_rows, ac_color_map, count_word="positioner")
                        render_drilldown(ac_rows, per_asset_class, ac_color_map)

                # === VALUTAER ===
                with tab_ccy:
                    if per_currency:
                        ccy_rows = _build_rows(per_currency, grand_total)
                        ccy_color_map = _make_color_map(ccy_rows, px.colors.qualitative.Bold)
                        render_breakdown(ccy_rows, ccy_color_map, count_word="aktiver")
                        render_drilldown(ccy_rows, per_currency, ccy_color_map, count_word="aktiver")

                # === REGIONER ===
                with tab_reg:
                    if per_region:
                        reg_rows = _build_rows(per_region, total_v)
                        reg_color_map = _make_color_map(reg_rows, px.colors.qualitative.Vivid)
                        render_breakdown(reg_rows, reg_color_map)
                        render_drilldown(reg_rows, per_region, reg_color_map)
                    else:
                        st.info("Ingen aktiepositioner at vise region-fordeling for.")

                # === LANDE ===
                with tab_country:
                    if per_country:
                        country_rows = _build_rows(per_country, total_v)
                        country_color_map = _make_color_map(country_rows, px.colors.qualitative.Pastel)
                        render_breakdown(country_rows, country_color_map)
                        render_drilldown(country_rows, per_country, country_color_map)
                    else:
                        st.info("Ingen aktiepositioner at vise lande-fordeling for.")

    # ===== TAB 2: PUNG =====
    with tab_wallet:
        total_dkk = cash_df[cash_df["Currency"] == "DKK"]["End cash balance"].sum()
        st.caption("Kontant i alt (rapportperiodens slut)")
        st.title(f"{_da_num(total_dkk)} kr.")

        st.subheader("Konti")
        c1, c2, c3 = st.columns(3)
        dkk_s = cash_df[cash_df["Currency"] == "DKK"]["End cash balance"].sum()
        usd_s = cash_df[cash_df["Currency"] == "USD"]["End cash balance"].sum()
        eur_s = cash_df[cash_df["Currency"] == "EUR"]["End cash balance"].sum()
        c1.metric("Danske Kroner", format_currency(dkk_s, "DKK"))
        c2.metric("US Dollar", format_currency(usd_s, "USD"))
        c3.metric("Euro", format_currency(eur_s, "EUR"))

        st.write("---")
        st.subheader("Indbetalinger")
        st.caption(
            "Justér tidspunktet hvis det afviger fra default kl. 09:00 — det forfiner "
            "TWR-beregningens vægtning af cashflows. Ændringer gemmes automatisk i "
            f"`{DEPOSIT_TIMES_FILE}`."
        )

        # Saml alle deposits/withdrawals fra DKK/USD/EUR transaction-sheets
        _all_deposits = []
        for _tx_df, _ccy in [(dkk_tx, "DKK"), (usd_tx, "USD"), (eur_tx, "EUR")]:
            if _tx_df.empty:
                continue
            _mask = _tx_df["Description"].str.contains(
                r"deposit|withdraw|indbetal|udbetal", case=False, na=False, regex=True
            )
            for _, _r in _tx_df[_mask].iterrows():
                _dnorm = pd.to_datetime(_r["Date"], dayfirst=True, errors="coerce")
                if pd.isna(_dnorm):
                    continue
                _all_deposits.append({
                    "date": _dnorm.normalize(),
                    "date_str": _dnorm.strftime("%Y-%m-%d"),
                    "amount": float(_r["Amount"]),
                    "currency": _ccy,
                    "desc": str(_r["Description"]),
                })
        _all_deposits.sort(key=lambda x: x["date"], reverse=True)

        if not _all_deposits:
            st.info("Ingen indbetalinger fundet i kontoudtoget.")
        else:
            _stored_times = load_deposit_times()
            _times_changed = False
            for _idx, _dep in enumerate(_all_deposits):
                _key = _deposit_key(_dep["date_str"], _dep["amount"], _dep["currency"])
                _saved = _stored_times.get(_key, DEFAULT_DEPOSIT_TIME)
                try:
                    _h, _m = (int(x) for x in _saved.split(":")[:2])
                    _default_time = time(_h, _m)
                except (ValueError, IndexError):
                    _default_time = time(9, 0)

                _d1, _d2, _d3, _d4 = st.columns([2, 2, 2, 1])
                _d1.markdown(
                    f"**{_dep['date'].strftime('%d. %b %Y')}**<br>"
                    f"<small style='color:#888;'>{_dep['desc']}</small>",
                    unsafe_allow_html=True,
                )
                _amount_color = "#388e3c" if _dep["amount"] >= 0 else "#d32f2f"
                _amount_prefix = "+" if _dep["amount"] >= 0 else ""
                _d2.markdown(
                    f"<p style='color:{_amount_color}; font-weight:bold; margin:6px 0;'>"
                    f"{_amount_prefix}{_da_num(_dep['amount'])} {_dep['currency']}</p>",
                    unsafe_allow_html=True,
                )
                _new_time = _d3.time_input(
                    "Tidspunkt",
                    value=_default_time,
                    key=f"deptime_{_key}",
                    label_visibility="collapsed",
                    step=60,  # 1-minut step
                )
                _new_str = f"{_new_time.hour:02d}:{_new_time.minute:02d}"
                if _new_str != _saved:
                    _stored_times[_key] = _new_str
                    _times_changed = True
                _d4.caption(_dep["currency"])
                st.divider()

            if _times_changed:
                if save_deposit_times(_stored_times):
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.warning(f"Kunne ikke gemme tidspunkter til {DEPOSIT_TIMES_FILE}.")

        st.write("---")
        st.subheader("Bevægelser")
        ccy = st.radio("Vis historik for:", ["DKK", "USD", "EUR"], horizontal=True)
        moves = orders_df[orders_df["Account currency"] == ccy].sort_values("Date", ascending=False)

        if not moves.empty:
            for _, row in moves.iterrows():
                m1, m2 = st.columns([4, 1])
                d_str = row["Date"].strftime("%d. %b %H:%M") if pd.notnull(row["Date"]) else ""
                m1.markdown(
                    f"**{row['Name']}**\n\n<small>{row['Side']} • {d_str}</small>",
                    unsafe_allow_html=True,
                )
                val = row["Notional (account currency)"]
                color = "#d32f2f" if row["Side"] == "BUY" else "#388e3c"
                prefix = "-" if row["Side"] == "BUY" else "+"
                m2.markdown(
                    f"<p style='text-align:right; color:{color}; font-weight:bold; margin-top:10px;'>"
                    f"{prefix}{_da_num(val)}</p>",
                    unsafe_allow_html=True,
                )
                st.divider()
        else:
            st.info("Ingen bevægelser i denne valuta.")

except Exception as e:
    st.error(f"Fejl ved behandling af data: {e}")
    st.exception(e)

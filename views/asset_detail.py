import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from data.cached import fetch_asset_history, fetch_period_reference_price
from datetime import timedelta

from config import PORTFOLIO_START, PLUTO_FX_SPREAD_RATE
from utils.formatting import (
    _safe_float, _da_num, format_currency, format_big_number,
    format_quantity, _flatten_html, _CCY_SYMBOLS
)
from utils.svg_charts import (
    _make_sparkline_data_url, _make_volume_bar_html,
    _make_range_bar_html, _make_performance_bars_html,
)
from data.cached import (
    fetch_live_quotes, fetch_live_fx_rates, fetch_ticker_quote_info,
    fetch_ticker_meta, fetch_performance,
)
from data.market_status import get_market_status_for_currency
from analytics.portfolio import slice_period, cumulative_return_series


def _fmt_time_2tz(unix_ts):
    """Unix-sekunder -> 'HH:MM EDT (HH:MM CEST)' — US-børstid + dansk tid."""
    if not unix_ts:
        return ""
    t = pd.Timestamp(int(unix_ts), unit="s", tz="UTC")
    et = t.tz_convert("America/New_York")
    cph = t.tz_convert("Europe/Copenhagen")
    return (
        f"{et.strftime('%H:%M')} {et.strftime('%Z')} "
        f"({cph.strftime('%H:%M')} {cph.strftime('%Z')})"
    )


def _add_extended_hours_markers(fig, price_x, y_lo, y_hi):
    """Markér extended hours på intraday-pris-grafen.

    Gråtoner extended-hours-perioderne let, tegner stiplede lodrette linjer
    ved regular-session-grænserne (09:30 og 16:00 ET) for hver handelsdag i
    data, og placerer én "Premarket"- og én "After-hours"-tekstboks på den
    seneste session. price_x er grafens x-værdier som ET tz-naive
    DatetimeIndex; y_lo/y_hi er grafens y-range.

    Gråtoning og linjer tegnes som scatter-traces (ikke add_vrect/add_vline,
    der er upålidelige på dato-akser i subplots)."""
    px = pd.DatetimeIndex(price_x)
    if len(px) == 0:
        return

    days = sorted(set(px.normalize()))
    pre_windows, post_windows = [], []
    line_x = []

    for day in days:
        open_t = day + pd.Timedelta(hours=9, minutes=30)
        close_t = day + pd.Timedelta(hours=16)
        end_t = day + pd.Timedelta(hours=20)

        pre_mask = (px >= day) & (px < open_t)
        reg_mask = (px >= open_t) & (px < close_t)
        post_mask = (px >= close_t) & (px < end_t)

        has_pre = bool(pre_mask.any())
        has_regular = bool(reg_mask.any())
        has_post = bool(post_mask.any())

        # Linje kun ved en grænse med data på begge sider
        if has_pre and has_regular:
            line_x.append(open_t)
        if has_regular and has_post:
            line_x.append(close_t)

        # Felterne klippes til faktiske data-punkter, så de aldrig spænder
        # ind i det rangebreak-skjulte interval (20:00-04:00 ET) — ellers
        # renderer Plotly dem forkert for de midterste dage.
        if has_pre:
            pre_windows.append((px[pre_mask].min(), open_t))
        if has_post:
            post_windows.append((close_t, px[post_mask].max()))

    # Let gråtoning af extended-hours-perioderne. Tegnes som go.Bar — samme
    # trace-type som volumen-panelet, der renderer pålideligt på denne
    # dato-akse med rangebreaks. (fill="toself" svigtede for de midterste
    # felter.) Lav opacitet, så pris-linjen kun påvirkes svagt.
    windows = pre_windows + post_windows
    if windows:
        centers = [w[0] + (w[1] - w[0]) / 2 for w in windows]
        widths = [(w[1] - w[0]).total_seconds() * 1000 for w in windows]
        fig.add_trace(
            go.Bar(
                x=centers, y=[y_hi - y_lo] * len(windows),
                base=y_lo, width=widths,
                marker=dict(color="rgba(120,120,135,0.07)", line=dict(width=0)),
                hoverinfo="skip", showlegend=False,
            ),
            row=1, col=1,
        )

    # Alle lodrette linjer i én trace, adskilt af None
    if line_x:
        xs, ys = [], []
        for t in line_x:
            xs += [t, t, None]
            ys += [y_lo, y_hi, None]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color="rgba(0,0,0,0.35)", width=1, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ),
            row=1, col=1,
        )

    def _label(window, text):
        start, end = window
        mid = start + (end - start) / 2
        fig.add_annotation(
            x=mid, y=y_hi, text=text, showarrow=False,
            row=1, col=1, yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="rgba(0,0,0,0.15)", borderwidth=1, borderpad=3,
            font=dict(size=11, color="#666"),
        )

    # Kun én label af hver type — placeret på den seneste session
    if pre_windows:
        _label(pre_windows[-1], "Premarket")
    if post_windows:
        _label(post_windows[-1], "After-hours")


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
    # Brug Plutos egne GAK fra positions-arket hvis tilgængeligt (samme kilde som aktieoversigten)
    gak_valuta = 0.0
    if (positions_df is not None
            and "Ticker" in positions_df.columns
            and "Average entry price (asset currency)" in positions_df.columns):
        pos_row = positions_df[positions_df["Ticker"] == ticker]
        if not pos_row.empty:
            gak_valuta = _safe_float(pos_row.iloc[0]["Average entry price (asset currency)"]) or 0.0
    if not gak_valuta and qty_total > 0 and rate_now > 0:
        gak_valuta = (cost_dkk / qty_total) / rate_now
    delta_dkk = (live - prev_close) if (live is not None and prev_close is not None) else 0
    delta_pct = (delta_dkk / prev_close * 100) if prev_close else 0

    # =====================================================================
    # SEKTION 1: HEADER
    # =====================================================================

    # --- Aktiv-navigator: rullemenu med bredde tilpasset det længste navn ---
    _ticker_name_map = (
        orders_df[orders_df["Ticker"].isin(all_active_tickers)]
        [["Ticker", "Name"]]
        .drop_duplicates("Ticker")
        .assign(_label=lambda d: d["Name"] + "  (" + d["Ticker"] + ")")
        .sort_values("_label")
        .reset_index(drop=True)
    )
    _labels = _ticker_name_map["_label"].tolist()
    _tickers = _ticker_name_map["Ticker"].tolist()
    _cur_idx = _tickers.index(ticker) if ticker in _tickers else 0

    # Beregn kolonnebredde ud fra det længste label (ca. 8px pr. tegn + 80px til pil/padding).
    # Streamlit wide-mode er ca. 1300px bred — oversæt til kolonnevægt ud af 10.
    _max_chars = max((len(l) for l in _labels), default=30)
    _target_px = max(220, _max_chars * 8 + 80)
    _col_weight = round(_target_px / 1300 * 10, 1)
    _spacer_weight = max(0.5, 10 - _col_weight)

    _nav_col, _ = st.columns([_col_weight, _spacer_weight])
    with _nav_col:
        _selected_label = st.selectbox(
            "Vælg aktiv",
            options=_labels,
            index=_cur_idx,
            label_visibility="collapsed",
        )

    _selected_ticker = _tickers[_labels.index(_selected_label)]
    if _selected_ticker != ticker:
        st.query_params["ticker"] = _selected_ticker
        st.rerun()

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
    def _price_box_html(price, ref, context_label, time_str=""):
        """Byg én pris-blok (label + stor pris + Δ vs ref + tidsstempel)."""
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
        time_part = (
            f"<div style='font-size:12px; color:#999; margin-top:6px;'>{time_str}</div>"
            if time_str else ""
        )
        return (
            f"<div>"
            f"  <div style='font-size:13px; color:#888; margin-bottom:4px;'>{context_label}</div>"
            f"  <div style='font-size:40px; font-weight:700; line-height:1;'>"
            f"    {ccy_sym} {_da_num(price)}"
            f"  </div>"
            f"  {delta_part}"
            f"  {time_part}"
            f"</div>"
        )

    # Tidsstempler (US-børstid + dansk tid) til pris-blokkene
    _reg_str = _fmt_time_2tz(info.get("regular_market_time"))
    _ext_candidates = [t for t in (info.get("post_market_time"),
                                   info.get("pre_market_time")) if t]
    _ext_str = _fmt_time_2tz(max(_ext_candidates)) if _ext_candidates else ""

    if sess_code == "regular":
        # Regular session: én pris (live), Δ vs forrige lukkepris
        main_box = _price_box_html(live, prev_close, "Aktuel pris (markedet åbent)", _reg_str)
        ext_box = ""
    elif today_close_val is not None:
        # Efter dagens regular close (samme dag): venstre = dagens close,
        # højre = nuværende extended-hours live
        main_box = _price_box_html(today_close_val, prev_close, "Dagens lukkepris (16:00 ET)", _reg_str)
        if live is not None and abs(live - today_close_val) > 0.001:
            ext_box = _price_box_html(live, today_close_val, sess_label, _ext_str)
        else:
            ext_box = ""
    else:
        # Pre-market / overnight (efter midnat) / weekend:
        # venstre = forrige lukkepris, højre = nuværende live (hvis afviger)
        main_box = _price_box_html(prev_close, prev_prev, "Seneste lukkepris", _reg_str)
        if (live is not None and prev_close is not None
                and abs(live - prev_close) > 0.001):
            ext_box = _price_box_html(live, prev_close, sess_label, _ext_str)
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
        det_prices, det_volumes = fetch_asset_history(
            ticker, det_period, include_extended
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
            # Find den rigtige forrige-dags-close ved at slaa op i prices-DataFrame.
            # Det er mere robust end prev_close/prev_prev fra live-quote, som opfoerer
            # sig forskelligt i weekender, extended hours og under regular session.
            # Strategi: find hvilken dag 1D-grafen viser (foerste timestamp i det_prices),
            # og returner det seneste close-punkt FOER den dag fra prices[ticker].
            _ref1d = None
            if not det_prices.empty and ticker in prices.columns:
                _first = pd.Timestamp(det_prices.index[0])
                if _first.tz is not None:
                    _first = _first.tz_convert("America/New_York").tz_localize(None)
                _chart_day = _first.normalize()
                _hist = prices[ticker].dropna()
                _hist_norm = _hist.index.normalize() if hasattr(_hist.index, "normalize") else _hist.index
                _before = _hist[_hist_norm < _chart_day]
                if not _before.empty:
                    _ref1d = float(_before.iloc[-1])
            ref_price = _ref1d if (_ref1d and _ref1d > 0) else prev_close
        elif det_period == "Maks":
            ref_price = None  # Brug første pris i grafen
        else:
            ref_price = fetch_period_reference_price(ticker, det_period)

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
            # Brug samme reference som ref_price (allerede beregnet ovenfor)
            baseline = period_start_price if period_start_price > 0 else (prev_close or 0)
            baseline_label = f"Forrige luk {ccy_sym} {_da_num(baseline)}"
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

        # Tooltip: ET-tidszone + dansk tid pr. punkt (kun intraday)
        tz_customdata = None
        if is_intraday and price_idx_orig.tz is not None:
            _et = price_idx_orig.tz_convert("America/New_York")
            _cph = price_idx_orig.tz_convert("Europe/Copenhagen")
            tz_customdata = [
                f"{e.strftime('%Z')} ({c.strftime('%H:%M')} {c.strftime('%Z')})"
                for e, c in zip(_et, _cph)
            ]
            hover_fmt = (
                f"<b>%{{x|%d. %b %Y, %H:%M}} %{{customdata}}</b>"
                f"<br>Pris: {ccy_sym} %{{y:,.2f}}<extra></extra>"
            )

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

        # Yahoo Finance-stil farvning:
        # - 1D: split grøn/rød ved baseline (prev_close). Fill går fra
        #        kurslinje ned til baseline — ikke til y=0. Linjen er ogsaa
        #        farvet segment for segment.
        # - Øvrige perioder: enkelt farve baseret paa samlet perioderetning,
        #        fill til bunden (tozeroy, men y-akse begrænser til data-range).
        line_color = "#2e7d32" if period_is_up else "#d32f2f"
        fill_rgba = "rgba(46,125,50,0.15)" if period_is_up else "rgba(211,47,47,0.15)"

        if det_period == "1D" and show_baseline_line:
            # Split-farvet 1D-graf (Yahoo Finance-stil)
            segments = _segment_by_sign(list(price_x), list(ys), baseline=y_baseline)

            # Hover-format: ET-tidszone (tz_customdata ikke tilgaengeligt per segment)
            _hover_seg = (
                f"<b>%{{x|%d. %b %Y %H:%M}} ET</b>"
                f"<br>Pris: {ccy_sym} %{{y:,.2f}}<extra></extra>"
            )

            # Trin 1: Fill-polygoner (tegnes bag linjen).
            # Hvert segment lukkes som polygon: pris frem + baseline tilbage.
            for seg_sign, seg_pts in segments:
                if not seg_pts or seg_sign is None:
                    continue
                sx = [p[0] for p in seg_pts]
                sy = [p[1] for p in seg_pts]
                f_rgba = ("rgba(46,125,50,0.15)" if seg_sign == "pos"
                          else "rgba(211,47,47,0.15)")
                poly_x = sx + sx[::-1] + [sx[0]]
                poly_y = sy + [y_baseline] * len(sx) + [sy[0]]
                fig_det.add_trace(
                    go.Scatter(
                        x=poly_x, y=poly_y,
                        mode="lines",
                        line=dict(width=0, color="rgba(0,0,0,0)"),
                        fill="toself", fillcolor=f_rgba,
                        hoverinfo="skip", showlegend=False,
                    ),
                    row=1, col=1,
                )

            # Trin 2: Farvede linje-segmenter ovenpå fill.
            for seg_sign, seg_pts in segments:
                if not seg_pts:
                    continue
                sx = [p[0] for p in seg_pts]
                sy = [p[1] for p in seg_pts]
                s_color = ("#2e7d32" if seg_sign == "pos"
                           else ("#d32f2f" if seg_sign == "neg" else "#888888"))
                fig_det.add_trace(
                    go.Scatter(
                        x=sx, y=sy,
                        mode="lines",
                        line=dict(color=s_color, width=2),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=1, col=1,
                )

            # Enkelt usynlig hover-trace der daekker al data — giver noejagtigt
            # ét tooltip-punkt pr. x-position (undgaar dubletter fra segmenter).
            fig_det.add_trace(
                go.Scatter(
                    x=list(price_x), y=list(ys),
                    mode="lines",
                    line=dict(width=0, color="rgba(0,0,0,0)"),
                    hovertemplate=_hover_seg,
                    showlegend=False,
                ),
                row=1, col=1,
            )
        else:
            # Enkelt-farvet linje for alle ikke-1D perioder
            fig_det.add_trace(
                go.Scatter(
                    x=price_x, y=ys,
                    mode="lines",
                    line=dict(color=line_color, width=2),
                    fill="tozeroy",
                    fillcolor=fill_rgba,
                    customdata=tz_customdata,
                    hovertemplate=hover_fmt,
                    showlegend=False,
                ),
                row=1, col=1,
            )

        # Stiplet baseline med label (kun for 1D og Maks)
        if show_baseline_line:
            fig_det.add_hline(
                y=y_baseline, line_dash="dash", line_color="rgba(0,0,0,0.4)",
                annotation_text=baseline_label, annotation_position="top right",
                annotation_font_size=11, row=1, col=1,
                annotation_bgcolor="rgba(255,255,255,0.88)",
                annotation_bordercolor="rgba(0,0,0,0.18)",
                annotation_borderwidth=1,
                annotation_borderpad=4,
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

            # Ingen auto-range-padding — grafen starter præcis ved data
            _x0, _x1 = price_x.min(), price_x.max()
            fig_det.update_xaxes(range=[_x0, _x1], row=1, col=1)
            fig_det.update_xaxes(range=[_x0, _x1], row=2, col=1)

        if is_intraday and include_extended:
            _add_extended_hours_markers(
                fig_det, price_x, y_min_data - y_pad, y_max_data + y_pad
            )

        st.plotly_chart(fig_det, use_container_width=True)

    st.write("---")

    # =====================================================================
    # SEKTION 4: MARKEDSDATA (range-bars)
    # =====================================================================
    st.subheader("Markedsdata")

    col_bars, col_perf = st.columns(2)
    with col_bars:
        st.markdown(
            _flatten_html(_make_volume_bar_html(info["volume"], info["average_volume"])),
            unsafe_allow_html=True,
        )
        st.markdown(
            _flatten_html(_make_range_bar_html(
                info["day_low"], info["day_high"],
                marker_low=today_close_val, marker_high=info["open"],
                marker_low_label="Luk", marker_high_label="Åbn",
                bottom_label="DAY LOW/HIGH",
                currency_symbol=ccy_sym, segment_fill=False,
                track_color="#d6e4f0",
            )),
            unsafe_allow_html=True,
        )
        st.markdown(
            _flatten_html(_make_range_bar_html(
                info["fifty_two_week_low"], info["fifty_two_week_high"],
                marker_low=info["day_low"], marker_high=info["day_high"],
                bottom_label="52 WEEK LOW/HIGH", segment_label="Dagens spænd",
                currency_symbol=ccy_sym, segment_fill=True,
            )),
            unsafe_allow_html=True,
        )

    with col_perf:
        st.markdown(
            "<div style='font-size:13px; font-weight:700; color:#888; "
            "letter-spacing:0.5px; padding:10px 0 4px;'>PERFORMANCE</div>",
            unsafe_allow_html=True,
        )
        _perf = fetch_performance(ticker)
        _perf_rows = [
            ("5 dage", _perf["5d"]),
            ("1 måned", _perf["1m"]),
            ("3 måneder", _perf["3m"]),
            ("YTD", _perf["ytd"]),
            ("1 år", _perf["1y"]),
        ]
        st.markdown(
            _flatten_html(_make_performance_bars_html(_perf_rows)),
            unsafe_allow_html=True,
        )

    st.write("")
    with st.expander("Nøgletal og selskabsinfo"):
        _info_rows = [
            ("Market cap", format_big_number(info["market_cap"]) if info["market_cap"] else "—"),
            ("Beta (5Y Monthly)", _da_num(info["beta"]) if info["beta"] else "—"),
            ("1 Year Target Estimate",
             f"{ccy_sym} {_da_num(info['target_mean_price'])}" if info["target_mean_price"] else "—"),
            ("EPS (TTM)", f"{ccy_sym} {_da_num(info['trailing_eps'])}" if info["trailing_eps"] else "—"),
            ("P/E (TTM)", _da_num(info["trailing_pe"]) if info["trailing_pe"] else "—"),
        ]
        _info_html = "".join(
            f"<tr>"
            f"<td style='padding:6px 24px 6px 0; color:#888;'>{label}</td>"
            f"<td style='padding:6px 0; font-weight:600;'>{value}</td>"
            f"</tr>"
            for label, value in _info_rows
        )
        _website = info.get("website")
        if _website:
            _info_html += (
                f"<tr><td style='padding:6px 24px 6px 0; color:#888;'>Hjemmeside</td>"
                f"<td style='padding:6px 0; font-weight:600;'>"
                f"<a href='{_website}' target='_blank' rel='noopener'>{_website}</a>"
                f"</td></tr>"
            )
        st.markdown(
            _flatten_html(
                f"<table style='font-size:14px; border-collapse:collapse;'>"
                f"<tbody>{_info_html}</tbody></table>"
            ),
            unsafe_allow_html=True,
        )
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
    _ccy_sym_map = {"USD": "$", "EUR": "€", "DKK": "kr."}
    for _, r in sub.sort_values("Date", ascending=False).iterrows():
        d_str = r["Date"].strftime("%d. %b %Y %H:%M") if pd.notnull(r["Date"]) else "—"
        ac = r.get("Account currency", "")
        qty = r["Quantity"] if r["Quantity"] else 0
        fx_rate      = _safe_float(r.get("FX rate"))
        notional_dkk = _safe_float(r.get("Notional, DKK")) or 0
        notional_ac  = _safe_float(r.get("Notional (account currency)")) or 0

        # Pris i aktivets valuta
        if ac == asset_ccy or not fx_rate:
            px_local = notional_ac / qty if qty else 0
        else:
            notional_asset = notional_dkk / fx_rate
            px_local = notional_asset / qty if qty else 0

        # Fra konto
        ac_sym = _ccy_sym_map.get(ac, ac)
        fra_konto_str = (
            f"{_da_num(notional_ac)} {ac_sym}" if ac == "DKK"
            else f"{ac_sym} {_da_num(notional_ac)}"
        )

        # Vekselgebyr
        veksel_str = (
            f"{_da_num(notional_dkk * PLUTO_FX_SPREAD_RATE)} kr."
            if ac != asset_ccy else "—"
        )

        # Kurtage i DKK
        comm    = _safe_float(r.get("Commission (account currency)")) or 0
        rate_t  = usd_dkk_now if ac == "USD" else (eur_dkk_now if ac == "EUR" else 1.0)
        comm_dkk = comm * rate_t

        trades_rows.append({
            "Dato":                d_str,
            "Side":                r["Side"],
            "Antal":               format_quantity(qty),
            f"Pris ({asset_ccy})": f"{ccy_sym} {_da_num(px_local)}",
            "Fra konto":           fra_konto_str,
            "Beløb (DKK)":         f"{_da_num(notional_dkk)} kr.",
            "Vekselgebyr (DKK)":   veksel_str,
            "Kurtage (DKK)":       f"{_da_num(comm_dkk)} kr." if comm_dkk else "—",
        })
    if trades_rows:
        trades_df_disp = pd.DataFrame(trades_rows)
        st.dataframe(trades_df_disp, use_container_width=True, hide_index=True)
        st.caption(
            f"*Pris ({asset_ccy})* er handelsprisen i aktivets valuta. "
            "Ved køb fra DKK-konto bruges den historiske FX-kurs fra XLSX-filen. "
            f"*Vekselgebyr* = Beløb (DKK) × {PLUTO_FX_SPREAD_RATE*100:.2f}% (Plutos prismodel)."
        )

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
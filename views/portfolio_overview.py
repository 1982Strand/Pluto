"""
views/portfolio_overview.py — Portefølje-tab
Viser TWR-afkast, porteføljegraf, aktietabel og fordelinger.
Ingen direkte datahentning her — alle data modtages som parametre.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import PLUTO_FX_SPREAD_RATE
from utils.formatting import (
    _safe_float, _da_num, format_currency, format_quantity,
    _flatten_html, color_change_str,
)
from data.fetch import (
    fetch_live_quotes, fetch_live_fx_rates,
    fetch_ticker_meta, fetch_intraday_sparklines,
)
from data.market_status import get_us_market_status, get_market_status_for_currency
from analytics.portfolio import (
    cumulative_return_series, slice_period,
    compute_portfolio_value_series_intraday, cashflow_timeline,
)
from utils.svg_charts import _make_sparkline_data_url
from views.breakdown import render_breakdown, render_drilldown, _lighten_rgb


def render_portfolio_overview(
    orders_df: pd.DataFrame,
    dkk_tx: pd.DataFrame,
    usd_tx: pd.DataFrame,
    eur_tx: pd.DataFrame,
    positions_df: pd.DataFrame,
    cash_df: pd.DataFrame,
    prices: pd.DataFrame,
    total_value: pd.Series,
    stock_value: pd.Series,
    cash_value_total: pd.Series,
    cashflows: pd.Series,
    cashflow_fracs,
    usd_dkk: pd.Series,
    eur_dkk: pd.Series,
) -> None:
    """Renderer Portefølje-tabben."""

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
        _segments = []
        if len(_ex_list) > 0:
            _cur_sign = None
            _cur_seg = []
            for _x, _y in zip(_ex_list, _ey_list):
                if _y > 0:
                    _new_sign = "pos"
                elif _y < 0:
                    _new_sign = "neg"
                else:
                    _new_sign = None  # zero — ambivalent, tilhører begge
                if _new_sign is None:
                    if _cur_seg:
                        _cur_seg.append((_x, _y))
                        _segments.append((_cur_sign, _cur_seg))
                    _cur_seg = [(_x, _y)]
                    _cur_sign = None
                elif _cur_sign is None or _cur_sign == _new_sign:
                    _cur_seg.append((_x, _y))
                    _cur_sign = _new_sign
                else:
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

            # Sparkline: regular-hours intraday-graf for seneste handelsdag
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
        # ~35 px pr. række + ~38 px til header
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
        # FX-spread 0,15% af "Notional, DKK" på USD/EUR-handler.
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
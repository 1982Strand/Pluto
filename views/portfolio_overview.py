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

from utils.formatting import (
    _da_num, format_currency, format_quantity,
    _flatten_html, color_change_str,
)
from data.cached import (
    fetch_live_quotes, fetch_live_fx_rates,
    fetch_ticker_meta, fetch_intraday_sparklines,
)
from data.market_status import get_us_market_status, get_market_status_for_currency
from analytics.portfolio import (
    cumulative_return_series, slice_period,
    compute_portfolio_value_series_intraday, cashflow_timeline,
    compute_portfolio_return_dynamics,
    compute_grouped_portfolio_return_dynamics,
    split_signed_return_segments,
)
from analytics.holdings import (
    compute_portfolio_costs, compute_pnl_summary, compute_holdings_breakdown,
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
    cash_live_dkk: float,
    cashflows: pd.Series,
    cashflow_fracs,
    usd_dkk: pd.Series,
    eur_dkk: pd.Series,
) -> None:
    """Renderer Portefølje-tabben."""

    # Defineres her så dynamik-grafen nederst altid kan genbruge den —
    # aktietabellen fylder den ud, hvis der er aktive positioner.
    ticker_meta = {}

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
        # Grøn for positive segmenter, rød for negative — én trace pr.
        # kontinuerligt fortegns-segment, så fill="tozeroy" ikke bløder.
        st.write("")
        # Intraday-data fra yfinance er tz-bevidst; den daglige fallback-serie
        # er tz-naiv. Plotly renderer tz-bevidste tidsstempler i UTC på aksen,
        # mens tooltippet viste dansk tid — derfor 2 timers mismatch. Vi
        # konverterer intraday-x til dansk tid (tz-naiv), så akse og tooltip
        # viser samme klokkeslæt.
        _d_idx = pd.DatetimeIndex(d_per)
        _is_intraday = _d_idx.tz is not None
        if _is_intraday:
            d_per_plot = _d_idx.tz_convert("Europe/Copenhagen").tz_localize(None)
        else:
            d_per_plot = _d_idx

        _segments = split_signed_return_segments(
            d_per_plot.to_numpy(),
            ret_pct_series.values,
            v_per.values,
        )

        fig = go.Figure()

        def _fmt_x(x):
            # x er dansk tid: tz-naiv for intraday, en ren dato for daglige
            # perioder. For intraday lokaliseres der for at få CEST/CET-mærket.
            ts = pd.Timestamp(x)
            if not _is_intraday:
                return ts.strftime("%d. %b %Y")
            if ts.tz is None:
                ts = ts.tz_localize("Europe/Copenhagen",
                                    nonexistent="shift_forward", ambiguous=True)
            return ts.strftime("%d. %b %Y %H:%M %Z")

        for _sign, _seg in _segments:
            if len(_seg) < 2:
                continue
            _xs = [p[0] for p in _seg]
            _ys = [p[1] for p in _seg]
            _vs = [p[2] for p in _seg]
            _cd = [[_fmt_x(x), f"{_da_num(v)} DKK"] for x, v in zip(_xs, _vs)]
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
                customdata=_cd,
                hovertemplate="<b>%{customdata[0]}</b><br>Afkast: %{y:.2f}%<br>Værdi: %{customdata[1]}<extra></extra>",
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
                x=d_per_plot, y=v_per.values,
                mode="lines", line=dict(color="#1976d2", width=2),
                name="Porteføljeværdi",
                hovertemplate="<b>%{x|%d. %b %Y}</b><br>Værdi: %{y:,.2f} DKK<extra></extra>",
            ))
            cum_cf_series = (cf_per.cumsum() - cf_per.iloc[0])
            fig2.add_trace(go.Scatter(
                x=d_per_plot,
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
        st.metric("Kontant (DKK)", _da_num(cash_live_dkk))
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

    orders_df = orders_df.copy()
    orders_df["Qty_Adj"] = np.where(orders_df["Side"] == "BUY", orders_df["Quantity"], -orders_df["Quantity"])
    orders_df["DKK_Adj"] = np.where(orders_df["Side"] == "BUY", orders_df["Notional, DKK"], -orders_df["Notional, DKK"])

    portfolio = (
        orders_df.groupby(["Ticker", "Name", "Asset currency"])
        .agg(Qty_Adj=("Qty_Adj", "sum"), DKK_Adj=("DKK_Adj", "sum"))
        .reset_index()
    )
    active = portfolio[portfolio["Qty_Adj"] > 0.001].copy()

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

        breakdown = compute_holdings_breakdown(
            orders_df, positions_df, prices, live_quotes, ticker_meta, cash_df,
            usd_dkk_now, eur_dkk_now, usd_dkk, eur_dkk, afkast_periode, ytd_start,
        )
        positions = breakdown["positions"]
        total_v = breakdown["total_value_dkk"]
        grand_total = breakdown["grand_total_dkk"]
        all_holdings = breakdown["all_holdings"]
        per_sector = breakdown["buckets"]["sector"]
        per_asset_class = breakdown["buckets"]["asset_class"]
        per_currency = breakdown["buckets"]["currency"]
        per_region = breakdown["buckets"]["region"]
        per_country = breakdown["buckets"]["country"]

        afkast_kr_label = "Afkast (kr.)" if afkast_periode == "Maks" else f"Afkast {afkast_periode} (kr.)"
        afkast_pct_label = "Pos. afkast" if afkast_periode == "Maks" else f"Pos. afkast {afkast_periode}"

        rows = []
        for pos in positions:
            t = pos["ticker"]
            ccy = pos["ccy"]

            if pos["d_yest"] is not None:
                d_yest_str = f"{_da_num(pos['d_yest'], signed=True)} ({_da_num(pos['d_yest_pct'], signed=True)}%)"
            else:
                d_yest_str = "—"

            if pos["d_now"] is not None:
                d_now_str = f"{_da_num(pos['d_now'], signed=True)} ({_da_num(pos['d_now_pct'], signed=True)}%)"
            else:
                d_now_str = "—"

            # Sparkline: regular-hours intraday-graf for seneste handelsdag
            _spark_url = _make_sparkline_data_url(
                sparklines_map.get(t, []), ref_value=pos["forrige_luk"]
            )

            rows.append({
                "Navn": pos["name"][:32],
                "Ticker": t,
                "🔍": f"?ticker={t}",
                "1D": _spark_url,
                "Antal": format_quantity(pos["qty"]),
                "GAK": format_currency(pos["gak_valuta"], ccy),
                "Sidste luk": format_currency(pos["sidste_luk"], ccy),
                "Δ sidste luk": d_yest_str,
                "Aktuel": format_currency(pos["live"], ccy),
                "Δ aktuel": f"{d_now_str} {session_cell_for(ccy)}",
                "Værdi (DKK)": format_currency(pos["vaerdi_dkk"], "DKK"),
                afkast_kr_label: f"{_da_num(pos['pnl_pos_dkk'], signed=True)} kr.",
                afkast_pct_label: f"{_da_num(pos['pnl_pos_pct'], signed=True)}%",
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

        # ----- Omkostninger (kurtage + FX-spread) og realiseret afkast -----
        costs = compute_portfolio_costs(orders_df, usd_dkk_now, eur_dkk_now)
        total_commission_dkk = costs["commission_dkk"]
        total_fx_spread_dkk = costs["fx_spread_dkk"]
        total_costs_dkk = costs["total_dkk"]

        _cash_real_now = cash_live_dkk
        pnl = compute_pnl_summary(
            total_v, total_i, _cash_real_now, total_deposits, total_costs_dkk
        )
        unrealized_pnl = pnl["unrealized_pnl"]
        unrealized_pct = pnl["unrealized_pct"]
        realized_pnl = pnl["realized_pnl"]
        realized_pct = pnl["realized_pct"]
        reelt_investeret = pnl["reelt_investeret"]

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

                    sec_col_chart, sec_col_list = st.columns([1, 2])
                    with sec_col_chart:
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

                    with sec_col_list:
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

    # ----- Dynamik i porteføljeafkast -----
    st.write("")
    st.write("---")
    st.subheader("Dynamik i porteføljeafkast")

    dyn_col_period, dyn_col_mode, dyn_col_group = st.columns([1, 1, 1])

    with dyn_col_period:
        dyn_grouping = st.selectbox(
            "Periode",
            ["Daglig", "Ugentlig", "Månedlig", "Årlig"],
            index=2,
            key="portfolio_return_dynamics_grouping",
        )

    with dyn_col_mode:
        dyn_mode = st.selectbox(
            "Visning",
            ["Procent", "Værdi"],
            index=0,
            key="portfolio_return_dynamics_mode",
        )

    with dyn_col_group:
        dyn_breakdown = st.selectbox(
            "Gruppering",
            ["Ingen gruppering", "Aktivklasser", "Sektorer"],
            index=0,
            key="portfolio_return_dynamics_breakdown",
        )

    max_bars_map = {
        "Daglig": 90,
        "Ugentlig": 104,
        "Månedlig": 60,
        "Årlig": 20,
    }
    max_bars = max_bars_map.get(dyn_grouping, 60)

    if dyn_breakdown == "Ingen gruppering":
        dynamics_df = compute_portfolio_return_dynamics(
            total_value=total_value,
            cashflows=cashflows,
            cashflow_fracs=cashflow_fracs,
            grouping=dyn_grouping,
        )

        if dynamics_df.empty:
            st.info("Der er ikke nok data til at vise dynamik i porteføljeafkast.")
        else:
            plot_df = dynamics_df.tail(max_bars).copy()

            if dyn_mode == "Procent":
                y_col = "return_pct"
                y_title = "Afkast (%)"
            else:
                y_col = "return_dkk"
                y_title = "Afkast (DKK)"

            bar_colors = np.where(
                plot_df[y_col] >= 0,
                "#4fd1c5",
                "#f45b73",
            )

            fig_dyn = go.Figure()

            fig_dyn.add_trace(go.Bar(
                x=plot_df["period_label"],
                y=plot_df[y_col],
                marker=dict(
                    color=bar_colors,
                    line=dict(width=0),
                ),
                name="Portefølje",
                customdata=np.column_stack([
                    plot_df["period_start"].dt.strftime("%d-%m-%Y"),
                    plot_df["period_end"].dt.strftime("%d-%m-%Y"),
                    plot_df["return_dkk"],
                    plot_df["return_pct"],
                ]),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Periode: %{customdata[0]} – %{customdata[1]}<br>"
                    "Afkast DKK: %{customdata[2]:,.2f}<br>"
                    "Afkast %: %{customdata[3]:.2f}%"
                    "<extra></extra>"
                ),
            ))

            fig_dyn.add_hline(
                y=0,
                line_dash="dash",
                line_color="rgba(120,120,120,0.55)",
                line_width=1,
            )

            fig_dyn.update_layout(
                height=430,
                margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(
                    title="",
                    tickangle=-45 if len(plot_df) > 12 else 0,
                ),
                yaxis=dict(
                    title=y_title,
                    zeroline=False,
                ),
                hovermode="closest",
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.22,
                    xanchor="center",
                    x=0.5,
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                separators=".,",
            )

            st.plotly_chart(fig_dyn, use_container_width=True)

            st.caption(
                f"Grafen viser periodisk porteføljeafkast grupperet efter {dyn_grouping.lower()} "
                f"og vist som {dyn_mode.lower()}. Afkastet er korrigeret for ind- og udbetalinger "
                "efter samme princip som porteføljens øvrige TWR/Modified Dietz-beregninger."
            )

    else:
        if not active.empty:
            dyn_active_tickers = active["Ticker"].dropna().unique().tolist()
        else:
            dyn_active_tickers = []

        if not dyn_active_tickers:
            st.info("Der er ingen aktive positioner at gruppere.")
        else:
            if ticker_meta:
                dyn_ticker_meta = ticker_meta
            else:
                with st.spinner("Henter sektor- og aktivklasse-info..."):
                    dyn_ticker_meta = fetch_ticker_meta(tuple(dyn_active_tickers))

            if dyn_breakdown == "Aktivklasser":
                dyn_group_map = {
                    t: dyn_ticker_meta.get(t, {}).get("asset_class", "Andet")
                    for t in dyn_active_tickers
                }
                residual_group_name = "Kontanter"
                y_title_pct = "%-point bidrag til porteføljeafkastet"
            else:
                dyn_group_map = {
                    t: dyn_ticker_meta.get(t, {}).get("sector", "Andet")
                    for t in dyn_active_tickers
                }
                residual_group_name = "Kontanter/øvrigt"
                y_title_pct = "%-point bidrag til porteføljeafkastet"

            grouped_df = compute_grouped_portfolio_return_dynamics(
                orders=orders_df,
                prices=prices,
                usd_dkk=usd_dkk,
                eur_dkk=eur_dkk,
                total_value=total_value,
                cashflows=cashflows,
                cashflow_fracs=cashflow_fracs,
                group_map=dyn_group_map,
                grouping=dyn_grouping,
                residual_group_name=residual_group_name,
            )

            if grouped_df.empty:
                st.info("Der er ikke nok data til at vise grupperet dynamik i porteføljeafkast.")
            else:
                period_order = (
                    grouped_df[["period_label", "period_end"]]
                    .drop_duplicates()
                    .sort_values("period_end")
                    .tail(max_bars)
                )

                keep_labels = period_order["period_label"].tolist()
                plot_df = grouped_df[grouped_df["period_label"].isin(keep_labels)].copy()

                plot_df["period_label"] = pd.Categorical(
                    plot_df["period_label"],
                    categories=keep_labels,
                    ordered=True,
                )

                if dyn_mode == "Procent":
                    y_col = "return_pct"
                    y_title = y_title_pct
                else:
                    y_col = "return_dkk"
                    y_title = "Afkastbidrag (DKK)"

                group_order = (
                    plot_df.groupby("group")["return_dkk"]
                    .apply(lambda s: abs(float(s.sum())))
                    .sort_values(ascending=False)
                    .index
                    .tolist()
                )

                palette = (
                    px.colors.qualitative.Plotly
                    + px.colors.qualitative.Set2
                    + px.colors.qualitative.Bold
                    + px.colors.qualitative.Pastel
                )

                fig_dyn = go.Figure()

                for i, group_name in enumerate(group_order):
                    sub_g = (
                        plot_df[plot_df["group"] == group_name]
                        .sort_values("period_label")
                        .copy()
                    )

                    color = palette[i % len(palette)]

                    fig_dyn.add_trace(go.Bar(
                        x=sub_g["period_label"],
                        y=sub_g[y_col],
                        name=group_name,
                        marker=dict(
                            color=color,
                            line=dict(width=0),
                        ),
                        customdata=np.column_stack([
                            sub_g["period_start"].dt.strftime("%d-%m-%Y"),
                            sub_g["period_end"].dt.strftime("%d-%m-%Y"),
                            sub_g["return_dkk"],
                            sub_g["return_pct"],
                            sub_g["group"],
                        ]),
                        hovertemplate=(
                            "<b>%{customdata[4]}</b><br>"
                            "Periode: %{customdata[0]} – %{customdata[1]}<br>"
                            "Bidrag DKK: %{customdata[2]:,.2f}<br>"
                            "Bidrag %-point: %{customdata[3]:.2f}"
                            "<extra></extra>"
                        ),
                    ))

                fig_dyn.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color="rgba(120,120,120,0.55)",
                    line_width=1,
                )

                fig_dyn.update_layout(
                    height=470,
                    margin=dict(l=0, r=0, t=10, b=0),
                    barmode="relative",
                    xaxis=dict(
                        title="",
                        tickangle=-45 if len(keep_labels) > 12 else 0,
                    ),
                    yaxis=dict(
                        title=y_title,
                        zeroline=False,
                    ),
                    hovermode="closest",
                    showlegend=True,
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.25,
                        xanchor="center",
                        x=0.5,
                    ),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    separators=".,",
                )

                st.plotly_chart(fig_dyn, use_container_width=True)

                st.caption(
                    f"Grafen viser porteføljens afkastbidrag grupperet efter {dyn_breakdown.lower()} "
                    f"og periodiseret efter {dyn_grouping.lower()}. "
                    "Løbende køb og salg korrigeres ved at behandle køb som kapital ind i gruppen "
                    "og salg som kapital ud af gruppen, så handler ikke fejlagtigt vises som afkast. "
                    "Ved procentvisning vises bidrag i procentpoint til porteføljeafkastet, "
                    "ikke gruppens interne afkastprocent."
                )


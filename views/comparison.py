"""
views/comparison.py — Sammenligning-fanen (al Streamlit-UI).

To modes:
  • Portefølje vs. benchmark — din TWR-kurve mod 1-4 benchmarks
  • Aktiv vs. aktiv          — 2-4 frit valgte tickers, med relativ-toggle

Alle beregninger ligger i analytics/comparison.py — denne fil er kun UI.
"""
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import PORTFOLIO_START
from utils.formatting import _da_num
from analytics.comparison import (
    compute_portfolio_twr_series, compute_benchmark_series,
    slice_price_to_period, align_to_common_index, rebase_series,
    compute_comparison_stats, compute_relative_series, ticker_name,
)


BENCHMARK_PRESETS = {
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "MSCI World": "IWDA.AS",
    "C25": "^OMXC25",
    "STOXX 600": "^STOXX",
    "Guld": "GC=F",
    "Bitcoin": "BTC-USD",
}
_TICKER_TO_PRESET = {tk: label for label, tk in BENCHMARK_PRESETS.items()}

# Farver til ikke-fremhævede serier (fremhævet serie er altid blå).
_CHART_COLORS = ["#d32f2f", "#f57c00", "#7b1fa2", "#00838f", "#558b2f"]

# Periodelængder i dage — bruges til at vise relevante periodeknapper i Mode 1.
_PERIOD_DAYS = {
    "1U": 7, "1M": 30, "3M": 90, "6M": 180,
    "1Å": 365, "3Å": 3 * 365, "5Å": 5 * 365,
}


# ----------------------------------------------------------------------
# Hoved-renderer
# ----------------------------------------------------------------------
def render_comparison(total_value, cashflows, cashflow_fracs):
    """Kaldt fra app.py. Viser mode-toggle og den valgte sub-renderer.
    Porteføljens TWR beregnes internt — total_value er en kroneværdiserie."""
    st.subheader("Sammenligning")
    mode = st.radio(
        "Sammenligningstype",
        ["Portefølje vs. benchmark", "Aktiv vs. aktiv"],
        horizontal=True, label_visibility="collapsed",
    )
    st.write("")
    if mode == "Portefølje vs. benchmark":
        _render_portfolio_vs_benchmark(total_value, cashflows, cashflow_fracs)
    else:
        _render_asset_vs_asset()


# ----------------------------------------------------------------------
# Mode 1: Portefølje vs. benchmark
# ----------------------------------------------------------------------
def _available_periods(start_date, end_date):
    """Periodeknapper der giver mening for porteføljens faktiske alder.
    Perioder længere end historikken udelades (de ville blot dublere "Maks")."""
    age = (end_date - start_date).days
    periods = [k for k, d in _PERIOD_DAYS.items() if d < age]
    ytd_start = pd.Timestamp(year=end_date.year, month=1, day=1)
    if pd.Timestamp(start_date) < ytd_start:
        periods.append("YTD")
    periods.append("Maks")
    return periods


def _render_portfolio_vs_benchmark(total_value, cashflows, cashflow_fracs):
    if total_value is None or len(total_value) == 0:
        st.info("Ingen porteføljedata.")
        return
    start_date = total_value.index[0]
    end_date = total_value.index[-1]

    if "cmp_benchmarks" not in st.session_state:
        st.session_state.cmp_benchmarks = ["^GSPC"]

    # --- Hurtigvalg-chips ---
    st.caption("Hurtigvalg — klik for at tilføje/fjerne benchmark (maks. 4)")
    cols = st.columns(len(BENCHMARK_PRESETS))
    for col, (label, tk) in zip(cols, BENCHMARK_PRESETS.items()):
        is_sel = tk in st.session_state.cmp_benchmarks
        if col.button(("✓ " if is_sel else "") + label,
                      key=f"bpreset_{tk}", use_container_width=True):
            if is_sel:
                st.session_state.cmp_benchmarks.remove(tk)
            elif len(st.session_state.cmp_benchmarks) < 4:
                st.session_state.cmp_benchmarks.append(tk)
            else:
                st.warning("Maks. 4 benchmarks ad gangen.")
            st.rerun()

    # --- Eget ticker-søgefelt ---
    custom = st.text_input("Tilføj benchmark (yfinance-ticker)",
                           key="bm_custom", placeholder="fx ^GDAXI")
    if custom:
        tk = custom.strip().upper()
        if tk and tk not in st.session_state.cmp_benchmarks:
            if len(st.session_state.cmp_benchmarks) < 4:
                st.session_state.cmp_benchmarks.append(tk)
                st.rerun()
            else:
                st.warning("Maks. 4 benchmarks ad gangen.")

    if st.session_state.cmp_benchmarks:
        st.caption("Valgte benchmarks — klik for at fjerne")
        chip_cols = st.columns(len(st.session_state.cmp_benchmarks))
        for col, tk in zip(chip_cols, list(st.session_state.cmp_benchmarks)):
            if col.button(f"✕ {_chip_label(tk)}", key=f"bmrm_{tk}",
                          use_container_width=True):
                st.session_state.cmp_benchmarks.remove(tk)
                st.rerun()

    # --- Periode ---
    periods = _available_periods(start_date, end_date)
    period = st.radio("Periode", periods, horizontal=True,
                      index=len(periods) - 1, label_visibility="collapsed")

    # --- Beregn serier ---
    twr_pct = compute_portfolio_twr_series(
        total_value, cashflows, cashflow_fracs, period)
    levels = {"Din portefølje": 1.0 + twr_pct / 100.0}

    fetch_start = str(PORTFOLIO_START.date())
    fetch_end = str((end_date + timedelta(days=1)).date())
    with st.spinner("Henter benchmark-data..."):
        for tk in st.session_state.cmp_benchmarks:
            bench = _safe_benchmark(tk, fetch_start, fetch_end)
            if bench is None or bench.empty:
                st.warning(f"Kunne ikke hente data for '{tk}'.")
                continue
            levels[_label_for(tk)] = slice_price_to_period(bench, period)

    aligned = align_to_common_index(levels)
    if not aligned:
        st.info("Ingen data at vise for den valgte periode.")
        return
    rebased = {k: rebase_series(v) for k, v in aligned.items()}

    _render_comparison_chart(rebased, highlight="Din portefølje")
    _render_stats_table(
        compute_comparison_stats(rebased),
        mode="benchmark", highlight="Din portefølje",
    )


# ----------------------------------------------------------------------
# Mode 2: Aktiv vs. aktiv
# ----------------------------------------------------------------------
def _render_asset_vs_asset():
    if "cmp_assets" not in st.session_state:
        st.session_state.cmp_assets = ["AAPL", "MSFT"]

    # --- Hurtigvalg-chips ---
    st.caption("Hurtigvalg")
    cols = st.columns(len(BENCHMARK_PRESETS))
    for col, (label, tk) in zip(cols, BENCHMARK_PRESETS.items()):
        is_sel = tk in st.session_state.cmp_assets
        if col.button(("✓ " if is_sel else "") + label,
                      key=f"apreset_{tk}", use_container_width=True):
            if is_sel:
                st.session_state.cmp_assets.remove(tk)
            elif len(st.session_state.cmp_assets) < 4:
                st.session_state.cmp_assets.append(tk)
            else:
                st.warning("Maks. 4 aktiver ad gangen.")
            st.rerun()

    # --- Eget ticker-søgefelt ---
    custom = st.text_input("Tilføj aktiv (yfinance-ticker)",
                           key="asset_custom", placeholder="fx AAPL")
    if custom:
        tk = custom.strip().upper()
        if tk and tk not in st.session_state.cmp_assets:
            if len(st.session_state.cmp_assets) < 4:
                st.session_state.cmp_assets.append(tk)
                st.rerun()
            else:
                st.warning("Maks. 4 aktiver ad gangen.")

    # --- Valgte aktiver med fjern-knapper ---
    if st.session_state.cmp_assets:
        st.caption("Valgte aktiver — klik for at fjerne")
        chip_cols = st.columns(len(st.session_state.cmp_assets))
        for col, tk in zip(chip_cols, list(st.session_state.cmp_assets)):
            if col.button(f"✕ {_chip_label(tk)}", key=f"rm_{tk}",
                          use_container_width=True):
                st.session_state.cmp_assets.remove(tk)
                st.rerun()

    if len(st.session_state.cmp_assets) < 2:
        st.info("Vælg mindst 2 aktiver at sammenligne.")
        return

    # --- Periode ---
    period = st.radio(
        "Periode", ["1M", "3M", "6M", "YTD", "1Å", "3Å", "5Å", "Maks"],
        horizontal=True, index=4, label_visibility="collapsed",
    )

    # --- Absolut / Relativt-toggle ---
    col_a, col_b = st.columns([1, 2])
    with col_a:
        rel_mode = st.toggle("Relativt til baseline")
    baseline = None
    if rel_mode:
        with col_b:
            baseline = st.selectbox(
                "Baseline", st.session_state.cmp_assets,
                label_visibility="collapsed",
            )

    # --- Beregn serier ---
    today = date.today()
    fetch_start = str(today - timedelta(days=6 * 365 + 30))
    fetch_end = str(today + timedelta(days=1))
    levels = {}
    with st.spinner("Henter kursdata..."):
        for tk in st.session_state.cmp_assets:
            s = _safe_benchmark(tk, fetch_start, fetch_end)
            if s is None or s.empty:
                st.warning(f"Kunne ikke hente data for '{tk}'.")
                continue
            levels[tk] = slice_price_to_period(s, period)

    aligned = align_to_common_index(levels)
    if len(aligned) < 2:
        st.info("Ikke nok data at vise for den valgte periode.")
        return
    rebased = {k: rebase_series(v) for k, v in aligned.items()}

    # Nøgletal beregnes altid på de absolutte afkast.
    stats = compute_comparison_stats(rebased)

    chart_data = rebased
    highlight = None
    if rel_mode and baseline in rebased:
        chart_data = compute_relative_series(rebased, baseline)
        highlight = baseline
        st.caption(
            f"Viser overskudsafkast relativt til **{baseline}** — "
            "linjer over 0 % outperformer baseline."
        )
    _render_comparison_chart(chart_data, highlight=highlight)
    _render_stats_table(stats, mode="asset")


# ----------------------------------------------------------------------
# Genanvendelige komponenter
# ----------------------------------------------------------------------
def _render_comparison_chart(series_dict, highlight=None):
    """Multi-linjegraf. Alle serier er allerede rebaseret til procent.
    Samme plotly-tilgang som den eksisterende TWR-graf i portfolio_overview."""
    fig = go.Figure()
    color_i = 0
    for label, s in series_dict.items():
        s = s.dropna()
        if s.empty:
            continue
        if label == highlight:
            line = dict(color="#1976d2", width=3.5)
        else:
            line = dict(color=_CHART_COLORS[color_i % len(_CHART_COLORS)],
                        width=2, dash="dot")
            color_i += 1
        fig.add_trace(go.Scatter(
            x=s.index, y=s.values, mode="lines", name=label, line=line,
            hovertemplate="<b>%{x|%d. %b %Y}</b><br>"
                          + label + ": %{y:.2f}%<extra></extra>",
        ))
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(0,0,0,0.3)")
    fig.update_layout(
        height=440, margin=dict(l=0, r=0, t=30, b=0),
        yaxis=dict(title="Afkast (%)", tickformat=".2f"),
        xaxis=dict(title=""),
        hovermode="x unified", plot_bgcolor="white", separators=".,",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_stats_table(stats, mode, highlight=None):
    """Nøgletals-tabel under grafen. mode="benchmark" viser outperformance;
    mode="asset" viser volatilitet og bedste måned."""
    if stats is None or stats.empty:
        return
    labels = list(stats.index)

    def pct(v, signed=True):
        if v is None or pd.isna(v):
            return "—"
        return f"{_da_num(v, signed=signed)} %"

    annual_vals = [stats.loc[l, "annualized_return"] for l in labels]
    rows = [["Afkast"] + [pct(stats.loc[l, "total_return"]) for l in labels]]
    rows.append(["Annualiseret"] + [pct(v) for v in annual_vals])
    rows.append(["Max drawdown"]
                + [pct(stats.loc[l, "max_drawdown"]) for l in labels])

    if mode == "benchmark" and highlight in labels:
        port_ret = stats.loc[highlight, "total_return"]
        op_row = ["Outperformance"]
        for l in labels:
            if l == highlight:
                op_row.append("—")
            else:
                diff = port_ret - stats.loc[l, "total_return"]
                op_row.append(f"{_da_num(diff, signed=True)} pp")
        rows.append(op_row)
    elif mode == "asset":
        rows.append(["Volatilitet"]
                    + [pct(stats.loc[l, "volatility"], signed=False)
                       for l in labels])
        rows.append(["Bedste måned"]
                    + [pct(stats.loc[l, "best_month"]) for l in labels])

    table = pd.DataFrame(rows, columns=["Nøgletal"] + labels)
    table = table.set_index("Nøgletal")
    st.table(table)
    if all(pd.isna(v) for v in annual_vals):
        st.caption("Annualiseret afkast vises ikke for perioder under "
                   "~6 måneder — CAGR bliver kraftigt misvisende på korte vinduer.")


# ----------------------------------------------------------------------
# Hjælpere
# ----------------------------------------------------------------------
def _label_for(ticker):
    """Kort navn til graf/tabel: preset-navn hvis kendt, ellers selve tickeren."""
    return _TICKER_TO_PRESET.get(ticker, ticker)


def _chip_label(ticker):
    """Chip-tekst med firmanavn: preset-navn, eller 'TICKER — Firmanavn' for
    søgte tickers (så man kan se hvad fx 'BE' dækker over)."""
    if ticker in _TICKER_TO_PRESET:
        return _TICKER_TO_PRESET[ticker]
    name = ticker_name(ticker)
    return f"{ticker} — {name}" if name else ticker


def _safe_benchmark(ticker, start, end):
    """compute_benchmark_series med fejlhåndtering — returnerer None ved fejl."""
    try:
        return compute_benchmark_series(ticker, start, end)
    except Exception:
        return None

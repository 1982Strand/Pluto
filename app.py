"""
Pluto Portefølje — Streamlit app
Korrekt afkastberegning efter Plutos princip:
  • Tidsvægtet afkast (TWR) — industri-standard performance-måling
  • Korrigeret for ind- og udbetalinger
  • Omregnet til DKK
  • Gebyrer automatisk inkluderet (de er allerede fratrukket i cashflows)
"""
import os

import numpy as np
import pandas as pd
import streamlit as st

# -------------------- KONFIGURATION --------------------
st.set_page_config(layout="wide", page_title="Pluto Portefølje", page_icon="📈")
from config import *
from styles import inject_styles
inject_styles()

# -------------------- IMPORTS --------------------
from data.cached import fetch_live_quotes, fetch_live_fx_rates
from analytics.portfolio import compute_portfolio_value_series, compute_deposits_dkk
from views.asset_detail import render_asset_detail
from views.portfolio_overview import render_portfolio_overview
from views.wallet import render_wallet

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
    # --- Indlæs data fra XLSX ---
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

    # --- Byg porteføljehistorik ---
    with st.spinner("Henter kurser og bygger porteføljehistorik..."):
        total_value, stock_value, cash_value_total, holdings, prices, usd_dkk, missing = \
            compute_portfolio_value_series(orders_df, dkk_tx, usd_tx, eur_tx, date_range)
        eur_dkk = prices.get("EURDKK=X", pd.Series(7.46, index=date_range)).reindex(date_range, method="ffill").bfill()
        cashflows, cashflow_fracs = compute_deposits_dkk(dkk_tx, usd_tx, eur_tx, usd_dkk, eur_dkk, date_range)

    # --- Override seneste punkt med live-priser ---
    # Så Total porteføljeværdi, TWR og Aktieværdi alle bruger samme priskilde
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

    # --- Tabs ---
    tab_main, tab_wallet = st.tabs(["📈 Portefølje", "👛 Pung"])

    with tab_main:
        render_portfolio_overview(
            orders_df, dkk_tx, usd_tx, eur_tx, positions_df, cash_df,
            prices, total_value, stock_value, cash_value_total,
            cashflows, cashflow_fracs, usd_dkk, eur_dkk,
        )

    with tab_wallet:
        render_wallet(cash_df, dkk_tx, usd_tx, eur_tx, orders_df)

except Exception as e:
    st.error(f"Fejl ved behandling af data: {e}")
    st.exception(e)
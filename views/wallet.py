"""
views/wallet.py — Pung-tab
Viser kontante beholdninger, indbetalingstidspunkter og bevægelser.
"""
import pandas as pd
import streamlit as st

from config import DEFAULT_DEPOSIT_TIME, DEPOSIT_TIMES_FILE
from data.deposits import load_deposit_times, save_deposit_times, _deposit_key
from utils.formatting import _da_num, format_currency
from datetime import time


def render_wallet(cash_df: pd.DataFrame, dkk_tx: pd.DataFrame,
                  usd_tx: pd.DataFrame, eur_tx: pd.DataFrame,
                  orders_df: pd.DataFrame) -> None:
    """Renderer Pung-tabben."""

    total_dkk = cash_df[cash_df["Currency"] == "DKK"]["End cash balance"].sum()
    st.caption("Kontant i alt (rapportperiodens slut)")
    st.title(f"{_da_num(total_dkk)} kr.")

    st.subheader("Konti")
    c1, c2, c3 = st.columns(3)
    dkk_s = cash_df[cash_df["Currency"] == "DKK"]["End cash balance"].sum()
    usd_s = cash_df[cash_df["Currency"] == "USD"]["End cash balance"].sum()
    eur_s = cash_df[cash_df["Currency"] == "EUR"]["End cash balance"].sum()
    c1.metric("Danske Kroner", format_currency(dkk_s, "DKK"))
    c2.metric("US Dollar",     format_currency(usd_s, "USD"))
    c3.metric("Euro",          format_currency(eur_s, "EUR"))

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
                "date":     _dnorm.normalize(),
                "date_str": _dnorm.strftime("%Y-%m-%d"),
                "amount":   float(_r["Amount"]),
                "currency": _ccy,
                "desc":     str(_r["Description"]),
            })
    _all_deposits.sort(key=lambda x: x["date"], reverse=True)

    if not _all_deposits:
        st.info("Ingen indbetalinger fundet i kontoudtoget.")
    else:
        _stored_times = load_deposit_times()
        _times_changed = False
        for _dep in _all_deposits:
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
            _amount_color  = "#388e3c" if _dep["amount"] >= 0 else "#d32f2f"
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
                step=60,
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
            val    = row["Notional (account currency)"]
            color  = "#d32f2f" if row["Side"] == "BUY" else "#388e3c"
            prefix = "-" if row["Side"] == "BUY" else "+"
            m2.markdown(
                f"<p style='text-align:right; color:{color}; font-weight:bold; margin-top:10px;'>"
                f"{prefix}{_da_num(val)}</p>",
                unsafe_allow_html=True,
            )
            st.divider()
    else:
        st.info("Ingen bevægelser i denne valuta.")
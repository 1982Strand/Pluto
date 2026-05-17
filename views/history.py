"""
views/history.py — Historik-tab
Viser livscyklus og performance for alle handlede tickers.
Aktive positioner øverst, lukkede nederst.
"""
import pandas as pd
import streamlit as st

from config import PLUTO_FX_SPREAD_RATE
from utils.formatting import (
    _safe_float, _da_num, format_quantity, _flatten_html,
)

# Valutasymboler
_CCY_SYM = {"USD": "$", "EUR": "€", "DKK": "kr."}


# ─────────────────────────────────────────────────────────────────────────────
# Hjælpefunktion: handelshistorik-tabel (delt logik med asset_detail.py)
# ─────────────────────────────────────────────────────────────────────────────
def _render_trade_table(
    sub: pd.DataFrame,
    asset_ccy: str,
    usd_dkk: float,
    eur_dkk: float,
) -> None:
    """Renderer handelshistorik-tabel for én ticker.
    Bruger FX Rate fra XLSX til at beregne præcis handelspris i aktivvaluta."""
    ccy_sym = _CCY_SYM.get(asset_ccy, asset_ccy)
    rows = []
    for _, r in sub.sort_values("Date", ascending=False).iterrows():
        d_str = r["Date"].strftime("%d. %b %Y %H:%M") if pd.notnull(r["Date"]) else "—"
        ac    = r.get("Account currency", "")
        qty   = r["Quantity"] if r["Quantity"] else 0
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
        ac_sym = _CCY_SYM.get(ac, ac)
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
        comm     = _safe_float(r.get("Commission (account currency)")) or 0
        rate_t   = usd_dkk if ac == "USD" else (eur_dkk if ac == "EUR" else 1.0)
        comm_dkk = comm * rate_t

        rows.append({
            "Dato":                d_str,
            "Side":                r["Side"],
            "Antal":               format_quantity(qty),
            f"Pris ({asset_ccy})": f"{ccy_sym} {_da_num(px_local)}",
            "Fra konto":           fra_konto_str,
            "Beløb (DKK)":         f"{_da_num(notional_dkk)} kr.",
            "Vekselgebyr (DKK)":   veksel_str,
            "Kurtage (DKK)":       f"{_da_num(comm_dkk)} kr." if comm_dkk else "—",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption(
            f"*Pris ({asset_ccy})* beregnes fra XLSX-felternes FX Rate. "
            f"*Vekselgebyr* = Beløb (DKK) × {PLUTO_FX_SPREAD_RATE * 100:.2f}% (Plutos prismodel)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Hjælpefunktion: livscyklus-tabel (HTML)
# ─────────────────────────────────────────────────────────────────────────────
def _render_lifecycle_table(entry: dict) -> None:
    """Renderer livscyklus-opsummering som HTML-tabel."""
    is_active = entry["is_active"]

    def _row(label, value, color=None):
        color_style = f"color:{color}; font-weight:600;" if color else ""
        return (
            f"<tr style='border-bottom:1px solid #f0f0f0;'>"
            f"  <td style='padding:8px 12px; color:#666; width:40%;'>{label}</td>"
            f"  <td style='padding:8px 12px; {color_style}'>{value}</td>"
            f"</tr>"
        )

    rows_html = ""
    rows_html += _row(
        "Første køb",
        entry["first_buy"].strftime("%d. %b %Y") if pd.notnull(entry["first_buy"]) else "—",
    )
    rows_html += _row(
        "Seneste handel",
        entry["last_activity"].strftime("%d. %b %Y") if pd.notnull(entry["last_activity"]) else "—",
    )
    if not is_active and entry["last_sale"]:
        rows_html += _row(
            "Afsluttet (last sale)",
            entry["last_sale"].strftime("%d. %b %Y"),
        )
    rows_html += _row("Investeret", f"{_da_num(entry['invested_dkk'])} kr.")
    if entry["realized_dkk"] > 0:
        rows_html += _row("Realiseret (salgsbeløb)", f"{_da_num(entry['realized_dkk'])} kr.")
    if is_active and entry["cost_basis_dkk"] > 0:
        rows_html += _row("Kostbasis (beholdning)", f"{_da_num(entry['cost_basis_dkk'])} kr.")
    if is_active:
        rows_html += _row("Nuværende værdi", f"{_da_num(entry['current_value_dkk'])} kr.")

    def _afkast_row(label, dkk, pct):
        return _row(
            label,
            f"{_da_num(dkk, signed=True)} kr. ({_da_num(pct, signed=True)}%)",
            color="#2e7d32" if dkk >= 0 else "#d32f2f",
        )

    if is_active:
        rows_html += _afkast_row(
            "Urealiseret afkast", entry["unrealized_dkk"], entry["unrealized_pct"]
        )
    if entry["realized_dkk"] > 0:
        rows_html += _afkast_row(
            "Realiseret afkast", entry["realized_gain_dkk"], entry["realized_pct"]
        )
    rows_html += _afkast_row(
        "Samlet afkast", entry["total_return_dkk"], entry["total_return_pct"]
    )

    table_html = f"""
    <table style='width:100%; border-collapse:collapse; font-size:14px; margin-bottom:8px;'>
      <tbody>{rows_html}</tbody>
    </table>
    """
    st.markdown(_flatten_html(table_html), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hjælpefunktion: nøgletal-metrics
# ─────────────────────────────────────────────────────────────────────────────
def _render_keyfigures(entry: dict, usd_dkk: float, eur_dkk: float) -> None:
    """Renderer GAK og evt. vægtet salgspris."""
    is_active   = entry["is_active"]
    asset_ccy   = entry["asset_ccy"]
    ccy_sym     = _CCY_SYM.get(asset_ccy, asset_ccy)
    fx_rate     = usd_dkk if asset_ccy == "USD" else (eur_dkk if asset_ccy == "EUR" else 1.0)

    cols = []

    # GAK — kun hvis aktiv og har beholdning
    if is_active and entry["total_qty"] > 0.001 and entry["invested_dkk"] > 0:
        gak_dkk    = entry["invested_dkk"] / entry["total_qty"]
        gak_valuta = gak_dkk / fx_rate if fx_rate else 0
        if is_active and entry.get("gak_valuta") is not None:
            gak_valuta = entry["gak_valuta"]
            gak_dkk    = gak_valuta * fx_rate
            cols.append(("GAK (DKK)",        f"{_da_num(gak_dkk)} kr."))
            cols.append((f"GAK ({asset_ccy})", f"{ccy_sym} {_da_num(gak_valuta)}"))

    # Vægtet salgspris — kun hvis der er solgt
    sells = entry["orders"][entry["orders"]["Side"] == "SELL"]
    if not sells.empty:
        total_sell_qty = float(
            sells["Quantity"].sum() if "Quantity" in sells.columns else 0
        )
        total_sell_dkk = float(entry["realized_dkk"])
        if total_sell_qty > 0:
            avg_sell_dkk    = total_sell_dkk / total_sell_qty
            avg_sell_valuta = avg_sell_dkk / fx_rate if fx_rate else 0
            cols.append(("Vægtet salgspris (DKK)", f"{_da_num(avg_sell_dkk)} kr."))
            cols.append((f"Vægtet salgspris ({asset_ccy})", f"{ccy_sym} {_da_num(avg_sell_valuta)}"))

    if cols:
        metric_cols = st.columns(len(cols))
        for col, (label, value) in zip(metric_cols, cols):
            col.metric(label, value)


# ─────────────────────────────────────────────────────────────────────────────
# Hoved-funktion
# ─────────────────────────────────────────────────────────────────────────────
def render_history(
    lifecycle: list,
    usd_dkk: float,
    eur_dkk: float,
) -> None:
    """Renderer Historik-tabben med expanders pr. ticker."""

    if not lifecycle:
        st.info("Ingen handelshistorik fundet. Upload venligst en Pluto-fil.")
        return

    # ── Resumé-metrics ──────────────────────────────────────────────────────
    active_count  = sum(1 for x in lifecycle if x["is_active"])
    closed_count  = sum(1 for x in lifecycle if not x["is_active"])
    realized_gain = sum(x["realized_gain_dkk"] for x in lifecycle)
    unrealized    = sum(x["unrealized_dkk"] for x in lifecycle)
    total_gain    = realized_gain + unrealized

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Aktive positioner",  active_count)
    m2.metric("Afsluttede positioner", closed_count)
    m3.metric("Urealiseret afkast", f"{_da_num(unrealized, signed=True)} kr.")
    m4.metric("Realiseret afkast",  f"{_da_num(realized_gain, signed=True)} kr.")
    m5.metric("Samlet afkast",      f"{_da_num(total_gain, signed=True)} kr.")

    st.write("---")

    # ── Expanders pr. ticker ─────────────────────────────────────────────────
    for entry in lifecycle:
        ticker    = entry["ticker"]
        name      = entry["name"]
        is_active = entry["is_active"]
        ret_dkk   = entry["total_return_dkk"]
        ret_pct   = entry["total_return_pct"]

        # Badge og returntekst til expander-titlen (kun ren tekst muligt her)
        badge      = "🟢 Aktiv" if is_active else "⚫ Afsluttet"
        ret_sign   = "+" if ret_dkk >= 0 else ""
        ret_label  = f"{ret_sign}{_da_num(ret_dkk)} kr.  ({ret_sign}{_da_num(ret_pct)}%)"
        exp_title  = f"{badge}  |  {ticker} — {name}  |  {ret_label}"

        with st.expander(exp_title, expanded=False):

            # Afkast-farvet overskrift inde i expander
            ret_color = "#2e7d32" if ret_dkk >= 0 else "#d32f2f"
            st.markdown(
                f"<p style='font-size:20px; font-weight:700; color:{ret_color}; margin:4px 0 12px;'>"
                f"{ret_sign}{_da_num(ret_dkk)} kr. &nbsp;"
                f"<span style='font-size:15px; font-weight:500;'>"
                f"({ret_sign}{_da_num(ret_pct)}%)</span></p>",
                unsafe_allow_html=True,
            )

            # ── Livscyklus-tabel ─────────────────────────────────────────
            st.caption("**Livscyklus**")
            _render_lifecycle_table(entry)

            st.write("")

            # ── Nøgletal ─────────────────────────────────────────────────
            st.caption("**Nøgletal**")
            _render_keyfigures(entry, usd_dkk, eur_dkk)

            st.write("")

            # ── Handelshistorik ───────────────────────────────────────────
            st.caption("**Handelshistorik**")
            _render_trade_table(
                entry["orders"],
                entry["asset_ccy"],
                usd_dkk,
                eur_dkk,
            )
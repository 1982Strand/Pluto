"""
analytics/holdings.py — Beregning bag "Mine Aktier"-sektionen.

100% Streamlit-fri og uden datahentning: alle priser, live-quotes og meta
modtages som parametre. Renderingen ligger i views/portfolio_overview.py.
"""
import numpy as np
import pandas as pd

from config import PLUTO_FX_SPREAD_RATE
from utils.formatting import _safe_float


def compute_portfolio_costs(orders_df, usd_dkk_now, eur_dkk_now):
    """Samlede omkostninger i DKK: kurtage (fra XLSX) + FX-spread (0,15%).

    FX-spread = PLUTO_FX_SPREAD_RATE * abs(Notional, DKK) for USD/EUR-handler.
    Returnerer dict: commission_dkk, fx_spread_dkk, total_dkk.
    """
    commission_dkk = 0.0
    fx_spread_dkk = 0.0
    for _, order in orders_df.iterrows():
        comm = _safe_float(order.get("Commission (account currency)"))
        if comm and comm != 0:
            acc = order.get("Account currency")
            if acc == "USD":
                rate = usd_dkk_now or 6.85
            elif acc == "EUR":
                rate = eur_dkk_now or 7.46
            else:
                rate = 1.0
            commission_dkk += comm * rate

        asset_ccy = order.get("Asset currency")
        if asset_ccy in ("USD", "EUR"):
            notional = _safe_float(order.get("Notional, DKK"))
            if notional:
                fx_spread_dkk += abs(notional) * PLUTO_FX_SPREAD_RATE

    return {
        "commission_dkk": commission_dkk,
        "fx_spread_dkk": fx_spread_dkk,
        "total_dkk": commission_dkk + fx_spread_dkk,
    }


def compute_pnl_summary(total_value_now, total_invested, cash_value_now,
                        total_deposits, total_costs):
    """Urealiseret/realiseret afkast + reelt investeret. Ren aritmetik.

    Returnerer dict: unrealized_pnl, unrealized_pct, realized_pnl,
    realized_pct, reelt_investeret.
    """
    unrealized_pnl = total_value_now - total_invested
    realized_all_in = total_invested + cash_value_now - total_deposits
    realized_pnl = realized_all_in + total_costs
    reelt_investeret = total_deposits - total_costs

    unrealized_pct = (unrealized_pnl / total_invested * 100) if total_invested else 0
    realized_pct = (realized_pnl / total_deposits * 100) if total_deposits else 0

    return {
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pct": unrealized_pct,
        "realized_pnl": realized_pnl,
        "realized_pct": realized_pct,
        "reelt_investeret": reelt_investeret,
    }


def active_positions(orders_df):
    """Aktive positioner (Qty_Adj > 0.001) grupperet pr. ticker + valuta.

    Returnerer DataFrame med kolonner: Ticker, Asset currency, Qty_Adj.
    """
    orders = orders_df.copy()
    orders["Qty_Adj"] = np.where(
        orders["Side"] == "BUY", orders["Quantity"], -orders["Quantity"]
    )
    grouped = (
        orders.groupby(["Ticker", "Asset currency"])
        .agg(Qty_Adj=("Qty_Adj", "sum"))
        .reset_index()
    )
    return grouped[grouped["Qty_Adj"] > 0.001].copy()


def compute_live_stock_value(active_df, prices, live_quotes, usd_now, eur_now):
    """Samlet live-aktieværdi i DKK for de aktive positioner.

    Pris pr. ticker: live-quote hvis muligt, ellers seneste daglige slutkurs.
    Tickere uden nogen pris springes over.
    """
    total_dkk = 0.0
    for _, row in active_df.iterrows():
        ticker = row["Ticker"]
        price = live_quotes.get(ticker, {}).get("live")
        if price is None and ticker in prices.columns:
            ser = prices[ticker].dropna()
            if len(ser):
                price = float(ser.iloc[-1])
        if price is None:
            continue
        ccy = row["Asset currency"]
        rate = usd_now if ccy == "USD" else (eur_now if ccy == "EUR" else 1.0)
        total_dkk += float(row["Qty_Adj"]) * price * rate
    return total_dkk


def _empty_buckets():
    return {
        "sector": {}, "asset_class": {}, "currency": {},
        "region": {}, "country": {},
    }


def _add_to_bucket(bucket_dict, key, value_dkk, invested_dkk, pos_entry):
    """Akkumulér en position i en fordelings-bucket."""
    b = bucket_dict.setdefault(
        key, {"count": 0, "value_dkk": 0.0, "invested_dkk": 0.0, "positions": []}
    )
    b["count"] += 1
    b["value_dkk"] += value_dkk
    b["invested_dkk"] += invested_dkk
    b["positions"].append(pos_entry)


def compute_holdings_breakdown(orders_df, positions_df, prices, live_quotes,
                               ticker_meta, cash_df, usd_dkk_now, eur_dkk_now,
                               usd_dkk, eur_dkk, afkast_periode, ytd_start):
    """Beregner alt bag "Mine Aktier", "Alle beholdninger" og "Fordelinger".

    Returnerer kun rå tal (ingen formatstrenge):
      positions           liste af position-dicts
      all_holdings        positioner + kontanter (til donut)
      buckets             {sector, asset_class, currency, region, country}
      total_value_dkk     aktieværdi (live)
      grand_total_dkk     aktieværdi + kontanter
      total_invested_dkk  kostbasis for aktive positioner

    afkast_periode er "1D", "YTD" eller "Maks" og styrer pnl_pos_*-felterne.
    """
    orders = orders_df.copy()
    orders["Qty_Adj"] = np.where(
        orders["Side"] == "BUY", orders["Quantity"], -orders["Quantity"]
    )
    orders["DKK_Adj"] = np.where(
        orders["Side"] == "BUY", orders["Notional, DKK"], -orders["Notional, DKK"]
    )
    portfolio = (
        orders.groupby(["Ticker", "Name", "Asset currency"])
        .agg(Qty_Adj=("Qty_Adj", "sum"), DKK_Adj=("DKK_Adj", "sum"))
        .reset_index()
    )
    active = portfolio[portfolio["Qty_Adj"] > 0.001].copy()

    if active.empty:
        return {
            "positions": [],
            "all_holdings": [],
            "buckets": _empty_buckets(),
            "total_value_dkk": 0.0,
            "grand_total_dkk": 0.0,
            "total_invested_dkk": 0.0,
        }

    avg_entry_by_ticker = {}
    if ("Ticker" in positions_df.columns
            and "Average entry price (asset currency)" in positions_df.columns):
        for _, row in positions_df.iterrows():
            avg_entry_by_ticker[row["Ticker"]] = _safe_float(
                row["Average entry price (asset currency)"]
            )

    # Plutos egen kostbasis i DKK pr. ticker (glidende gennemsnit, historisk FX).
    # Korrekt for delvist solgte positioner — modsat sum(BUY)-sum(SELL) der
    # lækker realiseret gevinst ind i kostbasen.
    cost_dkk_by_ticker = {}
    if ("Ticker" in positions_df.columns
            and "Amount, DKK" in positions_df.columns):
        for _, row in positions_df.iterrows():
            cost_dkk_by_ticker[row["Ticker"]] = _safe_float(row["Amount, DKK"])

    per_sector, per_asset_class = {}, {}
    per_currency, per_region, per_country = {}, {}, {}
    positions = []
    all_holdings = []
    total_v = 0.0
    total_invested = 0.0

    for _, s in active.iterrows():
        t = s["Ticker"]
        q = live_quotes.get(t, {})
        ccy = s["Asset currency"]

        fallback_close = None
        if t in prices.columns:
            ser = prices[t].dropna()
            if len(ser) >= 1:
                fallback_close = float(ser.iloc[-1])

        # 'Sidste luk' = seneste komplette regular-session close.
        today_c = q.get("today_close")
        prev_c = q.get("prev_close")
        prev_prev_c = q.get("prev_prev_close")
        if today_c is not None:
            sidste_luk = today_c
            forrige_luk = prev_c
        else:
            sidste_luk = prev_c
            forrige_luk = prev_prev_c
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
        else:
            d_yest = None
            d_yest_pct = None

        if sidste_luk:
            d_now = live - sidste_luk
            d_now_pct = d_now / sidste_luk * 100
        else:
            d_now = None
            d_now_pct = None

        if ccy == "USD":
            rate = usd_dkk_now
        elif ccy == "EUR":
            rate = eur_dkk_now
        else:
            rate = 1.0

        # Kostbasis: Plutos 'Amount, DKK' (korrekt ved delvist salg).
        # Fald tilbage til nettopengestrøm hvis positionen mangler i arket.
        cost_basis = cost_dkk_by_ticker.get(t)
        if cost_basis is None:
            cost_basis = float(s["DKK_Adj"])

        gak_pluto = avg_entry_by_ticker.get(t)
        if gak_pluto is not None:
            gak_valuta = gak_pluto
        else:
            gak_valuta = (cost_basis / s["Qty_Adj"]) / rate if rate else 0
        vaerdi_dkk = s["Qty_Adj"] * live * rate
        pos_pct = (vaerdi_dkk - cost_basis) / cost_basis * 100 if cost_basis else 0
        total_v += vaerdi_dkk
        total_invested += cost_basis

        # Periodisk afkast (kr. + %) afhængigt af afkast_periode-vælger
        if afkast_periode == "1D":
            if sidste_luk and live is not None and sidste_luk > 0:
                pnl_pos_pct = (live - sidste_luk) / sidste_luk * 100
                pnl_pos_dkk = float(s["Qty_Adj"]) * (live - sidste_luk) * rate
            else:
                pnl_pos_pct = 0.0
                pnl_pos_dkk = 0.0
        elif afkast_periode == "YTD":
            first_buy = orders[orders["Ticker"] == t]["TradeDate"].min()
            used_jan1 = False
            if pd.notnull(first_buy) and first_buy < ytd_start and t in prices.columns:
                jan1_close = prices[t].asof(ytd_start)
                if pd.notnull(jan1_close) and float(jan1_close) > 0 and live is not None:
                    jan1_close_f = float(jan1_close)
                    pnl_pos_pct = (live - jan1_close_f) / jan1_close_f * 100
                    pnl_pos_dkk = float(s["Qty_Adj"]) * (live - jan1_close_f) * rate
                    used_jan1 = True
            if not used_jan1:
                pnl_pos_dkk = vaerdi_dkk - cost_basis
                pnl_pos_pct = pos_pct
        else:  # Maks
            pnl_pos_dkk = vaerdi_dkk - cost_basis
            pnl_pos_pct = pos_pct

        pos_entry = {
            "ticker": t,
            "name": s["Name"],
            "qty": float(s["Qty_Adj"]),
            "value_dkk": vaerdi_dkk,
            "invested_dkk": cost_basis,
            "gain_dkk": vaerdi_dkk - cost_basis,
            "pos_pct": pos_pct,
        }
        all_holdings.append(pos_entry)

        positions.append({
            "ticker": t,
            "name": s["Name"],
            "ccy": ccy,
            "qty": float(s["Qty_Adj"]),
            "gak_valuta": gak_valuta,
            "sidste_luk": sidste_luk,
            "forrige_luk": forrige_luk,
            "live": live,
            "d_yest": d_yest,
            "d_yest_pct": d_yest_pct,
            "d_now": d_now,
            "d_now_pct": d_now_pct,
            "vaerdi_dkk": vaerdi_dkk,
            "invested_dkk": cost_basis,
            "pos_pct": pos_pct,
            "pnl_pos_dkk": pnl_pos_dkk,
            "pnl_pos_pct": pnl_pos_pct,
        })

        meta = ticker_meta.get(t, {})
        invested = cost_basis
        _add_to_bucket(per_sector, meta.get("sector", "Andet"), vaerdi_dkk, invested, pos_entry)
        _add_to_bucket(per_asset_class, meta.get("asset_class", "Andet"), vaerdi_dkk, invested, pos_entry)
        _add_to_bucket(per_currency, ccy, vaerdi_dkk, invested, pos_entry)
        _add_to_bucket(per_region, meta.get("region", "Andet"), vaerdi_dkk, invested, pos_entry)
        _add_to_bucket(per_country, meta.get("country", "Andet"), vaerdi_dkk, invested, pos_entry)

    # Kontanter: tilføj til Aktivklasser ("Kontanter") og Valutaer (pr. ccy)
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
                "invested_dkk": bal_dkk,  # cash = ingen gevinst
                "gain_dkk": 0.0,
                "pos_pct": 0.0,
                "hide_zero_gain": True,
            })

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

    return {
        "positions": positions,
        "all_holdings": all_holdings,
        "buckets": {
            "sector": per_sector,
            "asset_class": per_asset_class,
            "currency": per_currency,
            "region": per_region,
            "country": per_country,
        },
        "total_value_dkk": total_v,
        "grand_total_dkk": grand_total,
        "total_invested_dkk": total_invested,
    }

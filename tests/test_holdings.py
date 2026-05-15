"""Tests for analytics/holdings.py — beregning bag "Mine Aktier"-sektionen."""
import pandas as pd

from analytics.holdings import (
    compute_portfolio_costs,
    compute_pnl_summary,
    active_positions,
    compute_live_stock_value,
    compute_holdings_breakdown,
)


def test_portfolio_costs_commission_and_fx_spread():
    orders = pd.DataFrame({
        "Commission (account currency)": [1.0, 2.0, 0.0],
        "Account currency": ["USD", "DKK", "USD"],
        "Asset currency": ["USD", "DKK", "EUR"],
        "Notional, DKK": [1000.0, 500.0, 2000.0],
    })
    costs = compute_portfolio_costs(orders, usd_dkk_now=7.0, eur_dkk_now=7.5)
    # Kurtage: 1.0*7.0 (USD) + 2.0*1.0 (DKK) + 0 = 9.0
    assert costs["commission_dkk"] == 9.0
    # FX-spread: 0,15% af (1000 USD-handel + 2000 EUR-handel) = 4.5
    assert abs(costs["fx_spread_dkk"] - 4.5) < 1e-9
    assert abs(costs["total_dkk"] - 13.5) < 1e-9


def test_pnl_summary_basic():
    pnl = compute_pnl_summary(
        total_value_now=12000.0,
        total_invested=10000.0,
        cash_value_now=500.0,
        total_deposits=11000.0,
        total_costs=300.0,
    )
    assert pnl["unrealized_pnl"] == 2000.0           # 12000 - 10000
    assert pnl["unrealized_pct"] == 20.0             # 2000/10000*100
    # realized_all_in = 10000 + 500 - 11000 = -500; realized = -500 + 300 = -200
    assert pnl["realized_pnl"] == -200.0
    assert pnl["reelt_investeret"] == 10700.0        # 11000 - 300


def _orders_fixture():
    return pd.DataFrame({
        "Side": ["BUY", "BUY", "SELL", "BUY"],
        "Ticker": ["AAA", "AAA", "AAA", "BBB"],
        "Name": ["Alfa", "Alfa", "Alfa", "Beta"],
        "Asset currency": ["USD", "USD", "USD", "DKK"],
        "Quantity": [10.0, 5.0, 15.0, 4.0],
        "Notional, DKK": [700.0, 350.0, 1200.0, 400.0],
        "TradeDate": pd.to_datetime(
            ["2026-02-01", "2026-02-02", "2026-03-01", "2026-02-05"]
        ),
    })


def test_active_positions_excludes_fully_sold():
    active = active_positions(_orders_fixture())
    # AAA: 10+5-15 = 0 → ekskluderet. BBB: 4 → med.
    assert list(active["Ticker"]) == ["BBB"]
    assert active.iloc[0]["Qty_Adj"] == 4.0


def test_live_stock_value_uses_live_and_skips_priceless():
    active = pd.DataFrame({
        "Ticker": ["AAA", "BBB"],
        "Asset currency": ["USD", "DKK"],
        "Qty_Adj": [10.0, 4.0],
    })
    prices = pd.DataFrame({"AAA": [100.0, 110.0]})
    live_quotes = {"AAA": {"live": 120.0}}  # BBB mangler helt
    value = compute_live_stock_value(
        active, prices, live_quotes, usd_now=7.0, eur_now=7.5
    )
    # AAA: 10 * 120 * 7.0 = 8400. BBB: ingen pris → springes over.
    assert value == 8400.0


def _breakdown_inputs():
    orders = pd.DataFrame({
        "Side": ["BUY", "BUY"],
        "Ticker": ["AAA", "BBB"],
        "Name": ["Alfa Inc", "Beta A/S"],
        "Asset currency": ["USD", "DKK"],
        "Quantity": [10.0, 5.0],
        "Notional, DKK": [700.0, 500.0],
        "TradeDate": pd.to_datetime(["2026-02-01", "2026-02-05"]),
    })
    positions_df = pd.DataFrame(
        columns=["Ticker", "Average entry price (asset currency)"]
    )
    prices = pd.DataFrame({"AAA": [10.0, 11.0], "BBB": [100.0, 100.0]})
    live_quotes = {
        "AAA": {"live": 12.0, "prev_close": 11.0,
                "prev_prev_close": 10.0, "today_close": None},
        "BBB": {"live": 110.0, "prev_close": 100.0,
                "prev_prev_close": 100.0, "today_close": None},
    }
    ticker_meta = {
        "AAA": {"sector": "Teknologi", "asset_class": "Aktier",
                "region": "Nordamerika", "country": "United States"},
        "BBB": {"sector": "Industri", "asset_class": "Aktier",
                "region": "Europa", "country": "Denmark"},
    }
    cash_df = pd.DataFrame({"Currency": ["DKK"], "End cash balance": [1000.0]})
    return orders, positions_df, prices, live_quotes, ticker_meta, cash_df


def test_holdings_breakdown_values_and_cash():
    orders, positions_df, prices, live_quotes, ticker_meta, cash_df = _breakdown_inputs()
    result = compute_holdings_breakdown(
        orders, positions_df, prices, live_quotes, ticker_meta, cash_df,
        usd_dkk_now=7.0, eur_dkk_now=7.5,
        usd_dkk=pd.Series([7.0, 7.0]), eur_dkk=pd.Series([7.5, 7.5]),
        afkast_periode="Maks", ytd_start=pd.Timestamp("2026-01-01"),
    )
    assert len(result["positions"]) == 2

    aaa = next(p for p in result["positions"] if p["ticker"] == "AAA")
    bbb = next(p for p in result["positions"] if p["ticker"] == "BBB")
    assert aaa["vaerdi_dkk"] == 840.0   # 10 * 12.0 live * 7.0
    assert bbb["vaerdi_dkk"] == 550.0   # 5 * 110 live * 1.0

    assert result["total_value_dkk"] == 1390.0
    assert result["grand_total_dkk"] == 2390.0   # + 1000 DKK kontant
    assert result["total_invested_dkk"] == 1200.0

    # Kontanter flettet ind i aktivklasse-bucket
    assert "Kontanter" in result["buckets"]["asset_class"]
    assert result["buckets"]["sector"]["Teknologi"]["value_dkk"] == 840.0
    assert result["buckets"]["sector"]["Industri"]["value_dkk"] == 550.0


def test_holdings_breakdown_empty_when_no_active_positions():
    orders = pd.DataFrame({
        "Side": ["BUY", "SELL"],
        "Ticker": ["AAA", "AAA"],
        "Name": ["Alfa", "Alfa"],
        "Asset currency": ["USD", "USD"],
        "Quantity": [10.0, 10.0],
        "Notional, DKK": [700.0, 800.0],
        "TradeDate": pd.to_datetime(["2026-02-01", "2026-03-01"]),
    })
    result = compute_holdings_breakdown(
        orders, pd.DataFrame(columns=["Ticker"]), pd.DataFrame(),
        {}, {}, pd.DataFrame({"Currency": [], "End cash balance": []}),
        usd_dkk_now=7.0, eur_dkk_now=7.5,
        usd_dkk=pd.Series([7.0]), eur_dkk=pd.Series([7.5]),
        afkast_periode="Maks", ytd_start=pd.Timestamp("2026-01-01"),
    )
    assert result["positions"] == []
    assert result["total_value_dkk"] == 0.0
    assert result["grand_total_dkk"] == 0.0

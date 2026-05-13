import numpy as np
import pandas as pd
from datetime import timedelta

from config import DEFAULT_DEPOSIT_TIME
from data.deposits import load_deposit_times, _deposit_key, _time_to_frac
from data.fetch import fetch_price_history, fetch_price_history_intraday

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
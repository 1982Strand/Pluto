import numpy as np
import pandas as pd
from datetime import timedelta

from config import DEFAULT_DEPOSIT_TIME
from data.deposits import load_deposit_times, _deposit_key, _time_to_frac
from data.cached import fetch_price_history, fetch_price_history_intraday
from utils.formatting import _safe_float

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


def split_signed_return_segments(x_arr, y_arr, v_arr):
    """Opdel en afkastserie i kontinuerlige fortegns-segmenter.

    Indsætter interpolerede nulpunkter ved hver fortegns-krydsning, så en
    fill-graf ikke bløder mellem grøn (positiv) og rød (negativ).

    x_arr: tidsstempler, y_arr: afkast (%), v_arr: porteføljeværdi (DKK).
    Returnerer liste af (sign, segment) hvor sign er "pos"/"neg"/None og
    segment er en liste af (x, y, v)-tupler. Tegningen ligger i view-laget.
    """
    y_arr = np.asarray(y_arr, dtype=float)
    v_arr = np.asarray(v_arr, dtype=float)

    ex_list, ey_list, ev_list = [], [], []
    for i in range(len(y_arr)):
        ex_list.append(x_arr[i])
        ey_list.append(y_arr[i])
        ev_list.append(v_arr[i])

        if i + 1 < len(y_arr):
            y0, y1 = y_arr[i], y_arr[i + 1]
            if (y0 > 0 and y1 < 0) or (y0 < 0 and y1 > 0):
                t0 = pd.Timestamp(x_arr[i]).value
                t1 = pd.Timestamp(x_arr[i + 1]).value
                frac = y0 / (y0 - y1)
                x0 = pd.Timestamp(int(t0 + frac * (t1 - t0))).to_numpy()
                v0, v1 = v_arr[i], v_arr[i + 1]
                v_zero = v0 + frac * (v1 - v0)
                ex_list.append(x0)
                ey_list.append(0.0)
                ev_list.append(v_zero)

    segments = []
    cur_sign = None
    cur_seg = []
    for x, y, v in zip(ex_list, ey_list, ev_list):
        if y > 0:
            new_sign = "pos"
        elif y < 0:
            new_sign = "neg"
        else:
            new_sign = None  # zero — ambivalent, tilhører begge

        if new_sign is None:
            # Luk eksisterende segment og start nyt omkring nulpunktet
            if cur_seg:
                cur_seg.append((x, y, v))
                segments.append((cur_sign, cur_seg))
                cur_seg = [(x, y, v)]
                cur_sign = None
            else:
                cur_seg = [(x, y, v)]
                cur_sign = None
        elif cur_sign is None or cur_sign == new_sign:
            cur_seg.append((x, y, v))
            cur_sign = new_sign
        else:
            # Fortegns-skift: luk gammelt segment og start nyt
            segments.append((cur_sign, cur_seg))
            cur_seg = [(x, y, v)]
            cur_sign = new_sign

    if cur_seg:
        segments.append((cur_sign, cur_seg))

    return segments


def compute_portfolio_return_dynamics(
    total_value: pd.Series,
    cashflows: pd.Series,
    cashflow_fracs=None,
    grouping: str = "Månedlig",
) -> pd.DataFrame:
    """
    Beregner periodisk porteføljeafkast til søjlediagram:
    daglig, ugentlig, månedlig eller årlig.

    Returnerer DataFrame med:
    - period_start
    - period_end
    - period_label
    - return_dkk
    - return_pct

    Beregningen er cashflow-korrigeret efter samme princip som resten af appen:
    Afkast DKK = slutværdi - startværdi - netto cashflows i perioden

    Afkast % beregnes med en Modified Dietz-lignende kapitalbase:
    kapitalbase = startværdi + sum(cashflow_i * vægt_i)

    Funktionen er 100% Streamlit-fri og hører derfor hjemme i analytics-laget.
    """

    if total_value is None or len(total_value) < 2:
        return pd.DataFrame(columns=[
            "period_start", "period_end", "period_label",
            "return_dkk", "return_pct"
        ])

    grouping_map = {
        "Daglig": "D",
        "Ugentlig": "W",
        "Månedlig": "M",
        "Årlig": "Y",
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Yearly": "Y",
        "D": "D",
        "W": "W",
        "M": "M",
        "Y": "Y",
    }

    freq = grouping_map.get(grouping, "M")

    values = total_value.copy()
    values.index = pd.to_datetime(values.index)
    values = values.sort_index().astype(float).replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) < 2:
        return pd.DataFrame(columns=[
            "period_start", "period_end", "period_label",
            "return_dkk", "return_pct"
        ])

    cf = cashflows.copy() if cashflows is not None else pd.Series(0.0, index=values.index)
    cf.index = pd.to_datetime(cf.index)
    cf = cf.reindex(values.index).fillna(0.0).astype(float)

    if cashflow_fracs is None:
        fracs = pd.Series(0.0, index=values.index)
    else:
        fracs = cashflow_fracs.copy()
        fracs.index = pd.to_datetime(fracs.index)
        fracs = fracs.reindex(values.index).fillna(0.0).astype(float)

    df = pd.DataFrame({
        "value": values,
        "cashflow": cf,
        "frac": fracs,
    }, index=values.index)

    df = df.sort_index()
    df["row_no"] = np.arange(len(df))

    if freq == "D":
        df["period"] = df.index.to_period("D")
    elif freq == "W":
        df["period"] = df.index.to_period("W-SUN")
    elif freq == "Y":
        df["period"] = df.index.to_period("Y")
    else:
        df["period"] = df.index.to_period("M")

    month_names_da = {
        1: "jan", 2: "feb", 3: "mar", 4: "apr",
        5: "maj", 6: "jun", 7: "jul", 8: "aug",
        9: "sep", 10: "okt", 11: "nov", 12: "dec",
    }

    def _period_label(period_key, period_end):
        if freq == "D":
            return period_end.strftime("%d-%m-%Y")

        if freq == "W":
            iso = period_end.isocalendar()
            return f"uge {int(iso.week)} {int(iso.year)}"

        if freq == "Y":
            return str(period_end.year)

        return f"{month_names_da.get(period_end.month, period_end.strftime('%b'))} {period_end.year}"

    rows = []

    for period_key, sub in df.groupby("period", sort=True):
        if sub.empty:
            continue

        first_row = int(sub["row_no"].iloc[0])
        last_row = int(sub["row_no"].iloc[-1])

        # For perioden bruges værdien lige før perioden som start,
        # hvis den findes. Ellers bruges første punkt i perioden.
        if first_row > 0:
            start_row = first_row - 1
            cf_sub = sub
        else:
            start_row = first_row
            # Første datapunkt er startværdi; cashflows på selve startpunktet
            # medtages ikke som periode-cashflow for at undgå dobbeltregning.
            cf_sub = sub.iloc[1:]

        start_ts = df.index[start_row]
        end_ts = df.index[last_row]

        if end_ts <= start_ts:
            continue

        start_value = float(df["value"].iloc[start_row])
        end_value = float(df["value"].iloc[last_row])

        cf_sum = float(cf_sub["cashflow"].sum()) if not cf_sub.empty else 0.0
        return_dkk = end_value - start_value - cf_sum

        period_days = max((end_ts - start_ts).total_seconds() / 86400.0, 1e-9)

        weighted_cf = 0.0
        if not cf_sub.empty:
            for cf_date, cf_row in cf_sub.iterrows():
                cf_amount = float(cf_row["cashflow"])
                cf_frac = float(cf_row["frac"])

                # Cashflow-tidspunkt indenfor dagen.
                cf_ts = cf_date + pd.to_timedelta(cf_frac, unit="D")

                # Vægt = hvor stor del af perioden cashflowet har været investeret.
                weight = (end_ts - cf_ts).total_seconds() / 86400.0 / period_days
                weight = min(max(weight, 0.0), 1.0)

                weighted_cf += cf_amount * weight

        denom = start_value + weighted_cf

        if abs(denom) > 1e-9:
            return_pct = return_dkk / denom * 100.0
        else:
            return_pct = 0.0

        period_start = sub.index[0]
        period_end = sub.index[-1]

        rows.append({
            "period_start": period_start,
            "period_end": period_end,
            "period_label": _period_label(period_key, period_end),
            "return_dkk": return_dkk,
            "return_pct": return_pct,
        })

    out = pd.DataFrame(rows)

    if out.empty:
        return pd.DataFrame(columns=[
            "period_start", "period_end", "period_label",
            "return_dkk", "return_pct"
        ])

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["return_dkk", "return_pct"])
    return out.reset_index(drop=True)


def compute_grouped_portfolio_return_dynamics(
    orders: pd.DataFrame,
    prices: pd.DataFrame,
    usd_dkk: pd.Series,
    eur_dkk: pd.Series,
    total_value: pd.Series,
    cashflows: pd.Series,
    cashflow_fracs=None,
    group_map: dict | None = None,
    grouping: str = "Månedlig",
    residual_group_name: str = "Kontanter/øvrigt",
) -> pd.DataFrame:
    """
    Beregner grupperet periodisk bidrag til porteføljeafkast.

    Bruges til grupperinger som:
    - Aktivklasser
    - Sektorer

    Returnerer long-format DataFrame med:
    - period_start
    - period_end
    - period_label
    - group
    - return_dkk
    - return_pct

    Vigtig beregningslogik:
    For hver gruppe beregnes periodebidrag som:

        slutværdi - startværdi - netto kapital ind i gruppen

    hvor netto kapital ind i gruppen er:
        BUY = positiv kapital ind
        SELL = negativ kapital ind

    Det betyder, at løbende køb/salg ikke fejlagtigt bliver vist som afkast.

    Procentvisning er %-point bidrag til porteføljeafkastet:
        gruppe_return_dkk / porteføljens Modified Dietz-lignende kapitalbase

    Funktionen er 100% Streamlit-fri og hører derfor hjemme i analytics-laget.
    """

    empty_cols = [
        "period_start", "period_end", "period_label",
        "group", "return_dkk", "return_pct"
    ]

    if (
        orders is None or orders.empty
        or prices is None or prices.empty
        or total_value is None or len(total_value) < 2
        or group_map is None or not group_map
    ):
        return pd.DataFrame(columns=empty_cols)

    grouping_map = {
        "Daglig": "D",
        "Ugentlig": "W",
        "Månedlig": "M",
        "Årlig": "Y",
        "Daily": "D",
        "Weekly": "W",
        "Monthly": "M",
        "Yearly": "Y",
        "D": "D",
        "W": "W",
        "M": "M",
        "Y": "Y",
    }

    freq = grouping_map.get(grouping, "M")

    values = total_value.copy()
    values.index = pd.to_datetime(values.index)
    values = values.sort_index().astype(float).replace([np.inf, -np.inf], np.nan).dropna()

    if len(values) < 2:
        return pd.DataFrame(columns=empty_cols)

    date_range = values.index

    cf_total = cashflows.copy() if cashflows is not None else pd.Series(0.0, index=date_range)
    cf_total.index = pd.to_datetime(cf_total.index)
    cf_total = cf_total.reindex(date_range).fillna(0.0).astype(float)

    if cashflow_fracs is None:
        fracs_total = pd.Series(0.0, index=date_range)
    else:
        fracs_total = cashflow_fracs.copy()
        fracs_total.index = pd.to_datetime(fracs_total.index)
        fracs_total = fracs_total.reindex(date_range).fillna(0.0).astype(float)

    orders_work = orders.copy()

    if "TradeDate" not in orders_work.columns:
        orders_work["TradeDate"] = pd.to_datetime(
            orders_work["Date"],
            dayfirst=True,
            errors="coerce",
        ).dt.normalize()
    else:
        orders_work["TradeDate"] = pd.to_datetime(
            orders_work["TradeDate"],
            errors="coerce",
        ).dt.normalize()

    orders_work = orders_work.dropna(subset=["TradeDate", "Ticker"]).copy()

    if orders_work.empty:
        return pd.DataFrame(columns=empty_cols)

    orders_work["Qty_Adj"] = np.where(
        orders_work["Side"] == "BUY",
        orders_work["Quantity"],
        -orders_work["Quantity"],
    )

    # BUY = kapital ind i gruppen, SELL = kapital ud af gruppen.
    orders_work["Capital_Flow_DKK"] = np.where(
        orders_work["Side"] == "BUY",
        orders_work["Notional, DKK"],
        -orders_work["Notional, DKK"],
    ).astype(float)

    tickers = sorted([
        t for t in orders_work["Ticker"].dropna().unique().tolist()
        if t in group_map
    ])

    if not tickers:
        return pd.DataFrame(columns=empty_cols)

    # Holdings pr. ticker pr. dag.
    holdings = pd.DataFrame(0.0, index=date_range, columns=tickers)

    for t in tickers:
        sub = orders_work[orders_work["Ticker"] == t]
        daily_change = sub.groupby("TradeDate")["Qty_Adj"].sum()
        holdings[t] = (
            daily_change
            .cumsum()
            .reindex(date_range, method="ffill")
            .fillna(0.0)
        )

    px = prices.copy()
    px.index = pd.to_datetime(px.index)
    px = px.reindex(date_range).ffill().bfill()

    usd_series = usd_dkk.copy() if usd_dkk is not None else pd.Series(6.85, index=date_range)
    usd_series.index = pd.to_datetime(usd_series.index)
    usd_series = usd_series.reindex(date_range).ffill().bfill().fillna(6.85)

    eur_series = eur_dkk.copy() if eur_dkk is not None else pd.Series(7.46, index=date_range)
    eur_series.index = pd.to_datetime(eur_series.index)
    eur_series = eur_series.reindex(date_range).ffill().bfill().fillna(7.46)

    asset_ccy = (
        orders_work
        .drop_duplicates("Ticker")
        .set_index("Ticker")["Asset currency"]
        .to_dict()
    )

    # Daglig værdi pr. ticker i DKK.
    ticker_values = pd.DataFrame(0.0, index=date_range, columns=tickers)

    for t in tickers:
        if t not in px.columns:
            continue

        price_t = px[t].ffill().bfill().fillna(0.0)
        ccy = asset_ccy.get(t, "USD")

        if ccy == "USD":
            rate = usd_series
        elif ccy == "EUR":
            rate = eur_series
        else:
            rate = pd.Series(1.0, index=date_range)

        ticker_values[t] = holdings[t] * price_t * rate

    # Sum ticker-værdier pr. gruppe.
    groups = sorted(set(group_map.get(t, "Andet") for t in tickers))
    group_values = pd.DataFrame(0.0, index=date_range, columns=groups)

    for t in tickers:
        g = group_map.get(t, "Andet")
        if g not in group_values.columns:
            group_values[g] = 0.0
        group_values[g] = group_values[g] + ticker_values[t]

    # Kapitalflow pr. gruppe pr. dag.
    group_capital_flows = pd.DataFrame(0.0, index=date_range, columns=group_values.columns)

    for t in tickers:
        g = group_map.get(t, "Andet")
        sub = orders_work[orders_work["Ticker"] == t]
        daily_flow = sub.groupby("TradeDate")["Capital_Flow_DKK"].sum()
        group_capital_flows[g] = (
            group_capital_flows[g]
            + daily_flow.reindex(date_range).fillna(0.0)
        )

    # Perioder.
    period_df = pd.DataFrame(index=date_range)
    period_df["value"] = values
    period_df["cashflow"] = cf_total
    period_df["frac"] = fracs_total
    period_df["row_no"] = np.arange(len(period_df))

    if freq == "D":
        period_df["period"] = period_df.index.to_period("D")
    elif freq == "W":
        period_df["period"] = period_df.index.to_period("W-SUN")
    elif freq == "Y":
        period_df["period"] = period_df.index.to_period("Y")
    else:
        period_df["period"] = period_df.index.to_period("M")

    month_names_da = {
        1: "jan", 2: "feb", 3: "mar", 4: "apr",
        5: "maj", 6: "jun", 7: "jul", 8: "aug",
        9: "sep", 10: "okt", 11: "nov", 12: "dec",
    }

    def _period_label(period_end):
        if freq == "D":
            return period_end.strftime("%d-%m-%Y")

        if freq == "W":
            iso = period_end.isocalendar()
            return f"uge {int(iso.week)} {int(iso.year)}"

        if freq == "Y":
            return str(period_end.year)

        return f"{month_names_da.get(period_end.month, period_end.strftime('%b'))} {period_end.year}"

    def _period_denominator(start_row, first_row, last_row, sub_period):
        start_ts = period_df.index[start_row]
        end_ts = period_df.index[last_row]
        start_value = float(period_df["value"].iloc[start_row])

        if end_ts <= start_ts:
            return start_value

        if first_row > 0:
            cf_sub = sub_period
        else:
            cf_sub = sub_period.iloc[1:]

        period_days = max((end_ts - start_ts).total_seconds() / 86400.0, 1e-9)

        weighted_cf = 0.0

        for cf_date, cf_row in cf_sub.iterrows():
            cf_amount = float(cf_row["cashflow"])
            cf_frac = float(cf_row["frac"])
            cf_ts = cf_date + pd.to_timedelta(cf_frac, unit="D")

            weight = (end_ts - cf_ts).total_seconds() / 86400.0 / period_days
            weight = min(max(weight, 0.0), 1.0)

            weighted_cf += cf_amount * weight

        return start_value + weighted_cf

    rows = []

    for _period_key, sub_period in period_df.groupby("period", sort=True):
        if sub_period.empty:
            continue

        first_row = int(sub_period["row_no"].iloc[0])
        last_row = int(sub_period["row_no"].iloc[-1])

        if first_row > 0:
            start_row = first_row - 1
            cf_slice_start_row = first_row
        else:
            start_row = first_row
            cf_slice_start_row = first_row + 1

        start_ts = period_df.index[start_row]
        end_ts = period_df.index[last_row]

        if end_ts <= start_ts:
            continue

        period_start = sub_period.index[0]
        period_end = sub_period.index[-1]
        label = _period_label(period_end)

        denom = _period_denominator(start_row, first_row, last_row, sub_period)

        group_returns_this_period = {}

        for g in group_values.columns:
            start_val_g = float(group_values[g].iloc[start_row])
            end_val_g = float(group_values[g].iloc[last_row])

            if cf_slice_start_row <= last_row:
                flow_g = float(group_capital_flows[g].iloc[cf_slice_start_row:last_row + 1].sum())
            else:
                flow_g = 0.0

            ret_g = end_val_g - start_val_g - flow_g
            group_returns_this_period[g] = ret_g

        # Total periodeafkast efter samme overordnede princip.
        if first_row > 0:
            cf_total_sub = sub_period["cashflow"]
        else:
            cf_total_sub = sub_period["cashflow"].iloc[1:]

        total_return_dkk = (
            float(period_df["value"].iloc[last_row])
            - float(period_df["value"].iloc[start_row])
            - float(cf_total_sub.sum())
        )

        stock_group_sum = float(sum(group_returns_this_period.values()))
        residual_return = total_return_dkk - stock_group_sum

        # Residual bruges til kontanter/øvrigt, så summeringen passer med totalporteføljen.
        if abs(residual_return) > 1e-9 or residual_group_name:
            group_returns_this_period[residual_group_name] = (
                group_returns_this_period.get(residual_group_name, 0.0)
                + residual_return
            )

        for g, ret_g in group_returns_this_period.items():
            if abs(ret_g) < 1e-9:
                continue

            ret_pct = (ret_g / denom * 100.0) if abs(denom) > 1e-9 else 0.0

            rows.append({
                "period_start": period_start,
                "period_end": period_end,
                "period_label": label,
                "group": g,
                "return_dkk": ret_g,
                "return_pct": ret_pct,
            })

    out = pd.DataFrame(rows)

    if out.empty:
        return pd.DataFrame(columns=empty_cols)

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["return_dkk", "return_pct"])
    return out.reset_index(drop=True)


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
    
def compute_ticker_lifecycle(
    orders_df: pd.DataFrame,
    prices: pd.DataFrame,
    live_quotes: dict,
    usd_dkk: float,
    eur_dkk: float,
    positions_df=None,
) -> list[dict]:
    """
    Beregn livscyklus (køb→salg, afkast) for hver enkelt ticker.
    Returnerer sorteret liste: aktive øverst (efter afkast%), lukkede nederst (efter salgsdato).
    """
    if orders_df.empty:
        return []

    lifecycle_list = []

    for ticker in orders_df["Ticker"].unique():
        sub = orders_df[orders_df["Ticker"] == ticker].copy()

        # Grundoplysninger
        name = sub["Name"].iloc[0] if not sub.empty else ticker
        asset_ccy = sub["Asset currency"].iloc[0] if not sub.empty else "USD"
        fx_rate = usd_dkk if asset_ccy == "USD" else (eur_dkk if asset_ccy == "EUR" else 1.0)
        
        # Kvantitet og beløb
        sub["Qty_Adj"] = np.where(sub["Side"] == "BUY", sub["Quantity"], -sub["Quantity"])
        total_qty = float(sub["Qty_Adj"].sum())
        is_active = total_qty > 0.001
        
        # Hent præcis GAK fra positions_df (kun aktive positioner)
        gak_valuta = None
        if is_active and positions_df is not None and not positions_df.empty:
            pos_row = positions_df[positions_df["Ticker"] == ticker]
            if not pos_row.empty:
                gak_valuta = _safe_float(
                    pos_row["Average entry price (asset currency)"].iloc[0]
                )
        
        # Datoer
        sub_sorted = sub.sort_values("Date")
        first_buy = sub_sorted["Date"].iloc[0]
        last_activity = sub_sorted["Date"].iloc[-1]

        last_sale = None
        sells = sub[sub["Side"] == "SELL"].sort_values("Date")
        if not sells.empty:
            last_sale = sells["Date"].iloc[-1]

        # Investeret og realiseret
        buys_dkk  = float(sub[sub["Side"] == "BUY"]["Notional, DKK"].sum())
        sells_dkk = float(sub[sub["Side"] == "SELL"]["Notional, DKK"].sum())

        # Nuværende værdi (kun hvis aktiv)
        current_value_dkk = 0.0
        if is_active:
            live_price = live_quotes.get(ticker, {}).get("live")
            if live_price is None and ticker in prices.columns:
                price_series = prices[ticker].dropna()
                live_price = float(price_series.iloc[-1]) if len(price_series) else None
            if live_price is not None:
                current_value_dkk = total_qty * live_price * fx_rate

        # Kostbasis for resterende beholdning — Plutos 'Amount, DKK'.
        # Korrekt ved delvist salg; fald tilbage til nettopengestrøm.
        cost_basis_dkk = 0.0
        if is_active:
            if (positions_df is not None and not positions_df.empty
                    and "Ticker" in positions_df.columns
                    and "Amount, DKK" in positions_df.columns):
                pos_row = positions_df[positions_df["Ticker"] == ticker]
                if not pos_row.empty:
                    cb = _safe_float(pos_row["Amount, DKK"].iloc[0])
                    if cb is not None:
                        cost_basis_dkk = cb
            if cost_basis_dkk == 0.0:
                cost_basis_dkk = buys_dkk - sells_dkk

        # Afkast delt op: realiseret (solgte) + urealiseret (beholdning).
        # realiseret + urealiseret == total_return_dkk pr. konstruktion.
        cost_of_sold_dkk = buys_dkk - cost_basis_dkk
        realized_gain_dkk = (sells_dkk - cost_of_sold_dkk) if sells_dkk > 0 else 0.0
        realized_pct = (realized_gain_dkk / cost_of_sold_dkk * 100) if cost_of_sold_dkk > 0 else 0.0
        unrealized_dkk = (current_value_dkk - cost_basis_dkk) if is_active else 0.0
        unrealized_pct = (unrealized_dkk / cost_basis_dkk * 100) if cost_basis_dkk > 0 else 0.0

        total_return_dkk = current_value_dkk + sells_dkk - buys_dkk
        total_return_pct = (total_return_dkk / buys_dkk * 100) if buys_dkk > 0 else 0.0

        lifecycle_list.append({
            "ticker":            ticker,
            "name":              name,
            "asset_ccy":         asset_ccy,
            "is_active":         is_active,
            "first_buy":         first_buy,
            "last_activity":     last_activity,
            "last_sale":         last_sale,
            "invested_dkk":      buys_dkk,
            "realized_dkk":      sells_dkk,
            "current_value_dkk": current_value_dkk,
            "cost_basis_dkk":    cost_basis_dkk,
            "total_qty":         total_qty,
            "realized_gain_dkk": realized_gain_dkk,
            "realized_pct":      realized_pct,
            "unrealized_dkk":    unrealized_dkk,
            "unrealized_pct":    unrealized_pct,
            "total_return_dkk":  total_return_dkk,
            "total_return_pct":  total_return_pct,
            "orders":            sub,
            "gak_valuta":        gak_valuta,
        })

    # Sortering: aktive øverst (afkast% DESC), lukkede nederst (last_activity DESC)
    active = sorted([x for x in lifecycle_list if     x["is_active"]], key=lambda x: x["total_return_pct"], reverse=True)
    closed = sorted([x for x in lifecycle_list if not x["is_active"]], key=lambda x: x["last_activity"],    reverse=True)

    return active + closed
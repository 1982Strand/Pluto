"""
analytics/comparison.py — Kerneberegninger til Sammenligning-fanen.

Ren Python: ingen Streamlit-kald. Cachede datakald går via data.cached
(samme mønster som analytics/portfolio.py).

Asymmetri: porteføljens afkastkurve er en Modified Dietz-TWR (cumulative_return_series),
mens benchmarks er simple prisserier der rebaseres. For at kunne plotte alle serier på
samme akse behandles TWR-kurven som en niveau-serie (1 + pct/100), hvorefter både
portefølje og benchmarks rebaseres ens fra et fælles startpunkt.
"""
import numpy as np
import pandas as pd

from data import cached
from analytics.portfolio import slice_period, cumulative_return_series


# Benchmark-/indeks-tickers hvor yfinance ikke altid oplyser valuta pålideligt.
# Brugerindtastede tickers slås i stedet op via fetch_ticker_meta.
_KNOWN_CURRENCIES = {
    "^GSPC": "USD", "^NDX": "USD", "^IXIC": "USD", "^DJI": "USD",
    "^OMXC25": "DKK", "^STOXX": "EUR", "^STOXX50E": "EUR", "^GDAXI": "EUR",
    "IWDA.AS": "EUR", "GC=F": "USD", "BTC-USD": "USD", "SPY": "USD", "QQQ": "USD",
}


def ticker_currency(ticker):
    """Bestemmer en tickers handelsvaluta. Kendte benchmarks slås op direkte
    (hurtigt); øvrige hentes via fetch_ticker_meta (cached 24t)."""
    if ticker in _KNOWN_CURRENCIES:
        return _KNOWN_CURRENCIES[ticker]
    try:
        meta = cached.fetch_ticker_meta((ticker,))
        return (meta.get(ticker, {}).get("currency") or "USD").upper()
    except Exception:
        return "USD"


def ticker_name(ticker):
    """Firma-/fondsnavn for en ticker via fetch_ticker_meta (cached 24t).
    Tom streng hvis ukendt."""
    try:
        meta = cached.fetch_ticker_meta((ticker,))
        return meta.get(ticker, {}).get("name") or ""
    except Exception:
        return ""


def rebase_series(series, base=None):
    """Rebaserer en niveau-/prisserie til 0 fra første punkt (eller given base).
    Output i procent ×100 (fx 18.4 = +18,4 %) — samme enhed som porteføljens TWR
    fra cumulative_return_series, så serier kan plottes på samme akse."""
    series = series.dropna()
    if series.empty:
        return series
    if base is None:
        base = series.iloc[0]
    if base is None or pd.isna(base) or base == 0:
        return pd.Series(0.0, index=series.index)
    return (series / base - 1.0) * 100.0


def compute_portfolio_twr_series(total_value, cashflows, cashflow_fracs, period):
    """Porteføljens TWR-procentkurve for perioden (procent ×100), rebaseret til
    periodens start.

    Slicer med slice_period og kører resultatet gennem cumulative_return_series —
    samme mønster som views/portfolio_overview.py. total_value er en DKK-
    kroneværdiserie (IKKE afkast); afkastet beregnes her."""
    v_per, cf_per, d_per = slice_period(total_value, cashflows, period)
    fracs = (
        cashflow_fracs.reindex(d_per).values
        if cashflow_fracs is not None else None
    )
    _, ret_pct = cumulative_return_series(
        v_per.values, cf_per.values, d_per, cashflow_fracs=fracs
    )
    return ret_pct


def compute_benchmark_series(ticker, start, end, currency=None):
    """Henter daglig prishistorik for ticker og konverterer til DKK pr. dag.

    fetch_price_history tilføjer selv USDDKK=X og EURDKK=X som kolonner, så
    valutakonverteringen bruger dagens kurs (en konstant kurs ville ophæves ved
    rebasering). Returnerer en tom serie hvis tickeren ikke findes."""
    df = cached.fetch_price_history((ticker,), str(start), str(end))
    if df is None or len(df) == 0 or ticker not in getattr(df, "columns", []):
        return pd.Series(dtype=float)
    price = pd.to_numeric(df[ticker], errors="coerce").dropna()
    if price.empty:
        return pd.Series(dtype=float)
    if currency is None:
        currency = ticker_currency(ticker)
    if currency in ("USD", "EUR"):
        fx_col = "USDDKK=X" if currency == "USD" else "EURDKK=X"
        if fx_col in df.columns:
            fx = (
                pd.to_numeric(df[fx_col], errors="coerce")
                .reindex(price.index).ffill().bfill()
            )
            price = price * fx
    return price.dropna()


def slice_price_to_period(series, period):
    """Slicer en enkelt prisserie til den valgte periode. Genbruger slice_period
    fra analytics/portfolio.py med en dummy-cashflow-serie."""
    if series is None or series.empty:
        return series
    dummy_cf = pd.Series(0.0, index=series.index)
    v, _, _ = slice_period(series, dummy_cf, period)
    return v


def align_to_common_index(series_dict):
    """Reindexer alle serier til ét fælles dagligt indeks (skæringsmængden af
    deres datointervaller) med forward fill.

    Kørt FØR rebasering, så alle serier starter på nøjagtig samme dato — løser
    forskydninger fra forskellige markeders helligdage."""
    valid = {
        k: v.dropna()
        for k, v in series_dict.items()
        if v is not None and not v.dropna().empty
    }
    if not valid:
        return {}
    start = max(s.index[0] for s in valid.values())
    end = min(s.index[-1] for s in valid.values())
    if start > end:
        return {}
    common = pd.date_range(start, end, freq="D")
    return {k: v.reindex(common, method="ffill") for k, v in valid.items()}


def compute_comparison_stats(series_dict):
    """series_dict: {label: rebaseret pd.Series (procent ×100)}.

    Returnerer DataFrame indekseret efter label med kolonnerne total_return,
    annualized_return, max_drawdown, volatility, best_month, worst_month.
    Drawdown og CAGR beregnes på niveau-serien (1 + pct/100), ikke på den
    0-centrerede pct-serie."""
    rows = {}
    for label, s in series_dict.items():
        s = s.dropna()
        if s.empty:
            continue
        level = 1.0 + s / 100.0
        total_return = float(s.iloc[-1])

        n_days = (s.index[-1] - s.index[0]).days
        years = n_days / 365.25 if n_days > 0 else 0.0
        last_level = float(level.iloc[-1])
        # CAGR er først meningsfuld for perioder over ~6 måneder — på kortere
        # vinduer eksploderer den (fx +100 % over 6 uger -> titusinder af %).
        if years >= 0.5 and last_level > 0:
            annualized = (last_level ** (1.0 / years) - 1.0) * 100.0
        else:
            annualized = np.nan

        running_max = level.cummax()
        drawdown = (level / running_max - 1.0) * 100.0
        max_dd = float(drawdown.min()) if len(drawdown) else np.nan

        daily_ret = level.pct_change().dropna()
        if len(daily_ret) > 1:
            volatility = float(daily_ret.std() * np.sqrt(252) * 100.0)
        else:
            volatility = np.nan

        monthly_ret = level.resample("ME").last().pct_change().dropna() * 100.0
        best_month = float(monthly_ret.max()) if len(monthly_ret) else np.nan
        worst_month = float(monthly_ret.min()) if len(monthly_ret) else np.nan

        rows[label] = {
            "total_return": total_return,
            "annualized_return": annualized,
            "max_drawdown": max_dd,
            "volatility": volatility,
            "best_month": best_month,
            "worst_month": worst_month,
        }
    return pd.DataFrame(rows).T


def compute_relative_series(series_dict, baseline_label):
    """Overskudsafkast relativt til baseline_label. Input og output er
    rebaserede pct-serier; baseline-serien bliver til en flad 0-linje."""
    baseline = series_dict.get(baseline_label)
    if baseline is None:
        return series_dict
    return {
        label: s - baseline.reindex(s.index)
        for label, s in series_dict.items()
    }

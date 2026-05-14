from utils.formatting import _safe_float
import pandas as pd
import yfinance as yf

def fetch_price_history(tickers_tuple, start_str, end_str):
    """Henter daglige slutkurser for tickere + USDDKK + EURDKK. Cached i 30 min."""
    symbols = list(tickers_tuple) + ["USDDKK=X", "EURDKK=X"]
    data = yf.download(
        symbols, start=start_str, end=end_str,
        progress=False, auto_adjust=True, group_by="column"
    )
    if isinstance(data.columns, pd.MultiIndex):
        data = data["Close"]
    elif "Close" in data.columns:
        data = data[["Close"]]
        data.columns = symbols
    return data


def fetch_price_history_intraday(tickers_tuple, period, interval):
    """Henter intraday Close-priser for tickere + USDDKK + EURDKK."""
    symbols = list(tickers_tuple) + ["USDDKK=X", "EURDKK=X"]
    data = yf.download(
        symbols,
        period=period,
        interval=interval,
        progress=False,
        auto_adjust=True,
        group_by="column",
        prepost=True,
        threads=True,
    )
    if data is None or len(getattr(data, "columns", [])) == 0:
        return pd.DataFrame()
    if isinstance(data.columns, pd.MultiIndex):
        # MultiIndex: (PriceField, Symbol)
        if "Close" in data.columns.get_level_values(0):
            data = data["Close"]
        else:
            return pd.DataFrame()
    elif "Close" in data.columns:
        data = data[["Close"]]
        data.columns = symbols
    return data
    
       
def fetch_live_quotes(tickers_tuple):
    """
    Henter aktuel pris + de seneste regular-session slutkurser via history.

    Kilder:
      - Daglig history (prepost=False) -> prev_close, prev_prev_close, today_close
      - 1-min history (prepost=True) -> seneste tilgængelige pris (live)

    Returnerede felter pr. ticker:
      - live: seneste 1-min tick (kan være regular/pre/after-hours)
      - prev_close: forrige handelsdags regular close (= Yahoo's 'Previous Close')
      - prev_prev_close: regular close to handelsdage før
      - today_close: dagens regular close (kun sat hvis dagens session ER
        afsluttet — dvs. efter 16:00 ET på en hverdag, op til markedsåbning
        næste handelsdag). Bruges til Yahoo-stil dual-pris-visning hvor
        venstre = dagens close, højre = efter-/pre-market live.
    """
    quotes = {}

    # Er vi i regular session lige nu? (bruges til at afgøre om dagens
    # daily-bar er ufuldstændig og skal droppes/ikke gemmes som today_close)
    et_now = pd.Timestamp.now(tz="America/New_York")
    et_min = et_now.hour * 60 + et_now.minute
    in_regular_now = (
        et_now.weekday() < 5
        and 9 * 60 + 30 <= et_min < 16 * 60
    )

    for t in tickers_tuple:
        prev = None
        prev_prev = None
        today_close = None
        live = None

        try:
            tk = yf.Ticker(t)

            hist_d = tk.history(period="10d", interval="1d", prepost=False, auto_adjust=False)
            if not hist_d.empty and "Close" in hist_d.columns:
                try:
                    et_today = et_now.date()
                    last_ts = pd.Timestamp(hist_d.index[-1])
                    last_date_et = (
                        last_ts.tz_convert("America/New_York").date()
                        if last_ts.tz is not None else last_ts.date()
                    )
                    # Tre cases for dagens bar:
                    #  1) I regular session: dagens bar er ufuldstændig (close
                    #     er bare den seneste intraday-pris) → drop fra prev,
                    #     ingen today_close.
                    #  2) Efter regular close samme dag: dagens bar er den
                    #     komplette regular close → gem som today_close, drop
                    #     fra prev så prev = gårsdagens close.
                    #  3) Sidste bar er en tidligere dag: prev = den, ingen
                    #     today_close.
                    if last_date_et == et_today:
                        if in_regular_now:
                            hist_d = hist_d.iloc[:-1]
                        else:
                            today_close = float(hist_d["Close"].iloc[-1])
                            hist_d = hist_d.iloc[:-1]
                    elif last_date_et > et_today:
                        # Defensiv: drop fremtidige bars (bør ikke ske)
                        hist_d = hist_d.iloc[:-1]
                except Exception:
                    pass
                closes = hist_d["Close"].dropna()
                if len(closes) >= 1:
                    prev = float(closes.iloc[-1])
                if len(closes) >= 2:
                    prev_prev = float(closes.iloc[-2])

            # Aktuel pris (inkl. pre-/after-market) som seneste minutbar
            hist_m = tk.history(period="2d", interval="1m", prepost=True, auto_adjust=False)
            if not hist_m.empty and "Close" in hist_m.columns:
                closes_m = hist_m["Close"].dropna()
                if len(closes_m) >= 1:
                    live = float(closes_m.iloc[-1])

        except Exception:
            pass

        quotes[t] = {
            "live": _safe_float(live),
            "prev_close": _safe_float(prev),
            "prev_prev_close": _safe_float(prev_prev),
            "today_close": _safe_float(today_close),
        }

    return quotes
    

def fetch_live_fx_rates():
    """
    Henter live USDDKK og EURDKK via 1-min minute-bars (prepost=True så vi får
    seneste tilgængelige tick også udenfor regular hours).

    Forex handles 24/5 så på hverdage vil minute-bars næsten altid være tilgængelige.
    På weekenden kan minute-bars mangle — vi returnerer None og falder tilbage til
    daglig close i kalden-koden.

    Returnerer dict[str -> float|None]: {"USDDKK": ..., "EURDKK": ...}
    """
    rates = {"USDDKK": None, "EURDKK": None}
    for sym, key in [("USDDKK=X", "USDDKK"), ("EURDKK=X", "EURDKK")]:
        try:
            tk = yf.Ticker(sym)
            hist = tk.history(period="2d", interval="1m", prepost=True, auto_adjust=False)
            if not hist.empty and "Close" in hist.columns:
                closes = hist["Close"].dropna()
                if len(closes) >= 1:
                    rates[key] = float(closes.iloc[-1])
        except Exception:
            pass
    return rates


_SECTOR_DA = {
    "Technology": "Information Technology",
    "Healthcare": "Sundhedspleje",
    "Communication Services": "Kommunikation",
    "Consumer Cyclical": "Forbrugsgoder",
    "Consumer Defensive": "Stabile forbrugsvarer",
    "Financial Services": "Finans",
    "Industrials": "Industri",
    "Energy": "Energi",
    "Basic Materials": "Materialer",
    "Real Estate": "Ejendomme",
    "Utilities": "Forsyning",
}


_REGION_MAP = {
    "United States": "Nordamerika", "Canada": "Nordamerika", "Mexico": "Nordamerika",
    "United Kingdom": "Europa", "Germany": "Europa", "France": "Europa",
    "Switzerland": "Europa", "Denmark": "Europa", "Sweden": "Europa",
    "Norway": "Europa", "Netherlands": "Europa", "Italy": "Europa",
    "Spain": "Europa", "Ireland": "Europa", "Finland": "Europa",
    "Belgium": "Europa", "Austria": "Europa", "Portugal": "Europa",
    "Luxembourg": "Europa", "Poland": "Europa", "Czech Republic": "Europa",
    "Japan": "Asien", "China": "Asien", "Hong Kong": "Asien",
    "South Korea": "Asien", "Singapore": "Asien", "Taiwan": "Asien",
    "India": "Asien", "Indonesia": "Asien", "Thailand": "Asien", "Vietnam": "Asien",
    "Australia": "Oceanien", "New Zealand": "Oceanien",
    "Brazil": "Sydamerika", "Argentina": "Sydamerika", "Chile": "Sydamerika",
    "South Africa": "Afrika", "Egypt": "Afrika",
    "Israel": "Mellemøsten", "United Arab Emirates": "Mellemøsten",
    "Saudi Arabia": "Mellemøsten", "Turkey": "Mellemøsten",
}


def fetch_ticker_meta(tickers_tuple):
    """Returnerer dict[ticker -> {"sector", "asset_class", "country", "region"}]. Cached 24t.

    asset_class er én af: "Aktier", "Fonde / ETF'er", "Krypto", "Andet".
    sector er den danske sektor-label (eller "Fonde" / "Krypto" / "Andet").
    country er yfinance's country-streng (fx "United States") eller "Andet".
    region er udledt fra country via _REGION_MAP.
    """
    out = {}
    for t in tickers_tuple:
        sector = "Andet"
        asset_class = "Andet"
        country = "Andet"
        try:
            info = yf.Ticker(t).info or {}
            qt = (info.get("quoteType") or "").upper()
            yf_sector = info.get("sector") or ""
            country_raw = info.get("country") or ""
            if country_raw:
                country = country_raw
            if qt in ("ETF", "MUTUALFUND"):
                sector = "Fonde"
                asset_class = "Fonde / ETF'er"
            elif qt == "CRYPTOCURRENCY":
                sector = "Krypto"
                asset_class = "Krypto"
            elif qt == "EQUITY":
                sector = _SECTOR_DA.get(yf_sector, yf_sector or "Andet")
                asset_class = "Aktier"
            else:
                sector = _SECTOR_DA.get(yf_sector, yf_sector or "Andet")
        except Exception:
            pass
        region = _REGION_MAP.get(country, "Andet")
        out[t] = {
            "sector": sector,
            "asset_class": asset_class,
            "country": country,
            "region": region,
        }
    return out
    
def fetch_ticker_quote_info(ticker):
    """yfinance Ticker.info-snapshot — kun de felter detalje-siden bruger.

    Cached i 1 time. Returnerer dict med nøgler:
    volume, average_volume, day_low, day_high, open, fifty_two_week_low,
    fifty_two_week_high, market_cap, trailing_pe, trailing_eps, beta,
    exchange, isin, long_name, website.
    """
    out = {
        "volume": None, "average_volume": None,
        "day_low": None, "day_high": None, "open": None,
        "fifty_two_week_low": None, "fifty_two_week_high": None,
        "market_cap": None, "trailing_pe": None, "trailing_eps": None,
        "beta": None, "exchange": None, "isin": None,
        "long_name": None, "website": None,
    }
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        return out
    out["volume"] = _safe_float(info.get("volume") or info.get("regularMarketVolume"))
    out["average_volume"] = _safe_float(
        info.get("averageVolume") or info.get("averageVolume10days")
    )
    out["day_low"] = _safe_float(info.get("dayLow") or info.get("regularMarketDayLow"))
    out["day_high"] = _safe_float(info.get("dayHigh") or info.get("regularMarketDayHigh"))
    out["open"] = _safe_float(info.get("open") or info.get("regularMarketOpen"))
    out["fifty_two_week_low"] = _safe_float(info.get("fiftyTwoWeekLow"))
    out["fifty_two_week_high"] = _safe_float(info.get("fiftyTwoWeekHigh"))
    out["market_cap"] = _safe_float(info.get("marketCap"))
    out["trailing_pe"] = _safe_float(info.get("trailingPE"))
    out["trailing_eps"] = _safe_float(info.get("trailingEps"))
    out["beta"] = _safe_float(info.get("beta"))
    out["exchange"] = info.get("fullExchangeName") or info.get("exchange") or None
    out["isin"] = info.get("isin") or None
    out["long_name"] = info.get("longName") or info.get("shortName") or None
    out["website"] = info.get("website") or None
    return out
    
def fetch_intraday_sparklines(tickers_tuple):
    """For hver ticker: liste af 5-min closes for seneste regular session
    (09:30 - 16:00 ET) — afsluttet eller igangværende. Pre/after-hours
    filtreres væk så sparklinen matcher 'Sidste luk' som er regular-
    session-close. I pre-market falder target tilbage til gårsdagens
    session, da dagens dato endnu ikke har regular-hours-bars."""
    if not tickers_tuple:
        return {}
    out = {t: [] for t in tickers_tuple}
    try:
        prices_i = fetch_price_history_intraday(tickers_tuple, "5d", "5m")
    except Exception:
        return out
    if prices_i is None or prices_i.empty:
        return out
    idx = pd.DatetimeIndex(prices_i.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx_et = idx.tz_convert("America/New_York")
    et_dates = idx_et.normalize()
    # Find seneste ET-dato der HAR regular-hours-bars. I pre-market vil
    # dagens dato ikke have nogen regular-bars endnu, så target falder
    # naturligt tilbage til seneste afsluttede handelsdag (typisk gårsdag,
    # eller fredag hvis vi er mandag-morgen).
    et_minutes = idx_et.hour * 60 + idx_et.minute
    regular_mask = (et_minutes >= 9 * 60 + 30) & (et_minutes < 16 * 60)
    weekday_mask = et_dates.weekday < 5
    candidate_dates = et_dates[regular_mask & weekday_mask]
    if len(candidate_dates) == 0:
        return out
    target = candidate_dates.max()
    mask = (et_dates == target) & regular_mask
    sliced = prices_i.loc[mask]
    for t in tickers_tuple:
        if t in sliced.columns:
            series = sliced[t].dropna()
            if len(series) >= 2:
                out[t] = [float(v) for v in series.tolist()]
    return out
    
# ==========================================================
# Asset detail helpers (REN Python, ingen Streamlit)
# ==========================================================
import pandas as pd
import yfinance as yf

def fetch_asset_history(ticker: str, period_key: str, include_extended: bool = False):
    """
    Hent pris- og volumen-historik for én ticker til asset_detail.
    Returnerer (prices_series, volume_series) med datetime-indeks (yfinance default tz).
    """
    try:
        tk = yf.Ticker(ticker)

        if period_key == "1D":
            hist = tk.history(period="2d", interval="1m", prepost=True, auto_adjust=False)
        elif period_key == "1U":
            hist = tk.history(period="5d", interval="5m", prepost=True, auto_adjust=False)
        elif period_key == "1M":
            hist = tk.history(period="1mo", interval="1h", prepost=include_extended, auto_adjust=False)
        elif period_key == "3M":
            hist = tk.history(period="3mo", interval="1d", prepost=False, auto_adjust=False)
        elif period_key == "6M":
            hist = tk.history(period="6mo", interval="1d", prepost=False, auto_adjust=False)
        elif period_key == "YTD":
            hist = tk.history(period="ytd", interval="1d", prepost=False, auto_adjust=False)
        elif period_key == "1Å":
            hist = tk.history(period="1y", interval="1d", prepost=False, auto_adjust=False)
        elif period_key == "5Å":
            hist = tk.history(period="5y", interval="1wk", prepost=False, auto_adjust=False)
        else:  # "Maks"
            hist = tk.history(period="max", interval="1mo", prepost=False, auto_adjust=False)
    except Exception:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    if hist is None or hist.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float)

    prices = hist["Close"].dropna() if "Close" in hist.columns else pd.Series(dtype=float)
    volumes = hist["Volume"].dropna() if "Volume" in hist.columns else pd.Series(dtype=float)
    return prices, volumes


def fetch_period_reference_price(ticker: str, period_key: str):
    """
    Yahoo-stil referencepris: close STRIKT før periode-start.
    Returnerer None for "1D" og "Maks" (caller bruger andre referencepunkter).
    """
    if period_key in ("1D", "Maks"):
        return None

    fetch_period_map = {
        "1U": "1mo", "1M": "3mo", "3M": "6mo", "6M": "1y",
        "YTD": "2y", "1Å": "2y", "5Å": "6y",
    }
    fetch_period = fetch_period_map.get(period_key, "1mo")

    try:
        tk = yf.Ticker(ticker)
        hist = tk.history(period=fetch_period, interval="1d", prepost=False, auto_adjust=False)
        if hist.empty or "Close" not in hist.columns:
            return None
        closes = hist["Close"].dropna()
        if closes.empty:
            return None
    except Exception:
        return None

    today = pd.Timestamp.now().normalize()
    if period_key == "1U":
        target = today - pd.Timedelta(days=7)
    elif period_key == "1M":
        target = today - pd.DateOffset(months=1)
    elif period_key == "3M":
        target = today - pd.DateOffset(months=3)
    elif period_key == "6M":
        target = today - pd.DateOffset(months=6)
    elif period_key == "YTD":
        target = pd.Timestamp(year=today.year, month=1, day=1)
    elif period_key == "1Å":
        target = today - pd.DateOffset(years=1)
    elif period_key == "5Å":
        target = today - pd.DateOffset(years=5)
    else:
        return None

    target = pd.Timestamp(target).normalize()

    idx = pd.DatetimeIndex(closes.index)
    idx_naive = idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()

    mask = idx_naive < target
    if mask.any():
        return float(closes[mask].iloc[-1])

    return float(closes.iloc[0])
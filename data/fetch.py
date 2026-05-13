from utils.formatting import _safe_float
import pandas as pd
import streamlit as st
import yfinance as yf

@st.cache_data(ttl=1800, show_spinner=False)
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


@st.cache_data(ttl=300, show_spinner=False)
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
    
       
@st.cache_data(ttl=60, show_spinner=False)
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
    

@st.cache_data(ttl=60, show_spinner=False)
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


@st.cache_data(ttl=86400, show_spinner=False)
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
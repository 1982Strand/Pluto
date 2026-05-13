import pandas as pd
from datetime import datetime, time
import pytz

def get_us_market_status():
    """
    Returnerer (kode, label, emoji, baggrundsfarve) baseret på aktuel ET-tid.
    Bemærk: Tjekker kun weekender — ikke amerikanske helligdage.
    """
    try:
        et_now = pd.Timestamp.now(tz="America/New_York")
    except Exception:
        # Fallback hvis tz-data ikke er tilgængelig: antag dansk tid - 6 timer
        et_now = pd.Timestamp.now() - pd.Timedelta(hours=6)

    # Weekend
    if et_now.weekday() >= 5:
        return ("weekend", "Lukket (weekend)", "🌙", "#f0f2f6")

    minutes = et_now.hour * 60 + et_now.minute

    if minutes < 4 * 60:                     # 00:00 - 04:00 ET
        return ("overnight", "Overnight (lukket)", "🌙", "#f0f2f6")
    elif minutes < 9 * 60 + 30:              # 04:00 - 09:30 ET
        return ("pre", "Pre-market", "🌅", "#fff3e0")
    elif minutes < 16 * 60:                  # 09:30 - 16:00 ET
        return ("regular", "Marked åbent", "🟢", "#e8f5e9")
    elif minutes < 20 * 60:                  # 16:00 - 20:00 ET
        return ("post", "After-hours", "🌆", "#fff3e0")
    else:                                    # 20:00 - 24:00 ET
        return ("overnight", "Overnight (lukket)", "🌙", "#f0f2f6")


def get_eu_market_status():
    """
    Forenklet status for europæiske børser (Nasdaq Copenhagen, Xetra, Euronext).
    Bruger Europe/Copenhagen-tid; primær handelstid 09:00-17:00 CET/CEST.
    Ingen pre-/after-market — disse børser har stort set ikke detailhandel udenfor.
    """
    try:
        cph_now = pd.Timestamp.now(tz="Europe/Copenhagen")
    except Exception:
        cph_now = pd.Timestamp.now()
    if cph_now.weekday() >= 5:
        return ("weekend", "Lukket (weekend)", "🌙", "#f0f2f6")
    minutes = cph_now.hour * 60 + cph_now.minute
    if 9 * 60 <= minutes < 17 * 60:
        return ("regular", "Marked åbent", "🟢", "#e8f5e9")
    return ("closed", "Lukket", "🌙", "#f0f2f6")


def get_market_status_for_currency(asset_ccy):
    """Vælger session-funktion ud fra aktivets noteringsvaluta."""
    if asset_ccy == "USD":
        return get_us_market_status()
    return get_eu_market_status()
import json
import os
from datetime import time

from config import DEPOSIT_TIMES_FILE, DEFAULT_DEPOSIT_TIME


def _deposit_key(date_str, amount, currency="DKK"):
    """Generér unik nøgle pr. deposit (dato + beløb + valuta)."""
    return f"{date_str}T{amount}_{currency}"


def _time_to_frac(hhmm):
    """Konverter 'HH:MM' eller datetime.time til fraktion af døgn (0..1)."""
    if isinstance(hhmm, time):
        return (hhmm.hour * 60 + hhmm.minute) / (24 * 60)
    if isinstance(hhmm, str) and ":" in hhmm:
        try:
            h, m = hhmm.split(":")[:2]
            return (int(h) * 60 + int(m)) / (24 * 60)
        except (ValueError, IndexError):
            pass
    return 9 / 24  # default 09:00


def load_deposit_times():
    """Læs deposit_times.json. Returnér tom dict hvis fil mangler/korrupt."""
    if not os.path.exists(DEPOSIT_TIMES_FILE):
        return {}
    try:
        with open(DEPOSIT_TIMES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_deposit_times(times_dict):
    """Skriv deposit_times.json. Returnér True ved succes."""
    try:
        with open(DEPOSIT_TIMES_FILE, "w", encoding="utf-8") as f:
            json.dump(times_dict, f, indent=2, ensure_ascii=False)
        return True
    except OSError:
        return False
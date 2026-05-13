import numpy as np


def _safe_float(x):
    """Konverter til float, returnér None ved NaN/None/parsefejl."""
    try:
        if x is None:
            return None
        f = float(x)
        return f if not np.isnan(f) else None
    except (TypeError, ValueError):
        return None


_CCY_SYMBOLS = {"USD": "$", "EUR": "€", "DKK": "kr."}


def _da_num(value, decimals=2, signed=False):
    """Formatér tal med dansk talnotation: . for tusinde, , for decimal."""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(f):
        return "—"
    sign_flag = "+" if signed else ""
    raw = f"{f:{sign_flag},.{decimals}f}"
    return raw.replace(",", "X").replace(".", ",").replace("X", ".")


def format_currency(value, ccy):
    """Formatér beløb med valuta-præfix og dansk talnotation (1.234,56)."""
    formatted = _da_num(value)
    if formatted == "—":
        return formatted
    symbol = _CCY_SYMBOLS.get(ccy, ccy or "")
    return f"{symbol} {formatted}".strip()


def format_big_number(value, decimals=2):
    """Formatér store tal kompakt med suffiks (K/M/B/T) og dansk decimal-komma."""
    if value is None:
        return "—"
    try:
        f = float(value)
    except (TypeError, ValueError):
        return "—"
    if np.isnan(f):
        return "—"
    abs_f = abs(f)
    if abs_f >= 1e12:
        return _da_num(f / 1e12, decimals=decimals) + "T"
    if abs_f >= 1e9:
        return _da_num(f / 1e9, decimals=decimals) + "B"
    if abs_f >= 1e6:
        return _da_num(f / 1e6, decimals=decimals) + "M"
    if abs_f >= 1e3:
        return _da_num(f / 1e3, decimals=decimals) + "K"
    return _da_num(f, decimals=decimals)


def format_quantity(qty):
    """Antal aktier — op til 9 decimaler, dansk komma, trim trailing zeros."""
    if qty is None:
        return "—"
    s = f"{qty:.9f}".rstrip("0").rstrip(".")
    if s == "" or s == "-":
        s = "0"
    return s.replace(".", ",")


def _flatten_html(html):
    """Strip leading whitespace pr. linje, så Streamlits markdown-parser ikke
    fortolker indrykket HTML som <pre>-kodeblokke."""
    return "".join(line.strip() for line in html.splitlines())
    
def color_change_str(val):
    """Farv strenge der starter med + (grøn) eller - (rød)."""
    if not isinstance(val, str):
        return ""
    if val.startswith("+"):
        return "color: #2e7d32; font-weight: 600"
    if val.startswith("-"):
        return "color: #d32f2f; font-weight: 600"
    return ""
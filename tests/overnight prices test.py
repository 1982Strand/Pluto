"""
test_overnight_data.py
======================
Test af gratis datakilder til overnight-priser (Blue Ocean ATS, 8pm-4am ET).

Kør scriptet i overnight-vinduet: 02:00-10:00 dansk tid (søndag-torsdag nat).
Du kan også køre det i dagtimerne — scriptet rapporterer hvad det finder og
markerer tydeligt om data stammer fra overnight-vinduet eller ej.

Ingen installation nødvendig udover det din app allerede bruger:
    pip install yfinance requests pandas pytz

Kør med:
    python test_overnight_data.py
    python test_overnight_data.py --ticker AAPL MSFT   # test specifikke tickers
"""

import sys
import json
import time
import argparse
import textwrap
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

DEFAULT_TICKERS = [
    "AAPL",   # Apple — stor, meget likvid
    "NVDA",   # Nvidia — høj overnatteomsætning
    "MSFT",   # Microsoft
    "TSLA",   # Tesla — populær i overnight-sessioner
    "SPY",    # S&P 500 ETF — benchmark
    "AMZN",   # Amazon
    "SNDK",   # Sandisk
]

ET_ZONE  = ZoneInfo("America/New_York")
CPH_ZONE = ZoneInfo("Europe/Copenhagen")

# ---------------------------------------------------------------------------
# Hjælpefunktioner
# ---------------------------------------------------------------------------

def _now_et() -> datetime:
    return datetime.now(ET_ZONE)

def _now_cph() -> datetime:
    return datetime.now(CPH_ZONE)

def _is_overnight_et(dt: datetime) -> bool:
    """Returnerer True hvis tidspunktet er inden for BOATS-sessionen (20:00-04:00 ET)."""
    h = dt.hour
    return h >= 20 or h < 4

def _ts_to_et(ts) -> datetime:
    """Konverter timestamp (int eller datetime) til ET datetime."""
    if isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    else:
        dt = pd.Timestamp(ts)
        if dt.tzinfo is None:
            dt = dt.tz_localize("UTC")
        dt = dt.to_pydatetime()
    return dt.astimezone(ET_ZONE)

def _fmt_et(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S ET")

def _is_row_overnight(row_dt) -> bool:
    et = _ts_to_et(row_dt)
    return _is_overnight_et(et)

def _print_header(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)

def _print_section(title: str) -> None:
    print()
    print(f"  ── {title} ──")

def _result_line(ok: bool, msg: str) -> str:
    icon = "✅" if ok else "❌"
    return f"  {icon}  {msg}"

def _overnight_tag(has_overnight: bool) -> str:
    return "🌙 OVERNIGHT DATA FUNDET" if has_overnight else "☀️  kun regular/extended hours"

def _summarize_df(df: pd.DataFrame, label: str) -> dict:
    """Analysér en DataFrame med tidsindekseret prisdata."""
    if df is None or df.empty:
        return {"ok": False, "rows": 0, "overnight_rows": 0, "first": None, "last": None}

    # Normaliser index til datetime
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    overnight_mask = pd.Series(
        [_is_overnight_et(ts.astimezone(ET_ZONE)) for ts in idx],
        index=df.index
    )
    overnight_rows = int(overnight_mask.sum())

    first_et = _ts_to_et(idx[0])
    last_et  = _ts_to_et(idx[-1])

    return {
        "ok": True,
        "rows": len(df),
        "overnight_rows": overnight_rows,
        "first": _fmt_et(first_et),
        "last":  _fmt_et(last_et),
    }

# ---------------------------------------------------------------------------
# TEST 1: yfinance history() med prepost=True
# ---------------------------------------------------------------------------

def test_yfinance_prepost(tickers: list[str]) -> dict:
    _print_header("TEST 1: yfinance .history(prepost=True)")
    print(textwrap.dedent("""
    Bruger yfinance's officielle parameter til at inkludere pre- og post-market
    data. Siden Yahoo Finance partnerskab med Blue Ocean (nov 2025) kan dette
    muligvis nu inkludere overnight-sessionen 20:00-04:00 ET.
    Interval: 1m | Period: 5d
    """).strip())

    results = {}
    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            df = t.history(period="5d", interval="1m", prepost=True)
            s = _summarize_df(df, ticker)
            results[ticker] = s

            has_on = s["overnight_rows"] > 0
            print(_result_line(s["ok"], f"{ticker}: {s['rows']} rækker  |  {s['overnight_rows']} overnight  |  {s['first']} → {s['last']}"))
            if has_on:
                # Vis et par overnight-rækker
                idx = pd.DatetimeIndex(df.index)
                if idx.tz is None:
                    idx = idx.tz_localize("UTC")
                mask = [_is_overnight_et(ts.astimezone(ET_ZONE)) for ts in idx]
                on_df = df[mask]
                close_col = "Close" if "Close" in on_df.columns else on_df.columns[3]
                sample = on_df[close_col].dropna().tail(3)
                for ts, price in sample.items():
                    print(f"           {_fmt_et(_ts_to_et(ts))}  →  {price:.4f}")
        except Exception as e:
            results[ticker] = {"ok": False, "error": str(e)}
            print(_result_line(False, f"{ticker}: FEJL — {e}"))

        time.sleep(0.3)

    return results

# ---------------------------------------------------------------------------
# TEST 2: Yahoo Finance v8 chart-endpoint direkte (requests)
# ---------------------------------------------------------------------------

def test_yahoo_v8_chart(tickers: list[str]) -> dict:
    _print_header("TEST 2: Yahoo Finance v8/chart direkte (requests + prepost=true)")
    print(textwrap.dedent("""
    Rammer Yahoo Finance's interne chart-API direkte — det samme endpoint som
    Yahoo Finance-websitet bruger til at vise overnight-priser.
    Returnerer 1m-data for seneste dag med prepost=true.
    """).strip())

    results = {}
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
    }

    for ticker in tickers:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "interval":  "1m",
            "range":     "1d",
            "prepost":   "true",
            "includePrePost": "true",
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            result = data.get("chart", {}).get("result", [])
            if not result:
                raise ValueError("Tomt result-felt i JSON")

            r = result[0]
            timestamps = r.get("timestamp", [])
            quotes = r.get("indicators", {}).get("quote", [{}])[0]
            closes = quotes.get("close", [])

            if not timestamps:
                raise ValueError("Ingen timestamps returneret")

            # Find overnight
            overnight_pts = [
                (ts, c) for ts, c in zip(timestamps, closes)
                if c is not None and _is_overnight_et(_ts_to_et(ts))
            ]

            has_on = len(overnight_pts) > 0
            last_ts  = _fmt_et(_ts_to_et(timestamps[-1]))
            first_ts = _fmt_et(_ts_to_et(timestamps[0]))

            meta = r.get("meta", {})
            market_state = meta.get("marketState", "UKENDT")
            regular_close = meta.get("regularMarketPrice")
            current_price = meta.get("postMarketPrice") or meta.get("preMarketPrice") or regular_close

            print(_result_line(True,
                f"{ticker}: {len(timestamps)} punkter  |  {len(overnight_pts)} overnight  |  "
                f"marketState={market_state}  |  pris={current_price}"))
            print(f"           Periode: {first_ts} → {last_ts}")

            if has_on:
                print(f"           {_overnight_tag(True)}")
                for ts, price in overnight_pts[-3:]:
                    print(f"             {_fmt_et(_ts_to_et(ts))}  →  {price:.4f}")

            results[ticker] = {
                "ok": True,
                "total_pts": len(timestamps),
                "overnight_pts": len(overnight_pts),
                "market_state": market_state,
                "current_price": current_price,
                "first": first_ts,
                "last": last_ts,
            }

        except Exception as e:
            results[ticker] = {"ok": False, "error": str(e)}
            print(_result_line(False, f"{ticker}: FEJL — {e}"))

        time.sleep(0.4)

    return results

# ---------------------------------------------------------------------------
# TEST 3: Yahoo Finance v7 quote — live metadata inkl. marketState
# ---------------------------------------------------------------------------

def test_yahoo_v7_quote(tickers: list[str]) -> dict:
    _print_header("TEST 3: Yahoo Finance v7/quote — live quote metadata")
    print(textwrap.dedent("""
    Henter live quote-metadata pr. ticker. Viser marketState, og om der er en
    postMarketPrice / preMarketPrice. Hvis Yahoo Finance nu viser overnight-priser
    på websitet, kan det dukke op her som et nyt felt (f.eks. 'overnightPrice').
    Vi udskriver alle kendte pris-felter så du kan se hvad der returneres.
    """).strip())

    results = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    ticker_str = ",".join(tickers)
    url = "https://query1.finance.yahoo.com/v7/finance/quote"
    params = {"symbols": ticker_str, "fields": (
        "regularMarketPrice,postMarketPrice,preMarketPrice,"
        "marketState,regularMarketTime,postMarketTime,preMarketTime,"
        "fiftyTwoWeekHigh,fiftyTwoWeekLow,shortName"
    )}

    PRICE_FIELDS = [
        "regularMarketPrice", "regularMarketTime",
        "postMarketPrice",    "postMarketTime",
        "preMarketPrice",     "preMarketTime",
        # mulige overnight-felter Yahoo kan tilføje:
        "overnightPrice",     "overnightTime",
        "extendedMarketPrice","extendedMarketTime",
    ]

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        quotes_list = data.get("quoteResponse", {}).get("result", [])

        if not quotes_list:
            print(_result_line(False, "Intet svar fra v7/quote"))
            return {}

        for q in quotes_list:
            sym = q.get("symbol", "?")
            state = q.get("marketState", "?")

            found = {}
            for field in PRICE_FIELDS:
                if field in q and q[field] is not None:
                    found[field] = q[field]

            has_overnight_field = any("overnight" in k.lower() for k in found)
            has_extended        = "extendedMarketPrice" in found

            print(_result_line(True, f"{sym}  marketState={state}"))
            for k, v in found.items():
                if "Time" in k and isinstance(v, (int, float)):
                    ts_str = _fmt_et(_ts_to_et(v))
                    print(f"           {k}: {ts_str}")
                else:
                    print(f"           {k}: {v}")

            if has_overnight_field:
                print(f"           {_overnight_tag(True)}")
            if has_extended:
                print("           ℹ️  extendedMarketPrice fundet — mulig overnight")

            results[sym] = {"ok": True, "marketState": state, "fields": found}

    except Exception as e:
        print(_result_line(False, f"Fejl: {e}"))

    return results

# ---------------------------------------------------------------------------
# TEST 4: yfinance fast_info / info — hvad returneres live?
# ---------------------------------------------------------------------------

def test_yfinance_fast_info(tickers: list[str]) -> dict:
    _print_header("TEST 4: yfinance fast_info — live pris-felter")
    print(textwrap.dedent("""
    yfinance's fast_info-objekt returnerer en håndfuld live-felter effektivt.
    Vi undersøger om der dukker overnight-priser op her.
    """).strip())

    results = {}
    INTERESTING = [
        "last_price", "open", "previous_close",
        "post_market_price", "pre_market_price",
        "regular_market_price",
    ]

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            fi = t.fast_info

            found = {}
            for attr in INTERESTING:
                try:
                    val = getattr(fi, attr, None)
                    if val is not None:
                        found[attr] = val
                except Exception:
                    pass

            # Forsøg også at se om der er et overnight_price attribut
            for attr in dir(fi):
                if "overnight" in attr.lower() or "extended" in attr.lower():
                    try:
                        found[f"[ny!] {attr}"] = getattr(fi, attr)
                    except Exception:
                        pass

            has_new = any("[ny!]" in k for k in found)
            print(_result_line(True, f"{ticker}:"))
            for k, v in found.items():
                print(f"           {k}: {v}")
            if has_new:
                print(f"           {_overnight_tag(True)}")

            results[ticker] = {"ok": True, "fields": found}

        except Exception as e:
            results[ticker] = {"ok": False, "error": str(e)}
            print(_result_line(False, f"{ticker}: FEJL — {e}"))

        time.sleep(0.2)

    return results

# ---------------------------------------------------------------------------
# TEST 5: yfinance download() multi-ticker med prepost=True
# ---------------------------------------------------------------------------

def test_yfinance_download_multi(tickers: list[str]) -> dict:
    _print_header("TEST 5: yfinance.download() multi-ticker (prepost=True, 1m, 2d)")
    print(textwrap.dedent("""
    Samme som Test 1, men bruger download()-funktionen der henter alle tickers
    i ét kald — mere effektivt og tæt på hvordan din app allerede henter data.
    """).strip())

    results = {}
    try:
        df = yf.download(
            tickers,
            period="2d",
            interval="1m",
            prepost=True,
            group_by="ticker",
            auto_adjust=True,
            progress=False,
            threads=True,
        )

        if df.empty:
            print(_result_line(False, "Tom DataFrame returneret"))
            return {}

        for ticker in tickers:
            try:
                if len(tickers) == 1:
                    sub = df
                else:
                    sub = df[ticker] if ticker in df.columns.get_level_values(0) else pd.DataFrame()

                sub = sub.dropna(how="all")
                s = _summarize_df(sub, ticker)

                has_on = s["overnight_rows"] > 0
                print(_result_line(s["ok"],
                    f"{ticker}: {s['rows']} rækker  |  {s['overnight_rows']} overnight  "
                    f"|  {s['first']} → {s['last']}"))
                if has_on:
                    print(f"           {_overnight_tag(True)}")
                results[ticker] = s

            except Exception as e:
                results[ticker] = {"ok": False, "error": str(e)}
                print(_result_line(False, f"{ticker}: FEJL — {e}"))

    except Exception as e:
        print(_result_line(False, f"Download-fejl: {e}"))

    return results

# ---------------------------------------------------------------------------
# TEST 6: Direkte Yahoo Finance chart-endpoint — range=5d for historik
# ---------------------------------------------------------------------------

def test_yahoo_v8_5day(tickers: list[str]) -> dict:
    _print_header("TEST 6: Yahoo Finance v8/chart — 5-dages historik med prepost")
    print(textwrap.dedent("""
    Henter 5 dages 1m-data med prepost=true via v8-endpointet. Giver et billede
    af om overnight-data er persisteret historisk (ikke kun live/seneste time).
    """).strip())

    results = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    for ticker in tickers[:3]:  # begrænset til 3 for at spare tid
        url = f"https://query2.finance.yahoo.com/v8/finance/chart/{ticker}"
        params = {
            "interval":  "1m",
            "range":     "5d",
            "prepost":   "true",
            "includePrePost": "true",
        }
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            r = data.get("chart", {}).get("result", [{}])[0]
            timestamps = r.get("timestamp", [])
            quotes = r.get("indicators", {}).get("quote", [{}])[0]
            closes = quotes.get("close", [])

            overnight_pts = [
                (ts, c) for ts, c in zip(timestamps, closes)
                if c is not None and _is_overnight_et(_ts_to_et(ts))
            ]

            print(_result_line(True,
                f"{ticker}: {len(timestamps)} punkter totalt  |  "
                f"{len(overnight_pts)} overnight"))

            if overnight_pts:
                print(f"           {_overnight_tag(True)}")
                # Gruppér overnight-punkter efter dato
                by_date: dict = {}
                for ts, price in overnight_pts:
                    et_dt = _ts_to_et(ts)
                    date_str = et_dt.strftime("%Y-%m-%d")
                    by_date.setdefault(date_str, []).append((et_dt, price))
                for date_str, pts in sorted(by_date.items()):
                    print(f"           Dato {date_str}: {len(pts)} overnight-punkter  "
                          f"({_fmt_et(pts[0][0])} → {_fmt_et(pts[-1][0])})")

            results[ticker] = {
                "ok": True,
                "total_pts": len(timestamps),
                "overnight_pts": len(overnight_pts),
            }

        except Exception as e:
            results[ticker] = {"ok": False, "error": str(e)}
            print(_result_line(False, f"{ticker}: FEJL — {e}"))

        time.sleep(0.5)

    return results

# ---------------------------------------------------------------------------
# Samlet opsummering
# ---------------------------------------------------------------------------

def _print_summary(all_results: dict, now_et: datetime) -> None:
    _print_header("SAMLET OPSUMMERING")

    in_overnight = _is_overnight_et(now_et)
    session_note = (
        "🌙 Du kører i OVERNIGHT-sessionen (20:00–04:00 ET) — optimalt tidspunkt!"
        if in_overnight else
        "☀️  Du kører UDENFOR overnight-sessionen. "
        "Kør igen mellem 02:00–10:00 dansk tid for at teste live overnight-data."
    )
    print(f"\n  Tidspunkt: {_fmt_et(now_et)}")
    print(f"  {session_note}\n")

    score = {}
    for test_name, ticker_results in all_results.items():
        if not isinstance(ticker_results, dict):
            continue
        overnight_found = False
        data_found = False
        for ticker, r in ticker_results.items():
            if not isinstance(r, dict):
                continue
            if r.get("ok"):
                data_found = True
            if (r.get("overnight_rows", 0) or r.get("overnight_pts", 0)) > 0:
                overnight_found = True
        score[test_name] = (data_found, overnight_found)

    print(f"  {'Test':<45} {'Data?':<10} {'Overnight?'}")
    print(f"  {'-'*45} {'-'*10} {'-'*12}")
    for test_name, (data_ok, on_ok) in score.items():
        data_str = "✅ ja" if data_ok else "❌ nej"
        on_str   = "🌙 JA!" if on_ok else ("⏳ nej (prøv i nat)" if data_ok else "❌")
        print(f"  {test_name:<45} {data_str:<10} {on_str}")

    print()
    print("  Anbefaling til implementering:")
    any_overnight = any(on for _, on in score.values())
    if any_overnight:
        print("  ✅ Mindst én kilde giver overnight-data!")
        print("     → Brug den/de markerede test(s) som grundlag for fetch_overnight_price()")
        print("     → Prioritér Test 2 (v8 direkte) da det er tættest på Yahoo's eget website")
    else:
        if in_overnight:
            print("  ❌ Ingen af kilderne returnerede overnight-data — selv i overnight-sessionen.")
            print("     → Yahoo Finance eksponerer endnu ikke BOATS-data via deres API-endpoints.")
            print("     → Databento er pt. eneste frit tilgængelige kilde (med free credit).")
        else:
            print("  ⏳ Kør scriptet igen i nat (02:00–10:00 dansk tid) for endeligt svar.")
    print()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test overnight-datakilder for US-aktier."
    )
    parser.add_argument(
        "--tickers", nargs="+", default=DEFAULT_TICKERS,
        help="Ticker-symboler der skal testes (standard: se DEFAULT_TICKERS)"
    )
    parser.add_argument(
        "--skip", nargs="+", default=[],
        help="Test-numre der skal springes over, f.eks. --skip 4 5"
    )
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]
    skip    = set(args.skip)

    now_et = _now_et()

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║          OVERNIGHT DATA TEST — Pluto Portefølje                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print(f"  Starttidspunkt (DK):  {_now_cph().strftime('%Y-%m-%d %H:%M:%S %Z')}")
    print(f"  Starttidspunkt (ET):  {_fmt_et(now_et)}")
    print(f"  Overnight-session:    {'🟢 AKTIV (20:00–04:00 ET)' if _is_overnight_et(now_et) else '🔴 INAKTIV'}")
    print(f"  Tickers:              {', '.join(tickers)}")

    all_results = {}

    if "1" not in skip:
        all_results["Test 1: yfinance history(prepost=True)"] = \
            test_yfinance_prepost(tickers)

    if "2" not in skip:
        all_results["Test 2: Yahoo v8/chart direkte (1d)"] = \
            test_yahoo_v8_chart(tickers)

    if "3" not in skip:
        all_results["Test 3: Yahoo v7/quote metadata"] = \
            test_yahoo_v7_quote(tickers)

    if "4" not in skip:
        all_results["Test 4: yfinance fast_info"] = \
            test_yfinance_fast_info(tickers)

    if "5" not in skip:
        all_results["Test 5: yfinance download() multi"] = \
            test_yfinance_download_multi(tickers)

    if "6" not in skip:
        all_results["Test 6: Yahoo v8/chart (5d historik)"] = \
            test_yahoo_v8_5day(tickers)

    _print_summary(all_results, now_et)

    # Gem råresultater til fil for evt. videre analyse
    output_file = "overnight_test_results.json"
    try:
        # Konverter til JSON-serialiserbar form
        def _jsonable(obj):
            if isinstance(obj, dict):
                return {k: _jsonable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_jsonable(i) for i in obj]
            if isinstance(obj, float):
                return round(obj, 6)
            return obj

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_at_et":  _fmt_et(now_et),
                    "run_at_cph": _now_cph().strftime("%Y-%m-%d %H:%M:%S %Z"),
                    "tickers":    tickers,
                    "results":    _jsonable(all_results),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"  Råresultater gemt i: {output_file}")
    except Exception as e:
        print(f"  (Kunne ikke gemme JSON: {e})")

    print()

if __name__ == "__main__":
    main()
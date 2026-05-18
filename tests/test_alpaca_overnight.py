"""
test_alpaca_overnight.py
========================
Tester Alpaca Data API v2 for overnight-priser (Blue Ocean ATS, 20:00–04:00 ET).

Alpaca har en eksplicit 'feed=boats' parameter — dette script undersøger om den
er tilgængelig på en gratis konto, og hvad den returnerer.

API-nøgler hentes fra miljøvariable (eller du indtaster dem når scriptet spørger):
    Windows:  set APCA_API_KEY_ID=din_nøgle
              set APCA_API_SECRET_KEY=din_secret
    Mac/Linux: export APCA_API_KEY_ID=din_nøgle
               export APCA_API_SECRET_KEY=din_secret

Kør med:
    python test_alpaca_overnight.py

Kør i overnight-vinduet for bedst resultat: 02:00–10:00 dansk tid (søn–tor nat).
"""

import os
import sys
import json
import time
import textwrap
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import requests

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "NVDA", "MSFT", "TSLA", "SPY", "AMZN", "SNDK"]

# ---------------------------------------------------------------------------
# 👇 Indsæt dine Alpaca API-nøgler her
# ---------------------------------------------------------------------------
API_KEY_ID     = "PKK4YPY6YFMP5J5733DOPECBCF"   # f.eks. "PKXXXXXXXXXXXXXXXX"
API_SECRET_KEY = "EQdQFzZn6KGv2uhrFk4chQLwX3k5knygoEf9G9JaT1pM"   # din secret key
# ---------------------------------------------------------------------------

BASE_URL  = "https://data.alpaca.markets"
PAPER_URL = "https://paper-api.alpaca.markets"   # til konto-info

ET_ZONE  = ZoneInfo("America/New_York")
CPH_ZONE = ZoneInfo("Europe/Copenhagen")

# ---------------------------------------------------------------------------
# Nøglehåndtering
# ---------------------------------------------------------------------------

def _get_credentials() -> tuple[str, str]:
    key    = API_KEY_ID.strip()     or os.environ.get("APCA_API_KEY_ID", "").strip()
    secret = API_SECRET_KEY.strip() or os.environ.get("APCA_API_SECRET_KEY", "").strip()

    if not key or not secret:
        print("  ❌ Ingen API-nøgler fundet.")
        print("     Udfyld API_KEY_ID og API_SECRET_KEY øverst i scriptet og prøv igen.")
        sys.exit(1)

    return key, secret

# ---------------------------------------------------------------------------
# Hjælpefunktioner
# ---------------------------------------------------------------------------

def _now_et()  -> datetime: return datetime.now(ET_ZONE)
def _now_cph() -> datetime: return datetime.now(CPH_ZONE)

def _is_overnight_et(dt: datetime) -> bool:
    h = dt.hour
    return h >= 20 or h < 4

def _fmt_et(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S ET")

def _fmt_cph(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S CEST")

def _parse_iso(s: str) -> datetime:
    """Parser ISO-8601 timestamp til ET datetime."""
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt.astimezone(ET_ZONE)

def _is_ts_overnight(ts_str: str) -> bool:
    try:
        return _is_overnight_et(_parse_iso(ts_str))
    except Exception:
        return False

def _print_header(title: str) -> None:
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)

def _ok(msg: str)   -> None: print(f"  ✅  {msg}")
def _fail(msg: str) -> None: print(f"  ❌  {msg}")
def _info(msg: str) -> None: print(f"  ℹ️   {msg}")
def _moon(msg: str) -> None: print(f"  🌙  {msg}")

def _get(url: str, headers: dict, params: dict = None,
         label: str = "") -> requests.Response | None:
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        return r
    except requests.RequestException as e:
        _fail(f"{label} — netværksfejl: {e}")
        return None

# ---------------------------------------------------------------------------
# TEST 0: Kontoverifikation
# ---------------------------------------------------------------------------

def test_account(headers: dict) -> dict:
    _print_header("TEST 0: Kontoverifikation")
    print("  Bekræfter at API-nøglerne er gyldige og viser kontoniveau.")

    r = _get(f"{PAPER_URL}/v2/account", headers, label="account")
    if r is None:
        return {}

    if r.status_code != 200:
        _fail(f"HTTP {r.status_code}: {r.text[:200]}")
        return {}

    data = r.json()
    plan    = data.get("plan", data.get("subscription", {}).get("name", "ukendt"))
    status  = data.get("status", "?")
    cash    = data.get("cash", "?")
    equity  = data.get("equity", "?")

    _ok(f"Konto aktiv — status: {status}")
    _info(f"Plan/niveau: {plan}")
    _info(f"Cash: {cash} USD  |  Equity: {equity} USD")

    # Forsøg live-konto endpoint hvis paper fejler
    return {"plan": plan, "status": status}

# ---------------------------------------------------------------------------
# TEST 1: Alpaca latest quote — alle feeds
# ---------------------------------------------------------------------------

def test_latest_quote(headers: dict) -> dict:
    _print_header("TEST 1: Latest Quote — sammenlign feeds (iex / sip / boats)")
    print(textwrap.dedent("""
    Henter seneste quote for hvert feed og sammenligner.
    'boats'-feed er Blue Ocean ATS-specifikt — understøttes det af din konto?
    """).strip())

    feeds   = ["iex", "sip", "boats"]
    results = {}

    for ticker in TICKERS[:4]:  # de første 4 er nok til at afklare
        print(f"\n  ── {ticker} ──")
        results[ticker] = {}

        for feed in feeds:
            url    = f"{BASE_URL}/v2/stocks/{ticker}/quotes/latest"
            params = {"feed": feed}
            r = _get(url, headers, params, label=f"{ticker}/{feed}")

            if r is None:
                results[ticker][feed] = {"ok": False, "error": "netværksfejl"}
                continue

            if r.status_code == 403:
                _fail(f"feed={feed}: 403 Forbidden — ikke tilladt på dette kontoniveau")
                results[ticker][feed] = {"ok": False, "error": "403 forbidden"}
                continue

            if r.status_code == 422:
                _fail(f"feed={feed}: 422 — feed-parameter ikke understøttet")
                results[ticker][feed] = {"ok": False, "error": "422 unsupported"}
                continue

            if r.status_code != 200:
                _fail(f"feed={feed}: HTTP {r.status_code} — {r.text[:100]}")
                results[ticker][feed] = {"ok": False, "error": f"HTTP {r.status_code}"}
                continue

            data  = r.json()
            quote = data.get("quote", {})
            ts    = quote.get("t", "")
            ap    = quote.get("ap")   # ask price
            bp    = quote.get("bp")   # bid price
            c     = quote.get("c", [])  # conditions

            is_on = _is_ts_overnight(ts) if ts else False

            et_str = _fmt_et(_parse_iso(ts)) if ts else "ingen timestamp"
            price_str = f"bid={bp}  ask={ap}" if bp is not None else "ingen pris"

            if is_on:
                _moon(f"feed={feed}: {et_str}  |  {price_str}  ← OVERNIGHT!")
            else:
                _ok(f"feed={feed}: {et_str}  |  {price_str}")

            results[ticker][feed] = {
                "ok": True,
                "timestamp": et_str,
                "is_overnight": is_on,
                "bid": bp,
                "ask": ap,
                "conditions": c,
            }

        time.sleep(0.3)

    return results

# ---------------------------------------------------------------------------
# TEST 2: Historical bars — overnight tidsvindue, alle feeds
# ---------------------------------------------------------------------------

def test_bars_overnight_window(headers: dict) -> dict:
    _print_header("TEST 2: Historical Bars — eksplicit overnight tidsvindue")
    print(textwrap.dedent("""
    Henter 1-minuts bars for gårsdagens overnight-session (20:00–04:00 ET).
    Tester alle feeds og rapporterer antal bars og prisdata.
    """).strip())

    now_et    = _now_et()
    # Gårsdagens 20:00 ET → i dag 04:00 ET
    today_et  = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    on_start  = (today_et - timedelta(days=1)).replace(hour=20)
    on_end    = today_et.replace(hour=4)

    start_str = on_start.isoformat()
    end_str   = on_end.isoformat()

    _info(f"Overnight-vindue der testes: {_fmt_et(on_start)} → {_fmt_et(on_end)}")

    feeds   = ["iex", "sip", "boats"]
    results = {}

    for ticker in TICKERS[:4]:
        print(f"\n  ── {ticker} ──")
        results[ticker] = {}

        for feed in feeds:
            url    = f"{BASE_URL}/v2/stocks/{ticker}/bars"
            params = {
                "start":     start_str,
                "end":       end_str,
                "timeframe": "1Min",
                "feed":      feed,
                "limit":     500,
            }
            r = _get(url, headers, params, label=f"{ticker}/bars/{feed}")

            if r is None:
                results[ticker][feed] = {"ok": False}
                continue

            if r.status_code in (403, 422):
                _fail(f"feed={feed}: HTTP {r.status_code} — ikke tilgængeligt")
                results[ticker][feed] = {"ok": False, "error": f"HTTP {r.status_code}"}
                continue

            if r.status_code != 200:
                _fail(f"feed={feed}: HTTP {r.status_code} — {r.text[:120]}")
                results[ticker][feed] = {"ok": False, "error": f"HTTP {r.status_code}"}
                continue

            data = r.json()
            bars = data.get("bars") or []

            if not bars:
                _info(f"feed={feed}: 0 bars i vinduet")
                results[ticker][feed] = {"ok": True, "bars": 0}
                continue

            first_ts = _fmt_et(_parse_iso(bars[0]["t"]))
            last_ts  = _fmt_et(_parse_iso(bars[-1]["t"]))
            first_c  = bars[0].get("c")
            last_c   = bars[-1].get("c")

            _moon(f"feed={feed}: {len(bars)} bars  |  {first_ts} → {last_ts}  "
                  f"|  pris {first_c} → {last_c}")

            results[ticker][feed] = {
                "ok": True,
                "bars": len(bars),
                "first": first_ts,
                "last":  last_ts,
                "first_close": first_c,
                "last_close":  last_c,
            }

        time.sleep(0.3)

    return results

# ---------------------------------------------------------------------------
# TEST 3: Latest bar — alle feeds
# ---------------------------------------------------------------------------

def test_latest_bar(headers: dict) -> dict:
    _print_header("TEST 3: Latest Bar — hvad er den seneste tilgængelige bar?")
    print(textwrap.dedent("""
    Henter den seneste 1-minuts bar for hvert feed.
    Tidsstemplet afslører om feeden er live i overnight-sessionen.
    """).strip())

    feeds   = ["iex", "sip", "boats"]
    results = {}

    for ticker in TICKERS[:3]:
        print(f"\n  ── {ticker} ──")
        results[ticker] = {}

        for feed in feeds:
            url    = f"{BASE_URL}/v2/stocks/{ticker}/bars/latest"
            params = {"feed": feed}
            r = _get(url, headers, params, label=f"{ticker}/latest-bar/{feed}")

            if r is None:
                results[ticker][feed] = {"ok": False}
                continue

            if r.status_code in (403, 422):
                _fail(f"feed={feed}: HTTP {r.status_code}")
                results[ticker][feed] = {"ok": False, "error": f"HTTP {r.status_code}"}
                continue

            if r.status_code != 200:
                _fail(f"feed={feed}: HTTP {r.status_code} — {r.text[:100]}")
                results[ticker][feed] = {"ok": False, "error": f"HTTP {r.status_code}"}
                continue

            data = r.json()
            bar  = data.get("bar", {})
            ts   = bar.get("t", "")
            c    = bar.get("c")
            o    = bar.get("o")
            v    = bar.get("v")

            is_on  = _is_ts_overnight(ts) if ts else False
            et_str = _fmt_et(_parse_iso(ts)) if ts else "?"

            if is_on:
                _moon(f"feed={feed}: {et_str}  close={c}  open={o}  vol={v}  ← OVERNIGHT!")
            else:
                _ok(f"feed={feed}: {et_str}  close={c}  open={o}  vol={v}")

            results[ticker][feed] = {
                "ok": True,
                "timestamp": et_str,
                "is_overnight": is_on,
                "close": c,
                "volume": v,
            }

        time.sleep(0.3)

    return results

# ---------------------------------------------------------------------------
# TEST 4: Latest trade — hvad er den seneste handel?
# ---------------------------------------------------------------------------

def test_latest_trade(headers: dict) -> dict:
    _print_header("TEST 4: Latest Trade — seneste handel pr. feed")
    print(textwrap.dedent("""
    Henter den seneste trade for hvert feed.
    Blue Ocean ATS-handler har en specifik 'exchange'-kode (N = BOATS).
    """).strip())

    feeds   = ["iex", "sip", "boats"]
    results = {}

    for ticker in TICKERS[:4]:
        print(f"\n  ── {ticker} ──")
        results[ticker] = {}

        for feed in feeds:
            url    = f"{BASE_URL}/v2/stocks/{ticker}/trades/latest"
            params = {"feed": feed}
            r = _get(url, headers, params, label=f"{ticker}/latest-trade/{feed}")

            if r is None:
                results[ticker][feed] = {"ok": False}
                continue

            if r.status_code in (403, 422):
                _fail(f"feed={feed}: HTTP {r.status_code}")
                results[ticker][feed] = {"ok": False, "error": f"HTTP {r.status_code}"}
                continue

            if r.status_code != 200:
                _fail(f"feed={feed}: HTTP {r.status_code} — {r.text[:100]}")
                results[ticker][feed] = {"ok": False, "error": f"HTTP {r.status_code}"}
                continue

            data  = r.json()
            trade = data.get("trade", {})
            ts    = trade.get("t", "")
            p     = trade.get("p")   # price
            s     = trade.get("s")   # size
            x     = trade.get("x")   # exchange
            c     = trade.get("c", [])  # conditions

            is_on  = _is_ts_overnight(ts) if ts else False
            et_str = _fmt_et(_parse_iso(ts)) if ts else "?"

            # BOATS-handler har exchange-kode 'N' (Blue Ocean) eller lign.
            boats_note = " ← BOATS-exchange!" if x in ("N", "BOATS", "BO") else ""

            if is_on:
                _moon(f"feed={feed}: {et_str}  pris={p}  størrelse={s}  "
                      f"exchange={x}{boats_note}  ← OVERNIGHT!")
            else:
                _ok(f"feed={feed}: {et_str}  pris={p}  størrelse={s}  exchange={x}{boats_note}")

            results[ticker][feed] = {
                "ok": True,
                "timestamp": et_str,
                "is_overnight": is_on,
                "price": p,
                "exchange": x,
            }

        time.sleep(0.3)

    return results

# ---------------------------------------------------------------------------
# TEST 5: Snapshots — samlet overblik
# ---------------------------------------------------------------------------

def test_snapshots(headers: dict) -> dict:
    _print_header("TEST 5: Snapshots — samlet markedsoverblik")
    print(textwrap.dedent("""
    Alpaca's snapshot-endpoint returnerer latest bar, daily bar, latest trade
    og latest quote i ét kald. Vi tjekker om overnight-priser dukker op her.
    """).strip())

    feeds   = ["iex", "boats"]   # sip er primært for betalte konti
    results = {}

    for feed in feeds:
        print(f"\n  ── feed={feed} ──")
        symbols = ",".join(TICKERS[:5])
        url     = f"{BASE_URL}/v2/stocks/snapshots"
        params  = {"symbols": symbols, "feed": feed}
        r = _get(url, headers, params, label=f"snapshots/{feed}")

        if r is None:
            continue

        if r.status_code in (403, 422):
            _fail(f"HTTP {r.status_code} — ikke tilgængeligt")
            continue

        if r.status_code != 200:
            _fail(f"HTTP {r.status_code} — {r.text[:120]}")
            continue

        data = r.json()

        for ticker in TICKERS[:5]:
            snap = data.get(ticker, {})
            if not snap:
                _info(f"{ticker}: intet snapshot")
                continue

            lt   = snap.get("latestTrade", {})
            lq   = snap.get("latestQuote", {})
            mb   = snap.get("minuteBar", {})

            lt_ts = lt.get("t", "")
            lq_ts = lq.get("t", "")
            mb_ts = mb.get("t", "")

            lt_on = _is_ts_overnight(lt_ts) if lt_ts else False
            lq_on = _is_ts_overnight(lq_ts) if lq_ts else False
            mb_on = _is_ts_overnight(mb_ts) if mb_ts else False

            any_on = lt_on or lq_on or mb_on

            lt_str = _fmt_et(_parse_iso(lt_ts)) if lt_ts else "?"
            mb_str = _fmt_et(_parse_iso(mb_ts)) if mb_ts else "?"

            if any_on:
                _moon(f"{ticker}: trade={lt_str}  bar={mb_str}  pris={lt.get('p')}  ← OVERNIGHT!")
            else:
                _ok(f"{ticker}: trade={lt_str}  bar={mb_str}  pris={lt.get('p')}")

            results[f"{ticker}_{feed}"] = {
                "ok": True,
                "trade_ts": lt_str,
                "bar_ts": mb_str,
                "price": lt.get("p"),
                "is_overnight": any_on,
            }

        time.sleep(0.4)

    return results

# ---------------------------------------------------------------------------
# Samlet opsummering
# ---------------------------------------------------------------------------

def _print_summary(all_results: dict, now_et: datetime) -> None:
    _print_header("SAMLET OPSUMMERING")

    in_overnight = _is_overnight_et(now_et)
    print(f"\n  Tidspunkt: {_fmt_et(now_et)}")
    if in_overnight:
        print("  🌙 Du kørte i OVERNIGHT-sessionen — resultatet er definitivt.")
    else:
        print("  ☀️  Du kørte UDEN FOR overnight-sessionen.")
        print("     Kør igen 02:00–10:00 dansk tid (søn–tor) for live overnight-test.")

    print()

    # Find alle feeds der gav overnight-data
    overnight_feeds = set()
    for test_name, test_data in all_results.items():
        if not isinstance(test_data, dict):
            continue
        for key, val in test_data.items():
            if isinstance(val, dict):
                # Direkte felt
                if val.get("is_overnight"):
                    feed = "boats/iex/sip"
                    overnight_feeds.add(f"{test_name}")
                # Nøgler med feed-undernøgler
                for feed_name in ("iex", "sip", "boats"):
                    sub = val.get(feed_name, {})
                    if isinstance(sub, dict) and sub.get("is_overnight"):
                        overnight_feeds.add(f"{test_name} (feed={feed_name})")

    if overnight_feeds:
        print("  🌙 Overnight-data fundet i følgende tests:")
        for f in sorted(overnight_feeds):
            print(f"     • {f}")
        print()
        print("  ✅ Alpaca kan bruges til overnight-priser!")
        print("     → Næste skridt: implementer fetch_overnight_price() i data/fetch.py")
        print("     → Brug feed=boats (eller det der virkede) med /v2/stocks/{ticker}/bars/latest")
    else:
        if in_overnight:
            print("  ❌ Ingen overnight-data fundet — selv i overnight-sessionen.")
            print("     → Alpaca gratis konto har ikke adgang til BOATS-feed.")
            print("     → Overvej Alpaca Algo Trader Plus ($9/md) for fuld adgang.")
        else:
            print("  ⏳ Ingen overnight-data (kørte uden for sessionen — prøv i nat).")
    print()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║       ALPACA OVERNIGHT DATA TEST — Pluto Portefølje             ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    key, secret = _get_credentials()

    headers = {
        "APCA-API-KEY-ID":     key,
        "APCA-API-SECRET-KEY": secret,
        "Accept":              "application/json",
    }

    now_et = _now_et()
    print(f"\n  Starttidspunkt (DK): {_fmt_cph(_now_cph())}")
    print(f"  Starttidspunkt (ET): {_fmt_et(now_et)}")
    print(f"  Overnight-session:   "
          f"{'🟢 AKTIV (20:00–04:00 ET)' if _is_overnight_et(now_et) else '🔴 INAKTIV'}")
    print(f"  Tickers:             {', '.join(TICKERS)}")

    all_results: dict = {}

    all_results["Test 0: Konto"]           = test_account(headers)
    all_results["Test 1: Latest Quote"]    = test_latest_quote(headers)
    all_results["Test 2: Bars overnight"]  = test_bars_overnight_window(headers)
    all_results["Test 3: Latest Bar"]      = test_latest_bar(headers)
    all_results["Test 4: Latest Trade"]    = test_latest_trade(headers)
    all_results["Test 5: Snapshots"]       = test_snapshots(headers)

    _print_summary(all_results, now_et)

    output_file = "alpaca_overnight_results.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            def _clean(obj):
                if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
                if isinstance(obj, list):  return [_clean(i) for i in obj]
                if isinstance(obj, float): return round(obj, 6)
                return obj
            json.dump(
                {"run_at_et": _fmt_et(now_et), "results": _clean(all_results)},
                f, ensure_ascii=False, indent=2,
            )
        print(f"  Råresultater gemt i: {output_file}\n")
    except Exception as e:
        print(f"  (Kunne ikke gemme JSON: {e})\n")

if __name__ == "__main__":
    main()
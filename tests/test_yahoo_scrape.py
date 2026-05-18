"""
test_yahoo_scrape.py
====================
Forsøger at aflæse overnight-priser direkte fra Yahoo Finance's hjemmeside.

Metode 1 (let):   requests + parsing af __NEXT_DATA__ JSON embeddet i HTML
Metode 2 (tung):  Playwright (rigtig browser, kræver: pip install playwright
                  + playwright install chromium)

Kør i overnight-vinduet: 02:00–10:00 dansk tid (søn–tor nat).

    pip install requests beautifulsoup4 playwright
    playwright install chromium
    python test_yahoo_scrape.py
"""

import json
import re
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------

TICKERS = ["AAPL", "NVDA", "MSFT", "TSLA", "SPY", "AMZN", "SNDK"]

ET_ZONE  = ZoneInfo("America/New_York")
CPH_ZONE = ZoneInfo("Europe/Copenhagen")

# Realistiske browser-headers — kritisk for at undgå blokering
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "da-DK,da;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT":             "1",
    "Connection":      "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

# ---------------------------------------------------------------------------
# Hjælpefunktioner
# ---------------------------------------------------------------------------

def _now_et()  -> datetime: return datetime.now(ET_ZONE)
def _now_cph() -> datetime: return datetime.now(CPH_ZONE)

def _is_overnight_et(dt: datetime) -> bool:
    return dt.hour >= 20 or dt.hour < 4

def _fmt_et(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S ET")

def _print_header(title: str) -> None:
    print(); print("=" * 70); print(f"  {title}"); print("=" * 70)

def _ok(msg):   print(f"  ✅  {msg}")
def _fail(msg): print(f"  ❌  {msg}")
def _info(msg): print(f"  ℹ️   {msg}")
def _moon(msg): print(f"  🌙  {msg}")


# ---------------------------------------------------------------------------
# Metode 1: requests + __NEXT_DATA__ JSON-parsing
# ---------------------------------------------------------------------------

def _fetch_nextdata(ticker: str, session: requests.Session) -> dict | None:
    """
    Henter Yahoo Finance-siden og udtrækker __NEXT_DATA__ JSON-blob.
    Returnerer den parsede dict eller None ved fejl.
    """
    url = f"https://finance.yahoo.com/quote/{ticker}/"
    try:
        r = session.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
    except requests.RequestException as e:
        _fail(f"{ticker}: HTTP-fejl — {e}")
        return None

    soup = BeautifulSoup(r.text, "html.parser")
    tag  = soup.find("script", {"id": "__NEXT_DATA__"})

    if not tag or not tag.string:
        # Prøv regex som fallback (Yahoo ændrer sommetider struktur)
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                      r.text, re.DOTALL)
        if not m:
            _fail(f"{ticker}: __NEXT_DATA__ ikke fundet i HTML")
            return None
        raw = m.group(1)
    else:
        raw = tag.string

    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        _fail(f"{ticker}: Kunne ikke parse __NEXT_DATA__ JSON — {e}")
        return None


def _extract_price_info(data: dict) -> dict:
    """
    Navigerer __NEXT_DATA__-strukturen og udtrækker prisfelter.
    Yahoo Finance ændrer indimellem sti — vi prøver flere.
    """
    found = {}

    # Sti 1: quoteType / price via QuoteSummaryStore (klassisk)
    try:
        stores = (data["props"]["pageProps"]
                     ["financialData"]
                     .get("QuoteSummaryStore", {}))
        price_block = stores.get("price", {})
        if price_block:
            found.update({
                "regularMarketPrice":     price_block.get("regularMarketPrice", {}).get("raw"),
                "postMarketPrice":        price_block.get("postMarketPrice", {}).get("raw"),
                "preMarketPrice":         price_block.get("preMarketPrice", {}).get("raw"),
                "marketState":            price_block.get("marketState"),
                "regularMarketTime":      price_block.get("regularMarketTime", {}).get("raw"),
                "postMarketTime":         price_block.get("postMarketTime", {}).get("raw"),
            })
    except (KeyError, TypeError):
        pass

    # Sti 2: streamingData / quote (nyere Next.js-struktur)
    try:
        page_props = data["props"]["pageProps"]
        for key in ("streamData", "initialData", "quoteData"):
            block = page_props.get(key, {})
            if isinstance(block, dict):
                for sub_key in ("quote", "summaryData", "price"):
                    sub = block.get(sub_key, {})
                    if isinstance(sub, dict) and sub.get("regularMarketPrice"):
                        found.setdefault("regularMarketPrice", sub.get("regularMarketPrice"))
                        found.setdefault("marketState",        sub.get("marketState"))
                        found.setdefault("postMarketPrice",    sub.get("postMarketPrice"))
                        found.setdefault("preMarketPrice",     sub.get("preMarketPrice"))
    except (KeyError, TypeError):
        pass

    # Sti 3: Bred søgning efter marketState og nøglefelter i hele JSON
    if not found.get("regularMarketPrice"):
        raw_str = json.dumps(data)
        for field in ("regularMarketPrice", "postMarketPrice", "preMarketPrice",
                      "marketState", "overnightPrice", "extendedMarketPrice"):
            m = re.search(rf'"{field}"\s*:\s*(\{{[^}}]+\}}|"[^"]*"|[\d.]+)', raw_str)
            if m and field not in found:
                val_str = m.group(1)
                try:
                    parsed = json.loads(val_str)
                    found[field] = parsed.get("raw", parsed) if isinstance(parsed, dict) else parsed
                except Exception:
                    found[field] = val_str

    return found


def test_nextdata_scrape(tickers: list[str]) -> dict:
    _print_header("METODE 1: requests + __NEXT_DATA__ JSON (ingen browser)")
    print("  Henter Yahoo Finance-siden og parser den indlejrede JSON-blob.")
    print("  Leder efter alle pris-felter inkl. eventuelle overnight-felter.\n")

    session = requests.Session()
    results = {}
    now_et  = _now_et()

    for ticker in tickers:
        data = _fetch_nextdata(ticker, session)
        if data is None:
            results[ticker] = {"ok": False}
            time.sleep(1)
            continue

        prices = _extract_price_info(data)

        # Vurder markedstilstand
        state     = prices.get("marketState", "?")
        reg_price = prices.get("regularMarketPrice")
        post_price = prices.get("postMarketPrice")
        pre_price  = prices.get("preMarketPrice")
        on_price   = prices.get("overnightPrice") or prices.get("extendedMarketPrice")

        in_overnight = _is_overnight_et(now_et)
        has_on_data  = bool(on_price)
        has_post     = bool(post_price)

        print(f"  ── {ticker} ──")
        print(f"     marketState:          {state}")
        print(f"     regularMarketPrice:   {reg_price}")
        if post_price:  print(f"     postMarketPrice:      {post_price}")
        if pre_price:   print(f"     preMarketPrice:       {pre_price}")
        if on_price:
            _moon(f"overnightPrice:       {on_price}  ← OVERNIGHT FELT FUNDET!")

        # Hvis vi er i overnight-sessionen, er postMarketPrice potentielt BOATS
        if in_overnight and post_price:
            _moon(f"{ticker}: postMarketPrice={post_price} under overnight-session — kan være BOATS-pris")
        elif in_overnight and not post_price and not on_price:
            _info(f"{ticker}: Ingen post/overnight pris i overnight-sessionen")
        elif not in_overnight:
            _info(f"{ticker}: Kørte uden for overnight-sessionen")

        results[ticker] = {
            "ok":                   True,
            "marketState":          state,
            "regularMarketPrice":   reg_price,
            "postMarketPrice":      post_price,
            "preMarketPrice":       pre_price,
            "overnightPrice":       on_price,
        }

        time.sleep(1.2)  # vær høflig mod Yahoo's servere

    return results


# ---------------------------------------------------------------------------
# Metode 2: Playwright (rigtig browser)
# ---------------------------------------------------------------------------

def _check_playwright() -> bool:
    try:
        import playwright  # noqa: F401
        return True
    except ImportError:
        return False


def test_playwright_scrape(tickers: list[str]) -> dict:
    _print_header("METODE 2: Playwright (rigtig browser med JavaScript)")
    print("  Starter Chromium, indlæser Yahoo Finance og aflæser prisen direkte.")
    print("  Kræver: pip install playwright && playwright install chromium\n")

    if not _check_playwright():
        _fail("Playwright ikke installeret.")
        _info("Kør: pip install playwright && playwright install chromium")
        return {}

    from playwright.sync_api import sync_playwright

    results = {}
    now_et  = _now_et()

    # CSS-selektorer Yahoo Finance bruger (kan ændre sig ved redesign)
    # Vi prøver flere kendte selektorer og tager den første der virker
    PRICE_SELECTORS = [
        '[data-testid="qsp-price"]',
        'fin-streamer[data-field="regularMarketPrice"]',
        '[data-field="regularMarketPrice"]',
        '.livePrice span',
        'span[data-reactid*="price"]',
    ]
    STATE_SELECTORS = [
        '[data-testid="qsp-market-notice"]',
        '[data-testid="marketStatus"]',
        '.marketStatus',
    ]
    POST_SELECTORS = [
        '[data-testid="qsp-post-price"]',
        'fin-streamer[data-field="postMarketPrice"]',
        '[data-field="postMarketPrice"]',
    ]

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx     = browser.new_context(
            user_agent=HEADERS["User-Agent"],
            locale="da-DK",
            viewport={"width": 1280, "height": 800},
        )
        page = ctx.new_page()
        # Blokér reklamer og tracking for at gøre det hurtigere
        page.route("**/(googlesyndication|doubleclick|analytics|adservice)/**",
                   lambda r: r.abort())

        for ticker in tickers:
            url = f"https://finance.yahoo.com/quote/{ticker}/"
            print(f"  ── {ticker} — indlæser {url}")

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                # Vent på at prisen dukker op
                page.wait_for_timeout(3_000)

                # Acceptér evt. cookie-banner
                for btn_text in ("Accept all", "Accepter alle", "Accept All"):
                    try:
                        page.get_by_text(btn_text, exact=True).first.click(timeout=2_000)
                        page.wait_for_timeout(1_000)
                        break
                    except Exception:
                        pass

                # Hent pris
                reg_price = None
                for sel in PRICE_SELECTORS:
                    try:
                        el = page.query_selector(sel)
                        if el:
                            reg_price = el.inner_text().strip()
                            break
                    except Exception:
                        pass

                # Hent post/overnight pris
                post_price = None
                for sel in POST_SELECTORS:
                    try:
                        el = page.query_selector(sel)
                        if el:
                            post_price = el.inner_text().strip()
                            break
                    except Exception:
                        pass

                # Hent markedstilstand
                market_state = None
                for sel in STATE_SELECTORS:
                    try:
                        el = page.query_selector(sel)
                        if el:
                            market_state = el.inner_text().strip()
                            break
                    except Exception:
                        pass

                # Hent hele page-titlen for kontekst
                title = page.title()

                in_overnight = _is_overnight_et(now_et)

                print(f"     Sidetitel:      {title[:60]}")
                print(f"     Markedstatus:   {market_state or '(ikke fundet)'}")
                print(f"     Regular pris:   {reg_price or '(ikke fundet)'}")
                if post_price:
                    if in_overnight:
                        _moon(f"Post/overnight pris: {post_price}  ← mulig BOATS-pris!")
                    else:
                        print(f"     Post pris:      {post_price}")
                else:
                    if in_overnight:
                        _info(f"{ticker}: Ingen post/overnight pris fundet på siden")

                results[ticker] = {
                    "ok":           True,
                    "regularPrice": reg_price,
                    "postPrice":    post_price,
                    "marketState":  market_state,
                    "title":        title,
                }

            except Exception as e:
                _fail(f"{ticker}: {e}")
                results[ticker] = {"ok": False, "error": str(e)}

            time.sleep(1.5)

        browser.close()

    return results


# ---------------------------------------------------------------------------
# Metode 3: Yahoo Finance undocumented v11 quote endpoint
# (nyere endpoint brugt af deres eget website)
# ---------------------------------------------------------------------------

def test_yahoo_v11(tickers: list[str]) -> dict:
    _print_header("METODE 3: Yahoo Finance v11/quoteSummary (intern API)")
    print("  Prøver Yahoo's nyere interne API-endpoint — kan indeholde")
    print("  overnight-felter som ikke er eksponeret i v7/v8.\n")

    results  = {}
    session  = requests.Session()
    now_et   = _now_et()
    in_on    = _is_overnight_et(now_et)

    MODULES = "price,summaryDetail,marketData"

    for ticker in tickers[:4]:
        url = f"https://query1.finance.yahoo.com/v11/finance/quoteSummary/{ticker}"
        params = {
            "modules":              MODULES,
            "corsDomain":          "finance.yahoo.com",
            "formatted":           "true",
            "lang":                "en-US",
            "region":              "US",
        }
        try:
            r = session.get(url, params=params, headers=HEADERS, timeout=10)

            if r.status_code != 200:
                _fail(f"{ticker}: HTTP {r.status_code}")
                results[ticker] = {"ok": False, "error": f"HTTP {r.status_code}"}
                time.sleep(0.5)
                continue

            data   = r.json()
            result = (data.get("quoteSummary", {})
                          .get("result", [{}]) or [{}])[0]

            price_data = result.get("price", {})
            state      = price_data.get("marketState", "?")
            reg        = (price_data.get("regularMarketPrice") or {}).get("raw")
            post       = (price_data.get("postMarketPrice")    or {}).get("raw")
            pre        = (price_data.get("preMarketPrice")     or {}).get("raw")

            # Bred søgning efter eventuelle nye overnight-felter
            raw_str    = json.dumps(price_data)
            new_fields = {}
            for field in re.findall(r'"(\w*[Oo]vernight\w*|extended\w*Price)"', raw_str):
                m = re.search(rf'"{field}"\s*:\s*(\{{.*?\}}|\d+\.?\d*)', raw_str)
                if m:
                    try:
                        val = json.loads(m.group(1))
                        new_fields[field] = val.get("raw", val) if isinstance(val, dict) else val
                    except Exception:
                        pass

            print(f"  ── {ticker} ──")
            print(f"     marketState:          {state}")
            print(f"     regularMarketPrice:   {reg}")
            if post: print(f"     postMarketPrice:      {post}")
            if pre:  print(f"     preMarketPrice:       {pre}")
            for k, v in new_fields.items():
                _moon(f"NYT FELT — {k}: {v}")

            if in_on and post:
                _moon(f"{ticker}: postMarketPrice={post} i overnight-session — mulig BOATS")
            elif in_on and not post:
                _info(f"{ticker}: Ingen overnight/post pris fra v11")

            results[ticker] = {
                "ok": True, "marketState": state,
                "regularMarketPrice": reg, "postMarketPrice": post,
                "preMarketPrice": pre, "newFields": new_fields,
            }

        except Exception as e:
            _fail(f"{ticker}: {e}")
            results[ticker] = {"ok": False, "error": str(e)}

        time.sleep(0.8)

    return results


# ---------------------------------------------------------------------------
# Opsummering
# ---------------------------------------------------------------------------

def _print_summary(results: dict, now_et: datetime) -> None:
    _print_header("SAMLET OPSUMMERING")
    in_on = _is_overnight_et(now_et)

    print(f"\n  Tidspunkt: {_fmt_et(now_et)}")
    if in_on:
        print("  🌙 Kørte i OVERNIGHT-sessionen (20:00–04:00 ET) — optimalt!")
    else:
        print("  ☀️  Kørte UDEN FOR overnight-sessionen.")
        print("     Kør igen 02:00–10:00 dansk tid for at se live overnight-priser.")

    print()
    any_overnight = False
    for method, ticker_data in results.items():
        if not isinstance(ticker_data, dict):
            continue
        for ticker, r in ticker_data.items():
            if not isinstance(r, dict):
                continue
            if r.get("overnightPrice") or r.get("postPrice") or (
                in_on and r.get("postMarketPrice")
            ):
                any_overnight = True
                _moon(f"{method} → {ticker}: overnight-data tilgængeligt!")

    if not any_overnight:
        if in_on:
            print("  ❌ Ingen overnight-priser fundet — selv i overnight-sessionen.")
            print("     Yahoo Finance eksponerer tilsyneladende ikke BOATS-prisen")
            print("     via hverken HTML-embedding eller deres API-endpoints.")
        else:
            print("  ⏳ Kør scriptet i overnight-sessionen for endeligt svar.")

    print()
    print("  Næste skridt afhænger af resultatet:")
    print("  • Overnight-data fundet via Metode 1/3 → byg fetch-funktion med requests")
    print("  • Fundet via Metode 2 (Playwright) → kan integreres men er tungt")
    print("  • Ikke fundet → Yahoo eksponerer det endnu ikke via kode-tilgang")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║      YAHOO FINANCE SCRAPE TEST — Pluto Portefølje               ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    now_et = _now_et()
    print(f"\n  Starttidspunkt (DK): {_now_cph().strftime('%Y-%m-%d %H:%M:%S CEST')}")
    print(f"  Starttidspunkt (ET): {_fmt_et(now_et)}")
    print(f"  Overnight-session:   "
          f"{'🟢 AKTIV (20:00–04:00 ET)' if _is_overnight_et(now_et) else '🔴 INAKTIV'}")
    print(f"  Tickers:             {', '.join(TICKERS)}")

    all_results = {}
    all_results["Metode 1: __NEXT_DATA__"] = test_nextdata_scrape(TICKERS)
    all_results["Metode 2: Playwright"]    = test_playwright_scrape(TICKERS)
    all_results["Metode 3: v11 API"]       = test_yahoo_v11(TICKERS)

    _print_summary(all_results, now_et)

    output_file = "yahoo_scrape_results.json"
    try:
        def _clean(obj):
            if isinstance(obj, dict):  return {k: _clean(v) for k, v in obj.items()}
            if isinstance(obj, list):  return [_clean(i) for i in obj]
            if isinstance(obj, float): return round(obj, 6)
            return obj
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({"run_at_et": _fmt_et(now_et),
                       "results": _clean(all_results)},
                      f, ensure_ascii=False, indent=2)
        print(f"  Råresultater gemt i: {output_file}\n")
    except Exception as e:
        print(f"  (Kunne ikke gemme JSON: {e})\n")


if __name__ == "__main__":
    main()
# Pluto Portefølje — Kontekstfil til AI-samarbejde

Upload denne fil **sammen med den fil du vil arbejde på** når du starter en ny chat med Claude.
Skriv f.eks.: *"Her er min kontekstfil og den fil jeg vil ændre i. Jeg vil gerne [beskriv hvad du vil]."*

---

## Hvad appen gør

Streamlit-app der viser porteføljeafkast for en Pluto-investeringskonto.
- Data indlæses fra en XLSX-fil eksporteret fra Pluto-appen
- Beregner **tidsvægtet afkast (TWR, Modified Dietz)** korrigeret for ind- og udbetalinger
- Viser live-kurser via yfinance
- Omregner alt til DKK via live FX-kurser
- Tilgås i en browser via `streamlit run app.py`

---

## Arkitekturprincip

Koden er opdelt i tre lag: et **kerne-lag** (100% Streamlit-frit), et **cache-lag** (kun `@st.cache_data`) og et **UI-lag** (Streamlit).
Kernen og cache-laget kan genbruges hvis appen flyttes væk fra Streamlit — kun `data/cached.py` og `views/` skal udskiftes.

```
Regler:
  1. Ingen st.-kald må forekomme udenfor views/-mappen
  2. data/fetch.py er ren Python — ingen Streamlit overhovedet
  3. Al @st.cache_data er samlet i data/cached.py
  4. Alle kald til cachede funktioner går via data.cached, ikke data.fetch
```

---

## Filstruktur og ansvar

### Indgangspunkt
| Fil | Ansvar |
|-----|--------|
| `app.py` | Tyndt indgangspunkt. Loader data fra XLSX, kalder live-pris override, håndterer routing til detalje-side, sætter tabs op og kalder views. Ca. 150 linjer. |

### Konfiguration og styling
| Fil | Ansvar |
|-----|--------|
| `config.py` | Konstanter: `PORTFOLIO_START`, `PERSISTENT_FILE`, `DEPOSIT_TIMES_FILE`, `DEFAULT_DEPOSIT_TIME`, `PLUTO_FX_SPREAD_RATE` (0,15%). Ingen Streamlit. |
| `styles.py` | CSS injiceret via `inject_styles()`. Klasser: `big-value`, `return-pos`, `return-neg`. |

### utils/ — Hjælpefunktioner (ingen Streamlit, ingen datahentning)
| Fil | Ansvar |
|-----|--------|
| `utils/formatting.py` | `_safe_float`, `_da_num`, `format_currency`, `format_big_number`, `format_quantity`, `_flatten_html`, `_CCY_SYMBOLS`, `color_change_str` |
| `utils/svg_charts.py` | `_make_sparkline_data_url`, `_make_volume_bar_html`, `_make_range_bar_html` — returnerer HTML/base64-strenge, ingen Streamlit |

### data/ — Datahentning og -håndtering
| Fil | Ansvar |
|-----|--------|
| `data/fetch.py` | **Ren Python, ingen Streamlit.** yfinance-kald: `fetch_price_history`, `fetch_price_history_intraday`, `fetch_live_quotes`, `fetch_live_fx_rates`, `fetch_ticker_meta`, `fetch_ticker_quote_info`, `fetch_intraday_sparklines`. Ingen dekoratorer. |
| `data/cached.py` | **Eneste sted med `@st.cache_data`.** Importerer alle funktioner fra `data.fetch` og eksponerer cachede versioner med samme navne og TTL-værdier. Skift framework: ret kun denne fil. |
| `data/deposits.py` | Gemmer og henter tidspunkter for indbetalinger i `deposit_times.json`. Funktioner: `load_deposit_times`, `save_deposit_times`, `_deposit_key`, `_time_to_frac`. Ingen Streamlit. |
| `data/market_status.py` | `get_us_market_status`, `get_eu_market_status`, `get_market_status_for_currency` — returnerer (kode, label, emoji, baggrundsfarve). Ingen Streamlit. |

### analytics/ — Kerneberegninger (100% Streamlit-fri)
| Fil | Ansvar |
|-----|--------|
| `analytics/portfolio.py` | `compute_portfolio_value_series`, `compute_portfolio_value_series_intraday`, `compute_deposits_dkk`, `cashflow_timeline`, `cumulative_return_series`, `slice_period`, `build_holdings_matrix`, `compute_ticker_lifecycle`. Importerer fra `data.cached`. |

### views/ — Alt Streamlit-UI
| Fil | Ansvar |
|-----|--------|
| `views/portfolio_overview.py` | `render_portfolio_overview(...)` — TWR-graf, aktietabel med live-kurser, nøgletal (urealiseret/realiseret P/L, omkostninger), "Alle beholdninger"-donut, Fordelinger-tabs (Sektorer/Aktivklasser/Valutaer/Regioner/Lande) samt nederste graf "Dynamik i porteføljeafkast". Dynamik-grafen kan vises dagligt, ugentligt, månedligt og årligt, enten som procent eller værdi, og kan grupperes efter Ingen gruppering, Aktivklasser eller Sektorer.. |
| `views/wallet.py` | `render_wallet(...)` — Kontante beholdninger, indbetalingstidspunkter med TWR-tidskorrigering, bevægelser pr. valuta. |
| `views/asset_detail.py` | `render_asset_detail(...)` — Detalje-side for enkelt aktie, åbnes via `?ticker=XXX` i URL. |
| `views/breakdown.py` | `render_breakdown(...)`, `render_drilldown(...)`, `_lighten_rgb(...)` — Genanvendelige fordeling-komponenter brugt af `portfolio_overview`. |
| `views/history.py` | `render_history(...)` — Historik-tab med expanders pr. ticker. Viser livscyklus, nøgletal (GAK, vægtet salgspris) og handelshistorik for både aktive og lukkede positioner. |

---

## Vigtige tekniske detaljer

**Afkastberegning:**
TWR beregnes med Modified Dietz-metoden. `compute_deposits_dkk` konverterer USD/EUR-indbetalinger til DKK og returnerer en cashflow-serie. `cashflow_fracs` er tidsfraktioner inden for dagen (fra `deposit_times.json`) der bruges til præcis TWR-vægtning.

**Live-priser:**
`fetch_live_quotes` returnerer en dict med nøglerne `live`, `prev_close`, `today_close`, `prev_prev_close` per ticker. `app.py` overskriver det seneste punkt i `total_value` og `stock_value` med live-beregnede værdier så alle tal på siden bruger samme priskilde.

**FX-spread:**
Plutos prismodel: 0,15% af handelsbeløb i USD/EUR (`PLUTO_FX_SPREAD_RATE`). Verificeret mod konkret handel. Kurtage hentes direkte fra XLSX-kolonnen "Commission (account currency)".

**Handelshistorik og GAK:**
`compute_ticker_lifecycle` i `analytics/portfolio.py` beregner livscyklus pr. ticker: investeret, realiseret, nuværende værdi og simpelt afkast (ikke TWR). GAK i aktivvaluta hentes fra `positions_df["Average entry price (asset currency)"]` for aktive positioner — Plutos egne præcise tal. Handelspriser beregnes via `FX rate`-kolonnen i Orders-fanen (bemærk lille 'r'). Afkast = (nuværende værdi + realiseret salgsbeløb − investeret) / investeret.

**Dynamik i porteføljeafkast:**
Portefølje-fanen har nederst en graf "Dynamik i porteføljeafkast", inspireret af Pluto-visningen. Grafen beregnes i analytics-laget og renderes i views-laget. Den understøtter periodisering efter Daglig, Ugentlig, Månedlig og Årlig samt visning som Procent eller Værdi.

Ved "Ingen gruppering" bruges compute_portfolio_return_dynamics(...), som beregner periodisk porteføljeafkast ud fra total_value, cashflows og cashflow_fracs.

Ved gruppering efter Aktivklasser eller Sektorer bruges compute_grouped_portfolio_return_dynamics(...). Funktionen beregner historisk værdi pr. ticker, summerer efter group_map og korrigerer for køb/salg i perioden. BUY behandles som kapital ind i gruppen, SELL som kapital ud af gruppen, så løbende handler ikke fejlagtigt vises som afkast.

Ved procentvisning i grupperet graf vises bidrag i procentpoint til porteføljeafkastet, ikke gruppens interne afkastprocent. Kontanter/øvrigt medtages som residualgruppe, så grupperede bidrag summerer mod samlet porteføljeafkast.

**Routing:**
Detalje-siden åbnes via query parameter `?ticker=XXX`. `app.py` tjekker `st.query_params.get("ticker")` og kalder `render_asset_detail` + `st.stop()` hvis sat.

**Cache-TTL-oversigt:**
| Funktion | TTL |
|---|---|
| `fetch_price_history` | 30 min |
| `fetch_price_history_intraday` | 5 min |
| `fetch_live_quotes` | 1 min |
| `fetch_live_fx_rates` | 1 min |
| `fetch_ticker_meta` | 24 timer |
| `fetch_ticker_quote_info` | 1 time |
| `fetch_intraday_sparklines` | 5 min |

**Data-filer der ikke er på GitHub:**
- `last_statement.xlsx` og `Account_statement.xlsx` — personlige finansdata, i `.gitignore`
- `deposit_times.json` — gemmes lokalt, er på GitHub da den ikke indeholder følsomme data

---

## Dataflow (forenklet)

```
XLSX-fil
 └── app.py (indlæsning + rensning)
     ├── analytics/portfolio.py (TWR, porteføljeværdi-serier, ticker-livscyklus, dynamik i porteføljeafkast)
     │   └── data/cached.py (cachede yfinance-kald)
     │       └── data/fetch.py (rene yfinance-funktioner)
     └── views/
         ├── portfolio_overview.py ← tab 1, inkl. TWR-graf, beholdninger, fordelinger og dynamik-graf
         ├── wallet.py ← tab 2
         ├── history.py ← tab 3
         └── asset_detail.py ← detalje-side (?ticker=)
```

---

## Sådan starter du appen

```
streamlit run app.py
```
Eller dobbeltklik på `start_app_claude.bat` (Windows).

---

## GitHub

Repository: `https://github.com/1982Strand/Pluto` (privat)

**Arbejdsflow efter ændringer:**
```
git add .
git commit -m "Kort beskrivelse af hvad du ændrede"
git push
```

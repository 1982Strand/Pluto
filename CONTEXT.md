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

Koden er opdelt i et **kerne-lag** (Streamlit-frit) og et **UI-lag** (Streamlit).
Kernen kan genbruges hvis appen på et tidspunkt flyttes væk fra Streamlit.

```
Regel: Ingen st.-kald må forekomme udenfor views/-mappen
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

### data/ — Datahentning og -håndtering (Streamlit kun via @st.cache_data)
| Fil | Ansvar |
|-----|--------|
| `data/fetch.py` | yfinance-kald: `fetch_price_history`, `fetch_price_history_intraday`, `fetch_live_quotes`, `fetch_live_fx_rates`, `fetch_ticker_meta`, `fetch_intraday_sparklines`. Alle cached med `@st.cache_data`. |
| `data/deposits.py` | Gemmer og henter tidspunkter for indbetalinger i `deposit_times.json`. Funktioner: `load_deposit_times`, `save_deposit_times`, `_deposit_key`, `_time_to_frac` |
| `data/market_status.py` | `get_us_market_status`, `get_eu_market_status`, `get_market_status_for_currency` — returnerer (kode, label, emoji, baggrundsfarve) |

### analytics/ — Kerneberegninger (100% Streamlit-fri)
| Fil | Ansvar |
|-----|--------|
| `analytics/portfolio.py` | `compute_portfolio_value_series`, `compute_portfolio_value_series_intraday`, `compute_deposits_dkk`, `cashflow_timeline`, `cumulative_return_series`, `slice_period`, `build_holdings_matrix` |

### views/ — Alt Streamlit-UI
| Fil | Ansvar |
|-----|--------|
| `views/portfolio_overview.py` | `render_portfolio_overview(...)` — TWR-graf, aktietabel med live-kurser, "Alle beholdninger"-donut, Fordelinger-tabs (Sektorer/Aktivklasser/Valutaer/Regioner/Lande) |
| `views/wallet.py` | `render_wallet(...)` — Kontante beholdninger, indbetalingstidspunkter, bevægelser |
| `views/asset_detail.py` | `render_asset_detail(...)` — Detalje-side for enkelt aktie, åbnes via `?ticker=XXX` i URL |
| `views/breakdown.py` | `render_breakdown(...)`, `render_drilldown(...)`, `_lighten_rgb(...)` — Genanvendelige fordeling-komponenter brugt af portfolio_overview |

---

## Vigtige tekniske detaljer

**Afkastberegning:**
TWR beregnes med Modified Dietz-metoden. `compute_deposits_dkk` konverterer USD/EUR-indbetalinger til DKK og returnerer en cashflow-serie. `cashflow_fracs` er tidsfraktioner inden for dagen (fra `deposit_times.json`) der bruges til præcis TWR-vægtning.

**Live-priser:**
`fetch_live_quotes` returnerer en dict med nøglerne `live`, `prev_close`, `today_close`, `prev_prev_close` per ticker. `app.py` overskriver det seneste punkt i `total_value` og `stock_value` med live-beregnede værdier så alle tal på siden bruger samme priskilde.

**FX-spread:**
Plutos prismodel: 0,15% af handelsbeløb i USD/EUR (`PLUTO_FX_SPREAD_RATE`). Verificeret mod konkret handel. Kurtage hentes direkte fra XLSX-kolonnen "Commission (account currency)".

**Routing:**
Detalje-siden åbnes via query parameter `?ticker=XXX`. `app.py` tjekker `st.query_params.get("ticker")` og kalder `render_asset_detail` + `st.stop()` hvis sat.

**Data-filer der ikke er på GitHub:**
- `last_statement.xlsx` og `Account_statement.xlsx` — personlige finansdata, i `.gitignore`
- `deposit_times.json` — gemmes lokalt, er på GitHub da den ikke indeholder følsomme data

---

## Dataflow (forenklet)

```
XLSX-fil
  └── app.py (indlæsning + rensning)
        ├── analytics/portfolio.py (TWR, porteføljeværdi-serier)
        ├── data/fetch.py (live-kurser, FX, meta)
        └── views/
              ├── portfolio_overview.py  ← tab 1
              ├── wallet.py              ← tab 2
              └── asset_detail.py        ← detalje-side (?ticker=)
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

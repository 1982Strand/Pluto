import pandas as pd

# -------------------- KONFIGURATION --------------------

PORTFOLIO_START = pd.Timestamp("2026-04-01")
PERSISTENT_FILE = "last_statement.xlsx"
DEPOSIT_TIMES_FILE = "deposit_times.json"
DEFAULT_DEPOSIT_TIME = "09:00"  # CET — typisk dansk bank-overførsels-morgen

# Plutos faste prismodel (verificeret mod app-screenshots):
PLUTO_FX_SPREAD_RATE = 0.0015        # 0,15% af konverteret beløb (Vekselomkostninger)
# Kurtage 0,10% pr. handel — tages direkte fra XLSX-kolonnen "Commission (account currency)"
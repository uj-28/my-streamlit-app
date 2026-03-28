from datetime import time
from pathlib import Path
from zoneinfo import ZoneInfo


DAILY_INTERVAL = "1d"
LOOKBACK_PERIOD = "max"
INDIA_TZ = ZoneInfo("Asia/Kolkata")
NSE_MARKET_CLOSE = time(15, 30)
NSE_INDEX_BASE_URL = "https://archives.nseindia.com/content/indices/"

YF_BATCH_SIZE = 120
YF_MAX_WORKERS = 4
CACHE_INTERVALS = {"1d", "1wk", "1mo"}

CACHE_DIR = Path(__file__).resolve().parent / "data_cache"
AI_MODEL_DIR = CACHE_DIR / "models"

UNIVERSE_OPTIONS = [
    "NIFTY 50",
    "NIFTY 100",
    "NIFTY 500",
    "NIFTY IT",
    "NIFTY MIDCAP 50",
    "NIFTY SMALLCAP 50",
    "NIFTY 750",
    "BANK NIFTY",
    "SENSEX 30",
]

SYMBOL_COLUMN_CANDIDATES = ("symbol", "ticker", "stock", "scrip", "instrument")

TIMEFRAME_OPTIONS = {
    "1D": {"interval": "1d", "period": "max", "resample": None, "label": "1D"},
    "Weekly": {"interval": "1wk", "period": "max", "resample": None, "label": "1W"},
    "1 Month": {"interval": "1mo", "period": "max", "resample": None, "label": "1M"},
    "6 Month": {"interval": "1d", "period": "max", "resample": "6M", "label": "6M"},
    "12 Month": {"interval": "1d", "period": "max", "resample": "12M", "label": "12M"},
}

# Banner demo data (replace with live values later if needed)
MARKET_DATA = {
    "NIFTY 50": {"price": "24315.95", "change": "+1.2%", "up": True},
    "BANKNIFTY": {"price": "52187.40", "change": "+0.8%", "up": True},
    "SENSEX": {"price": "80248.65", "change": "-0.4%", "up": False},
    "INDIA VIX": {"price": "13.82", "change": "-2.1%", "up": False},
}

MARKET_TICKERS = {
    "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "SENSEX": "^BSESN",
    "INDIA VIX": "^INDIAVIX",
}


# Layout constants for banner
BANNER_H = 150
IFRAME_PAD = 100
PANEL_TOP = 12
PANEL_HEIGHT = 110
PANEL_TITLE_Y = 30
PANEL_DIVIDER_Y = 40
COL_HEADER_Y = 52
ROW_Y = [68, 78, 88, 98]
LIVE_Y = 115

NSE_INDEX_CSV_FILES = {
    "NIFTY 50": "ind_nifty50list.csv",
    "NIFTY 100": "ind_nifty100list.csv",
    "NIFTY 500": "ind_nifty500list.csv",
    "NIFTY IT": "ind_niftyitlist.csv",
    "NIFTY MIDCAP 50": "ind_niftymidcap50list.csv",
    "NIFTY SMALLCAP 50": "ind_niftysmallcap50list.csv",
    "NIFTY 750": "ind_nifty750list.csv",
    "BANK NIFTY": "ind_niftybanklist.csv",
}


# Fallback lists (Yahoo Finance symbols) used when live index files are unavailable.
NIFTY_50_FALLBACK = [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "ITC.NS",
    "JIOFIN.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHRIRAMFIN.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
]

BANK_NIFTY_FALLBACK = [
    "AUBANK.NS",
    "AXISBANK.NS",
    "BANDHANBNK.NS",
    "BANKBARODA.NS",
    "CANBK.NS",
    "FEDERALBNK.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "IDFCFIRSTB.NS",
    "INDUSINDBK.NS",
    "KOTAKBANK.NS",
    "PNB.NS",
    "SBIN.NS",
]

SENSEX_30_FALLBACK = [
    "ADANIPORTS.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BHARTIARTL.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "ITC.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBIN.NS",
    "SUNPHARMA.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
]

NIFTY_IT_FALLBACK = [
    "HCLTECH.NS",
    "INFY.NS",
    "TECHM.NS",
    "WIPRO.NS",
    "LT.NS",
    "BURBK.NS",
    "COFORGE.NS",
    "HEXAWARE.NS",
    "MPHASIS.NS",
    "PERSISTENT.NS",
]

NIFTY_MIDCAP_50_FALLBACK = [
    "AUROPHARMA.NS",
    "BAJAJFINSV.NS",
    "BOSCHLTD.NS",
    "BRITANNIA.NS",
    "CHOLAFIN.NS",
    "DALBHARAT.NS",
    "DRREDDY.NS",
    "ESCORTS.NS",
    "EXIDEIND.NS",
    "HAVELLS.NS",
    "HDFCAMC.NS",
    "HDFCLIFE.NS",
    "HDFC.NS",
    "ICICIPRULI.NS",
    "INDIGO.NS",
    "JSWSTEEL.NS",
    "JUSTDIAL.NS",
    "MARICO.NS",
    "MAXHEALTH.NS",
    "MCDOWELL-N.NS",
    "MOTHERSON.NS",
    "NESTLEIND.NS",
    "PGHH.NS",
    "PNB.NS",
    "SAGE.NS",
    "SHRIRAMFIN.NS",
    "SRF.NS",
    "SUVIDHAA.NS",
    "SUZLON.NS",
    "TATACONSUM.NS",
]

NIFTY_SMALLCAP_50_FALLBACK = [
    "ACC.NS",
    "ADANIGREEN.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "BAJAJFINSV.NS",
    "BAJAJTRANS.NS",
    "BELLTEXTILES.NS",
    "BEL.NS",
    "BHARATGRID.NS",
    "BHEL.NS",
    "BIGBLOC.NS",
    "BLUEDART.NS",
    "BOMDYEING.NS",
    "BPCL.NS",
    "CUMMINSIND.NS",
    "DEEPAKFERT.NS",
    "DHAYMERS.NS",
    "EMCURE.NS",
    "ENDURANCE.NS",
    "EVEREADY.NS",
    "FAME.NS",
    "FINEORG.NS",
    "FLUOROCHEM.NS",
    "GAIL.NS",
    "GBPIL.NS",
    "GETFRESH.NS",
    "GLAXO.NS",
    "GMRINFRA.NS",
    "GODREJAGRO.NS",
    "GRINDWELL.NS",
]

NIFTY_750_FALLBACK = [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "ITC.NS",
    "JIOFIN.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHRIRAMFIN.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "WIPRO.NS",
]


# Data quality + model settings
MIN_AVG_VOLUME = 50000
MIN_HISTORY_BARS = 120
MAX_NULL_PCT = 0.10

AI_MIN_TRAIN_ROWS = 200
AI_WALK_FORWARD_SPLITS = 5
AI_CONFIDENCE_DEFAULT = 0.65


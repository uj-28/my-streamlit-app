from datetime import datetime, time
from io import StringIO
import difflib
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import yfinance as yf
import numpy as np
import xgboost as xgb
import streamlit.components.v1 as components


DAILY_INTERVAL = "1d"
LOOKBACK_PERIOD = "max"
INDIA_TZ = ZoneInfo("Asia/Kolkata")
NSE_MARKET_CLOSE = time(15, 30)
NSE_INDEX_BASE_URL = "https://archives.nseindia.com/content/indices/"
YF_BATCH_SIZE = 120
YF_MAX_WORKERS = 4

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


# ── Layout constants — change BANNER_H to resize everything ──────────────────
BANNER_H = 150
IFRAME_PAD = 100
PANEL_TOP = 12
PANEL_HEIGHT = 110
PANEL_TITLE_Y = 30
PANEL_DIVIDER_Y = 40
COL_HEADER_Y = 52
ROW_Y = [68, 78, 88, 98]
LIVE_Y = 115



def build_ticker_rows(data: dict) -> str:
    rows = ""
    for (symbol, info), y in zip(data.items(), ROW_Y):
        css = "ticker-up" if info["up"] else "ticker-dn"
        arrow = "&#9650;" if info["up"] else "&#9660;"
        rows += (
            f'<text x="432" y="{y}" font-family="\'Courier New\',monospace" '
            f'font-size="8" font-weight="700" fill="#0c447c">{symbol}</text>'
            f'<text x="530" y="{y}" font-family="\'Courier New\',monospace" '
            f'font-size="8" fill="#444">{info["price"]}</text>'
            f'<text x="622" y="{y}" font-family="\'Courier New\',monospace" '
            f'font-size="8" class="{css}">{arrow} {info["change"]}</text>'
        )
    return rows


def stock_scanner_banner(data: dict = MARKET_DATA) -> None:
    rows_svg = build_ticker_rows(data)
    h = BANNER_H
    pt = PANEL_TOP
    ph = PANEL_HEIGHT
    clip_top = pt
    clip_h = h - pt * 3

    html = f"""<div style="width:100%;margin:0;padding:0;line-height:0;overflow:hidden;">
<svg width="100%" viewBox="0 0 680 {h}"
     xmlns="http://www.w3.org/2000/svg" style="display:block;">
<defs>
  <linearGradient id="bg" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="#dbeeff"/>
    <stop offset="100%" stop-color="#eef6ff"/>
  </linearGradient>
  <linearGradient id="chartbg" x1="0" y1="0" x2="0" y2="1">
    <stop offset="0%" stop-color="#1a6fc4" stop-opacity="0.09"/>
    <stop offset="100%" stop-color="#1a6fc4" stop-opacity="0"/>
  </linearGradient>
  <linearGradient id="linefade" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="#1a6fc4" stop-opacity="0"/>
    <stop offset="25%" stop-color="#1a6fc4" stop-opacity="1"/>
    <stop offset="100%" stop-color="#1a6fc4" stop-opacity="1"/>
  </linearGradient>
  <clipPath id="chartclip">
    <rect x="35" y="{clip_top}" width="1300" height="{clip_h}"/>
  </clipPath>
  <style>
    @keyframes scan  {{ 0%{{transform:translateX(0)}} 100%{{transform:translateX(368px)}} }}
    @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.60}} }}
    .scanline  {{ animation: scan  3s linear        infinite; }}
    .dot-pulse {{ animation: pulse 1.5s ease-in-out  infinite; }}
    .ticker-up {{ fill: #1aaa6f; }}
    .ticker-dn {{ fill: #e24b4a; }}
  </style>
</defs>

<rect width="700" height="{h}" rx="12" fill="url(#bg)"/>

<line x1="20" y1="{clip_top + clip_h*1//5}" x2="350" y2="{clip_top + clip_h*1//5}"
      stroke="#1a6fc4" stroke-width="0.4" stroke-opacity="0.18"/>
<line x1="20" y1="{clip_top + clip_h*2//5}" x2="350" y2="{clip_top + clip_h*2//5}"
      stroke="#1a6fc4" stroke-width="0.4" stroke-opacity="0.18"/>
<line x1="20" y1="{clip_top + clip_h*3//5}" x2="350" y2="{clip_top + clip_h*3//5}"
      stroke="#1a6fc4" stroke-width="0.4" stroke-opacity="0.18"/>
<line x1="20" y1="{clip_top + clip_h*4//5}" x2="350" y2="{clip_top + clip_h*4//5}"
      stroke="#1a6fc4" stroke-width="0.4" stroke-opacity="0.13"/>

<path d="M55,{h-30} L90,{h-60} L125,{h-50} L160,{h-80}
         L195,{h-75} L230,{h-96} L265,{h-80} L300,{h-103}
         L335,{h-88} L370,{h-108} L396,{h-118}
         L396,{h-28} L55,{h-28}Z"
      fill="url(#chartbg)" clip-path="url(#chartclip)"/>

<polyline
  points="55,{h-30} 90,{h-60} 125,{h-50} 160,{h-80}
          195,{h-75} 230,{h-96} 265,{h-80} 300,{h-103}
          335,{h-88} 370,{h-108} 396,{h-118}"
  fill="none" stroke="url(#linefade)" stroke-width="2.2"
  stroke-linecap="round" stroke-linejoin="round"
  clip-path="url(#chartclip)"/>

<line x1="90"  y1="{h-69}" x2="90"  y2="{h-53}" stroke="#1aaa6f" stroke-width="1" stroke-opacity="0.7"/>
<rect x="85"   y="{h-66}"  width="10" height="9" rx="1" fill="#1aaa6f" opacity="0.8"/>
<line x1="125" y1="{h-63}" x2="125" y2="{h-45}" stroke="#e24b4a" stroke-width="1" stroke-opacity="0.7"/>
<rect x="120"  y="{h-61}"  width="10" height="11" rx="1" fill="#e24b4a" opacity="0.8"/>
<line x1="160" y1="{h-93}" x2="160" y2="{h-73}" stroke="#1aaa6f" stroke-width="1" stroke-opacity="0.7"/>
<rect x="155"  y="{h-91}"  width="10" height="12" rx="1" fill="#1aaa6f" opacity="0.8"/>
<line x1="195" y1="{h-83}" x2="195" y2="{h-64}" stroke="#e24b4a" stroke-width="1" stroke-opacity="0.7"/>
<rect x="190"  y="{h-80}"  width="10" height="10" rx="1" fill="#e24b4a" opacity="0.8"/>
<line x1="230" y1="{h-104}" x2="230" y2="{h-85}" stroke="#1aaa6f" stroke-width="1" stroke-opacity="0.7"/>
<rect x="225"  y="{h-101}"  width="10" height="11" rx="1" fill="#1aaa6f" opacity="0.8"/>
<line x1="265" y1="{h-88}"  x2="265" y2="{h-70}" stroke="#e24b4a" stroke-width="1" stroke-opacity="0.7"/>
<rect x="260"  y="{h-85}"   width="10" height="10" rx="1" fill="#e24b4a" opacity="0.8"/>
<line x1="300" y1="{h-111}" x2="300" y2="{h-93}" stroke="#1aaa6f" stroke-width="1" stroke-opacity="0.7"/>
<rect x="295"  y="{h-108}"  width="10" height="10" rx="1" fill="#1aaa6f" opacity="0.8"/>
<line x1="335" y1="{h-97}"  x2="335" y2="{h-79}" stroke="#e24b4a" stroke-width="1" stroke-opacity="0.7"/>
<rect x="330"  y="{h-94}"   width="10" height="10" rx="1" fill="#e24b4a" opacity="0.8"/>
<line x1="370" y1="{h-116}" x2="370" y2="{h-98}" stroke="#1aaa6f" stroke-width="1" stroke-opacity="0.7"/>
<rect x="365"  y="{h-113}"  width="10" height="10" rx="1" fill="#1aaa6f" opacity="0.8"/>

<circle cx="396" cy="{h-118}" r="4.5" fill="#1a6fc4" opacity="0.9" class="dot-pulse"/>
<circle cx="396" cy="{h-118}" r="8"   fill="none" stroke="#1a6fc4" stroke-width="1" opacity="0.35" class="dot-pulse"/>

<g clip-path="url(#chartclip)">
  <line class="scanline" x1="28" y1="{clip_top}" x2="28" y2="{clip_top + clip_h}"
        stroke="#1a6fc4" stroke-width="1.5" stroke-opacity="0.25"/>
  <rect class="scanline" x="20" y="{clip_top}" width="16" height="{clip_h}"
        fill="#1a6fc4" fill-opacity="0.05"/>
</g>

<rect x="413" y="{pt}" width="258" height="{ph}" rx="10"
      fill="#1a6fc4" fill-opacity="0.07"
      stroke="#1a6fc4" stroke-opacity="0.2" stroke-width="0.8"/>

<text x="428" y="{PANEL_TITLE_Y}"
      font-family="'Segoe UI',sans-serif"
      font-size="12" font-weight="700" fill="#0c447c" letter-spacing="0.5">
  STOCK SCANNER
</text>
<line x1="428" y1="{PANEL_DIVIDER_Y}" x2="659" y2="{PANEL_DIVIDER_Y}"
      stroke="#1a6fc4" stroke-width="0.6" stroke-opacity="0.35"/>

<text x="430" y="{COL_HEADER_Y}"
      font-family="'Segoe UI',sans-serif" font-size="8.5" fill="#1a6fc4" fill-opacity="0.55">INDEX</text>
<text x="528" y="{COL_HEADER_Y}"
      font-family="'Segoe UI',sans-serif" font-size="8.5" fill="#1a6fc4" fill-opacity="0.55">PRICE (&#x20B9;)</text>
<text x="622" y="{COL_HEADER_Y}"
      font-family="'Segoe UI',sans-serif" font-size="8.5" fill="#1a6fc4" fill-opacity="0.55">CHG%</text>

{rows_svg}

<line x1="428" y1="{LIVE_Y - 11}" x2="659" y2="{LIVE_Y - 11}"
      stroke="#1a6fc4" stroke-width="0.4" stroke-opacity="0.2"/>

<circle cx="430" cy="{LIVE_Y - 4}" r="4" fill="#1aaa6f" class="dot-pulse"/>
<text x="440" y="{LIVE_Y}"
      font-family="'Segoe UI',sans-serif" font-size="9" fill="#1aaa6f" font-weight="600">LIVE</text>
<text x="470" y="{LIVE_Y}"
      font-family="'Segoe UI',sans-serif" font-size="9" fill="#1a6fc4" fill-opacity="0.65">
  NSE / BSE &#183; Real-time
</text>
</svg>
</div>"""
    components.html(html, height=BANNER_H + IFRAME_PAD, scrolling=False)




@st.cache_data(ttl=300, show_spinner=False)
def fetch_market_data() -> dict:
    """Fetch near real-time index data from yfinance; fallback to static data on failure."""
    out = {}
    for label, ticker in MARKET_TICKERS.items():
        try:
            hist = yf.download(ticker, period="5d", interval="1d", progress=False, auto_adjust=False)
            hist = hist.dropna()
            if len(hist) < 2:
                raise ValueError("Not enough data")
            latest_raw = hist["Close"].iloc[-1]
            prev_raw = hist["Close"].iloc[-2]
            latest = float(latest_raw.iloc[0]) if isinstance(latest_raw, pd.Series) else float(latest_raw)
            prev = float(prev_raw.iloc[0]) if isinstance(prev_raw, pd.Series) else float(prev_raw)
            chg_pct = ((latest - prev) / prev) * 100.0 if prev != 0 else 0.0
            out[label] = {
                "price": f"{latest:.2f}",
                "change": f"{chg_pct:+.2f}%",
                "up": chg_pct >= 0,
            }
        except Exception:
            out[label] = MARKET_DATA.get(label, {"price": "0.00", "change": "0.00%", "up": True})
    return out

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


def apply_ui_style() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #f4f7fb 0%, #eef3f9 50%, #e9f0f8 100%);
            color: #1a1f2b;
        }
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1500px;
        }
        .hero {
            background: linear-gradient(120deg, #e9f5ff 0%, #d8ecff 60%, #c7e3ff 100%);
            border-radius: 20px;
            padding: 30px 40px;
            color: #0f172a;
            margin-bottom: 20px;
            box-shadow: 0 10px 24px rgba(15, 23, 42, 0.12);
            border: 1px solid rgba(59, 130, 246, 0.2);
        }
        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
            line-height: 1.2;
            font-family: "Trebuchet MS", "Verdana", sans-serif;
            font-weight: 700;
            text-shadow: none;
        }
        .hero p {
            margin: 10px 0 0;
            color: #1f2a44;
            font-size: 1.05rem;
            font-family: "Trebuchet MS", "Verdana", sans-serif;
            font-weight: 500;
        }
        .main .stMarkdown, .main .stMarkdown p, .main .stText, .main .stCaption, .main .stAlert,
        .main label, .main .stRadio label, .main .stCheckbox label {
            color: #1a1f2b;
        }
        .stSidebar, .stSidebar * {
            color: #1a1a2e;
        }
        div[data-testid="metric-container"] {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.08);
            border-radius: 15px;
            padding: 15px 20px;
            box-shadow: 0 6px 16px rgba(15, 23, 42, 0.08);
        }
        .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
            font-size: 1.1rem;
            font-weight: 700;
            color: #1f2a44;
        }
        .stTabs [data-baseweb="tab-list"] button {
            background: #f2f5fa;
            border-bottom: 3px solid transparent;
        }
        .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
            border-bottom: 3px solid #1d4ed8;
            background: #e6f0ff;
        }
        div[data-testid="stRadio"] > label {
            display: none;
        }
        div[data-testid="stRadio"] > div {
            background: #5bb3d1;
            border-radius: 8px;
            padding: 6px 10px;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 12px;
        }
        div[data-testid="stRadio"] label {
            color: #e9f7ff !important;
            font-weight: 700;
        }
        div[data-testid="stRadio"] input[type="radio"] {
            accent-color: #0f4c5c;
        }
        .stDataFrame {
            border: 1px solid rgba(15, 23, 42, 0.12) !important;
            border-radius: 10px !important;
        }
        .stDataFrame table, .stDataFrame th, .stDataFrame td {
            font-size: 1.1rem !important;
            text-align: center !important;
            vertical-align: middle !important;
        }
        div[data-testid="stDataFrame"] td div, div[data-testid="stDataFrame"] th div {
            justify-content: center !important;
        }
        div[data-testid="stDataFrame"] thead th {
            color: #2b3448 !important;
            font-weight: 700 !important;
        }
        .stDownloadButton button {
            background: linear-gradient(90deg, #2563eb 0%, #1d4ed8 100%);
            border: none;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            padding: 10px 20px;
        }
        div[data-testid="stTextInput"] input {
            border: 1px solid #000000 !important;
        }
        h1, h2, h3 {
            color: #0f172a;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_yf_cache() -> None:
    """Set local writable cache directory to avoid yfinance sqlite path issues."""
    cache_dir = Path(__file__).resolve().parent / ".yf_cache"
    cache_dir.mkdir(exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))


def fetch_csv_from_url(url: str) -> pd.DataFrame | None:
    """Fetch CSV with browser-like headers; return None on network/parse failure."""
    try:
        req = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "text/csv,*/*;q=0.9",
            },
        )
        with urlopen(req, timeout=20) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        return pd.read_csv(StringIO(raw))
    except (URLError, HTTPError, TimeoutError, ValueError):
        return None


def to_nse_ticker(symbol: str) -> str:
    sym = symbol.strip().upper()
    if not sym:
        return ""
    return sym if sym.endswith(".NS") else f"{sym}.NS"


def load_nse_index_constituents(universe: str) -> list[str]:
    csv_file = NSE_INDEX_CSV_FILES.get(universe)
    if not csv_file:
        return []

    df = fetch_csv_from_url(f"{NSE_INDEX_BASE_URL}{csv_file}")
    if df is None or df.empty:
        return []

    symbol_col = next((c for c in df.columns if str(c).strip().lower() == "symbol"), None)
    if symbol_col is None:
        return []

    out: list[str] = []
    for raw_symbol in df[symbol_col].astype(str):
        ticker = to_nse_ticker(raw_symbol)
        if ticker:
            out.append(ticker)
    return list(dict.fromkeys(out))


def load_symbols_from_file(uploaded_file) -> list[str]:
    """Load symbols from uploaded CSV/XLSX and normalize to NSE tickers."""
    if uploaded_file is None:
        return []

    filename = (uploaded_file.name or "").lower()
    try:
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except ImportError:
        st.error("Excel upload requires the 'openpyxl' package. Please upload a CSV or install openpyxl.")
        return []

    if df.empty:
        return []

    symbol_col = next(
        (c for c in df.columns if str(c).strip().lower() in SYMBOL_COLUMN_CANDIDATES),
        None,
    )
    if symbol_col is None:
        symbol_col = df.columns[0]

    out: list[str] = []
    for raw_symbol in df[symbol_col].astype(str):
        ticker = to_nse_ticker(raw_symbol)
        if ticker:
            out.append(ticker)
    return list(dict.fromkeys(out))


UNIVERSE_REFRESH_TTL_SECONDS = 30 * 24 * 60 * 60


@st.cache_data(ttl=UNIVERSE_REFRESH_TTL_SECONDS, show_spinner=False)
def resolve_universe(universe: str) -> tuple[list[str], str]:
    """Return tickers and source note for selected universe."""
    if universe in NSE_INDEX_CSV_FILES:
        symbols = load_nse_index_constituents(universe)
        if symbols:
            return symbols, "NSE live constituents"

    if universe == "NIFTY 50":
        return NIFTY_50_FALLBACK, "Fallback list"
    if universe == "NIFTY IT":
        return NIFTY_IT_FALLBACK, "Fallback list"
    if universe == "NIFTY MIDCAP 50":
        return NIFTY_MIDCAP_50_FALLBACK, "Fallback list"
    if universe == "NIFTY SMALLCAP 50":
        return NIFTY_SMALLCAP_50_FALLBACK, "Fallback list"
    if universe == "NIFTY 750":
        return NIFTY_750_FALLBACK, "Fallback list"
    if universe == "BANK NIFTY":
        return BANK_NIFTY_FALLBACK, "Fallback list"
    if universe == "SENSEX 30":
        return SENSEX_30_FALLBACK, "Fallback list"

    return [], "Unavailable (could not load index constituents)"


def resample_ohlcv(data: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLCV data to a higher timeframe."""
    if data.empty:
        return data
    agg = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    out = data.resample(rule, label="right", closed="right").agg(agg)
    return out.dropna()


@st.cache_data(ttl=900, show_spinner=False)
def fetch_bulk_history(
    symbols: tuple[str, ...],
    interval: str,
    period: str,
    start=None,
    end=None,
) -> pd.DataFrame:
    """Download OHLCV data for multiple symbols in one request."""
    if not symbols:
        return pd.DataFrame()
    symbols_list = list(symbols)

    def normalize_batch(df: pd.DataFrame, batch: list[str]) -> pd.DataFrame:
        if df.empty:
            return df
        if isinstance(df.columns, pd.MultiIndex):
            return df
        if len(batch) == 1:
            symbol = batch[0]
            df.columns = pd.MultiIndex.from_product([[symbol], df.columns])
        return df

    def fetch_batch(batch: list[str]) -> pd.DataFrame:
        try:
            if start is not None or end is not None:
                data = yf.download(
                    batch,
                    start=start,
                    end=end,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    actions=False,
                    threads=True,
                    progress=False,
                )
            else:
                data = yf.download(
                    batch,
                    period=period,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    actions=False,
                    threads=True,
                    progress=False,
                )
            return normalize_batch(data, batch)
        except Exception:
            return pd.DataFrame()

    batches = [symbols_list[i : i + YF_BATCH_SIZE] for i in range(0, len(symbols_list), YF_BATCH_SIZE)]
    if len(batches) == 1:
        return fetch_batch(batches[0])

    results: list[pd.DataFrame] = []
    max_workers = min(YF_MAX_WORKERS, len(batches))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_batch, batch) for batch in batches]
        for fut in as_completed(futures):
            df = fut.result()
            if not df.empty:
                results.append(df)

    if not results:
        return pd.DataFrame()
    if len(results) == 1:
        return results[0]
    return pd.concat(results, axis=1)


def ensure_series(obj: pd.Series | pd.DataFrame) -> pd.Series:
    """Normalize yfinance output to Series for indicator calculations."""
    if isinstance(obj, pd.DataFrame):
        if obj.shape[1] == 0:
            return pd.Series(index=obj.index, dtype="float64")
        return obj.iloc[:, 0]
    return obj


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    TradingView-style RSI:
    - Up/Down from close-to-close change
    - Wilder's RMA smoothing with SMA seed
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    def rma(series: pd.Series, length: int) -> pd.Series:
        out = pd.Series(index=series.index, dtype="float64")
        valid = series.dropna()
        if len(valid) < length:
            return out

        seed_idx = valid.index[length - 1]
        prev = float(valid.iloc[:length].mean())
        out.loc[seed_idx] = prev

        for idx in valid.index[length:]:
            curr = float(valid.loc[idx])
            prev = ((length - 1) * prev + curr) / length
            out.loc[idx] = prev
        return out

    avg_gain = rma(gain, period)
    avg_loss = rma(loss, period)
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


def get_signal_index(
    index: pd.Index,
    use_last_closed_candle: bool,
    timeframe_label: str,
) -> pd.Timestamp | None:
    """
    Pick candle used for signal generation.
    - If enabled, during market hours use previous fully closed daily candle.
    - Otherwise use latest available daily candle.
    """
    if len(index) == 0:
        return None

    latest = pd.Timestamp(index[-1])
    if not use_last_closed_candle or len(index) < 2:
        return latest

    now_ist = datetime.now(INDIA_TZ)
    latest_date = latest.date()
    if latest_date == now_ist.date() and now_ist.time() < NSE_MARKET_CLOSE:
        return pd.Timestamp(index[-2])
    return latest


def compute_supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr_period: int = 10,
    multiplier: float = 3.0,
) -> tuple[pd.Series, pd.Series]:
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()

    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    final_upper = upper_band.copy()
    final_lower = lower_band.copy()

    for i in range(1, len(close)):
        if upper_band.iloc[i] < final_upper.iloc[i - 1] or close.iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = upper_band.iloc[i]
        else:
            final_upper.iloc[i] = final_upper.iloc[i - 1]

        if lower_band.iloc[i] > final_lower.iloc[i - 1] or close.iloc[i - 1] < final_lower.iloc[i - 1]:
            final_lower.iloc[i] = lower_band.iloc[i]
        else:
            final_lower.iloc[i] = final_lower.iloc[i - 1]

    supertrend = pd.Series(index=close.index, dtype="float64")
    direction = pd.Series(index=close.index, dtype="int64")

    if len(close) == 0:
        return supertrend, direction

    supertrend.iloc[0] = final_upper.iloc[0]
    direction.iloc[0] = -1

    for i in range(1, len(close)):
        if supertrend.iloc[i - 1] == final_upper.iloc[i - 1]:
            if close.iloc[i] <= final_upper.iloc[i]:
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
        else:
            if close.iloc[i] >= final_lower.iloc[i]:
                supertrend.iloc[i] = final_lower.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = final_upper.iloc[i]
                direction.iloc[i] = -1

    return supertrend, direction


def compute_halftrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    amplitude: int = 2,
    channel_deviation: int = 2,
    atr_period: int = 100,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    HalfTrend indicator (Pine v6 reference):
    Returns: ht line, buy signal, sell signal, trend (0=up, 1=down).
    """
    n = len(close)
    ht = pd.Series(index=close.index, dtype="float64")
    buy_signal = pd.Series(False, index=close.index, dtype="bool")
    sell_signal = pd.Series(False, index=close.index, dtype="bool")
    trend = pd.Series(index=close.index, dtype="int64")

    if n == 0:
        return ht, buy_signal, sell_signal, trend

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()
    atr2 = atr / 2.0
    dev = channel_deviation * atr2

    high_price = high.rolling(window=amplitude, min_periods=1).max()
    low_price = low.rolling(window=amplitude, min_periods=1).min()
    high_ma = high.rolling(window=amplitude, min_periods=1).mean()
    low_ma = low.rolling(window=amplitude, min_periods=1).mean()

    trend_val = 0
    next_trend = 0
    max_low_price = float(low.shift(1).iloc[0]) if n > 0 else float(low.iloc[0])
    min_high_price = float(high.shift(1).iloc[0]) if n > 0 else float(high.iloc[0])
    up = float("nan")
    down = float("nan")
    prev_up = float("nan")
    prev_down = float("nan")

    for i in range(n):
        if next_trend == 1:
            max_low_price = max(float(low_price.iloc[i]), max_low_price)
            low_prev = float(low.shift(1).iloc[i]) if i > 0 else float(low.iloc[i])
            if float(high_ma.iloc[i]) < max_low_price and float(close.iloc[i]) < low_prev:
                trend_val = 1
                next_trend = 0
                min_high_price = float(high_price.iloc[i])
        else:
            min_high_price = min(float(high_price.iloc[i]), min_high_price)
            high_prev = float(high.shift(1).iloc[i]) if i > 0 else float(high.iloc[i])
            if float(low_ma.iloc[i]) > min_high_price and float(close.iloc[i]) > high_prev:
                trend_val = 0
                next_trend = 1
                max_low_price = float(low_price.iloc[i])

        if trend_val == 0:
            if i > 0 and trend.iloc[i - 1] != 0:
                up = prev_down if not pd.isna(prev_down) else down
                buy_signal.iloc[i] = True
            else:
                base_up = max_low_price
                up = base_up if pd.isna(prev_up) else max(base_up, prev_up)
            ht.iloc[i] = up
        else:
            if i > 0 and trend.iloc[i - 1] != 1:
                down = prev_up if not pd.isna(prev_up) else up
                sell_signal.iloc[i] = True
            else:
                base_down = min_high_price
                down = base_down if pd.isna(prev_down) else min(base_down, prev_down)
            ht.iloc[i] = down

        trend.iloc[i] = trend_val
        prev_up = up
        prev_down = down

    return ht, buy_signal, sell_signal, trend


def compute_feature_frame(
    data: pd.DataFrame,
    rsi_length: int,
    atr_period: int,
    atr_multiplier: float,
) -> pd.DataFrame:
    """Compute ML feature frame from OHLCV data."""
    close = ensure_series(data["Close"]).astype("float64")
    high = ensure_series(data["High"]).astype("float64")
    low = ensure_series(data["Low"]).astype("float64")
    volume = ensure_series(data["Volume"]).astype("float64")

    ohlc = pd.concat([high, low, close, volume], axis=1, keys=["High", "Low", "Close", "Volume"]).dropna()
    if ohlc.empty:
        return pd.DataFrame()

    high = ohlc["High"]
    low = ohlc["Low"]
    close = ohlc["Close"]
    volume = ohlc["Volume"]

    rsi = compute_rsi(close, period=rsi_length)
    ma50 = close.rolling(50, min_periods=1).mean()
    ma200 = close.rolling(200, min_periods=1).mean()
    dist_ma50 = (close - ma50) / ma50 * 100.0
    dist_ma200 = (close - ma200) / ma200 * 100.0
    vol_spike = volume / volume.rolling(20, min_periods=1).mean()

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / atr_period, adjust=False).mean()
    atr_volatility = (atr / close) * 100.0

    _, st_dir = compute_supertrend(
        high=high,
        low=low,
        close=close,
        atr_period=atr_period,
        multiplier=atr_multiplier,
    )

    features = pd.DataFrame(
        {
            "rsi": rsi,
            "dist_ma50": dist_ma50,
            "dist_ma200": dist_ma200,
            "vol_spike": vol_spike,
            "atr_volatility": atr_volatility,
            "supertrend_dir": st_dir,
        }
    )
    return features


def build_training_matrix(
    data: pd.DataFrame,
    rsi_length: int,
    atr_period: int,
    atr_multiplier: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build feature matrix and labels for next-day direction."""
    features = compute_feature_frame(
        data=data,
        rsi_length=rsi_length,
        atr_period=atr_period,
        atr_multiplier=atr_multiplier,
    )
    if features.empty:
        return pd.DataFrame(), pd.Series(dtype="int64")

    close = ensure_series(data["Close"]).astype("float64").reindex(features.index)
    next_return = close.pct_change().shift(-1)
    labels = (next_return > 0).astype("int64")

    frame = features.copy()
    frame["label"] = labels
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()

    if frame.empty:
        return pd.DataFrame(), pd.Series(dtype="int64")

    y = frame.pop("label")
    return frame, y


def train_xgb_model(X: pd.DataFrame, y: pd.Series):
    """Train an XGBoost classifier if data is sufficient (no sklearn dependency)."""
    if X.empty or y.empty:
        return None
    if y.nunique() < 2:
        return None
    if len(X) < 200:
        return None

    dtrain = xgb.DMatrix(X.to_numpy(dtype="float32"), label=y.to_numpy(dtype="float32"))
    params = {
        "max_depth": 4,
        "eta": 0.08,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "seed": 42,
    }
    booster = xgb.train(params=params, dtrain=dtrain, num_boost_round=150)
    return booster


def score_to_signal(score: float) -> str:
    if score >= 0.75:
        return "Strong Buy"
    if score >= 0.60:
        return "Buy"
    if score >= 0.45:
        return "Neutral"
    return "Avoid"


def bool_to_text(value: object) -> str:
    if pd.isna(value):
        return "-"
    return "YES" if bool(value) else "NO"


@st.cache_data(ttl=900, show_spinner=False)
def scan_universe(
    universe_name: str,
    symbols: tuple[str, ...],
    interval: str,
    period: str,
    resample_rule: str | None,
    timeframe_label: str,
    rsi_threshold: float,
    rsi_length: int,
    ema_period: int,
    ema_direction: str,
    atr_period: int,
    atr_multiplier: float,
    halftrend_amplitude: int,
    halftrend_channel_deviation: int,
    show_halftrend: bool,
    use_ema: bool,
    use_rsi: bool,
    use_supertrend: bool,
    use_last_closed_candle: bool,
    enable_ai: bool,
    ai_cache_key: str,
) -> pd.DataFrame:
    results = []
    ema_is_above = ema_direction == "Above EMA"
    ema_op = ">" if ema_is_above else "<"
    cond_ema_col = f"Close {ema_op} EMA{ema_period}"
    cond_rsi_col = f"RSI({rsi_length}) > {rsi_threshold:g}"
    failed_symbols = []
    ai_feature_rows: dict[str, pd.Series] = {}
    training_X: list[pd.DataFrame] = []
    training_y: list[pd.Series] = []
    feature_cols = ["rsi", "dist_ma50", "dist_ma200", "vol_spike", "atr_volatility", "supertrend_dir"]

    bulk_data = fetch_bulk_history(symbols, interval=interval, period=period)

    for symbol in symbols:
        try:
            if isinstance(bulk_data.columns, pd.MultiIndex):
                if symbol not in bulk_data.columns.get_level_values(0):
                    continue
                data = bulk_data[symbol].dropna(how="all")
            else:
                data = bulk_data.copy()
        except Exception as e:
            failed_symbols.append(f"{symbol} ({str(e)[:50]})")
            continue
        
        if resample_rule:
            data = resample_ohlcv(data, resample_rule)
        
        min_required = max(
            ema_period + 5,
            atr_period + 5,
            rsi_length + 20,
            105,
            halftrend_amplitude + 5,
        )
        if data.empty or len(data) < min_required:
            continue

        close = ensure_series(data["Close"]).astype("float64")
        high = ensure_series(data["High"]).astype("float64")
        low = ensure_series(data["Low"]).astype("float64")

        ohlc = pd.concat([high, low, close], axis=1, keys=["High", "Low", "Close"]).dropna()
        if len(ohlc) < min_required:
            continue

        high = ohlc["High"]
        low = ohlc["Low"]
        close = ohlc["Close"]
        symbol_clean = symbol.replace(".NS", "")

        ema = close.ewm(span=ema_period, adjust=False).mean()
        rsi = compute_rsi(close, period=rsi_length)
        _, st_dir = compute_supertrend(
            high=high,
            low=low,
            close=close,
            atr_period=atr_period,
            multiplier=atr_multiplier,
        )
        ht_buy = pd.Series(False, index=close.index, dtype="bool")
        ht_sell = pd.Series(False, index=close.index, dtype="bool")
        if show_halftrend:
            _, ht_buy, ht_sell, _ = compute_halftrend(
                high=high,
                low=low,
                close=close,
                amplitude=halftrend_amplitude,
                channel_deviation=halftrend_channel_deviation,
                atr_period=100,
            )

        signal_idx = get_signal_index(
            close.index,
            use_last_closed_candle=use_last_closed_candle,
            timeframe_label=timeframe_label,
        )
        if signal_idx is None:
            continue

        if enable_ai:
            end_date = pd.Timestamp(ohlc.index[-1])
            start_date = end_date - pd.DateOffset(years=2)
            data_ai = data.loc[data.index >= start_date].copy()

            X_sym, y_sym = build_training_matrix(
                data=data_ai,
                rsi_length=rsi_length,
                atr_period=atr_period,
                atr_multiplier=atr_multiplier,
            )
            if not X_sym.empty and not y_sym.empty:
                training_X.append(X_sym)
                training_y.append(y_sym)

            features_sym = compute_feature_frame(
                data=data_ai,
                rsi_length=rsi_length,
                atr_period=atr_period,
                atr_multiplier=atr_multiplier,
            )
            if not features_sym.empty and signal_idx in features_sym.index:
                ai_feature_rows[symbol_clean] = features_sym.loc[signal_idx]

        latest_close = float(close.loc[signal_idx])
        latest_ema = float(ema.loc[signal_idx])
        latest_rsi = float(rsi.loc[signal_idx]) if pd.notna(rsi.loc[signal_idx]) else None
        latest_st_dir = int(st_dir.loc[signal_idx]) if pd.notna(st_dir.loc[signal_idx]) else -1
        latest_ht_buy = bool(ht_buy.loc[signal_idx]) if pd.notna(ht_buy.loc[signal_idx]) else False
        latest_ht_sell = bool(ht_sell.loc[signal_idx]) if pd.notna(ht_sell.loc[signal_idx]) else False

        ema_cond = latest_close > latest_ema if ema_is_above else latest_close < latest_ema
        rsi_cond = latest_rsi is not None and latest_rsi > rsi_threshold
        supertrend_cond = latest_st_dir == 1
        ema_ok = ema_cond if use_ema else True
        rsi_ok = rsi_cond if use_rsi else True
        supertrend_ok = supertrend_cond if use_supertrend else True

        results.append(
            {
                "Universe": universe_name,
                "Symbol": symbol_clean,
                "Close": round(latest_close, 2),
                f"EMA{ema_period}": round(latest_ema, 2),
                "RSI": round(latest_rsi, 2) if latest_rsi is not None else None,
                "Supertrend": "Green" if supertrend_cond else "Red",
                "HalfTrend Signal": "Buy" if latest_ht_buy else "Sell" if latest_ht_sell else "Neutral",
                "EMA Filter Enabled": use_ema,
                "RSI Filter Enabled": use_rsi,
                "Supertrend Filter Enabled": use_supertrend,
                cond_ema_col: ema_cond if use_ema else pd.NA,
                cond_rsi_col: rsi_cond if use_rsi else pd.NA,
                "Supertrend Green": supertrend_cond if use_supertrend else pd.NA,
                "Pass": ema_ok and rsi_ok and supertrend_ok,
                "Signal Candle": signal_idx.date().isoformat(),
                "Latest Available Candle": pd.Timestamp(close.index[-1]).date().isoformat(),
                "Candle Mode": "Last Closed" if use_last_closed_candle else "Live",
                "Timeframe": timeframe_label,
            }
        )

    if not results:
        return pd.DataFrame()

    out = pd.DataFrame(results)
    if enable_ai:
        model = None
        cached_meta = st.session_state.get("ai_model_meta", {})
        if st.session_state.get("ai_model") is not None and cached_meta.get("key") == ai_cache_key:
            model = st.session_state.get("ai_model")
        else:
            if training_X and training_y:
                X_train = pd.concat(training_X, ignore_index=True)
                y_train = pd.concat(training_y, ignore_index=True)
            else:
                X_train = pd.DataFrame()
                y_train = pd.Series(dtype="int64")

            with st.spinner("Training AI model (XGBoost)..."):
                model = train_xgb_model(X_train, y_train)
            st.session_state["ai_model"] = model
            st.session_state["ai_model_meta"] = {
                "key": ai_cache_key,
                "rows": int(len(X_train)),
                "symbols": int(len(symbols)),
            }

        ai_probs_pct = []
        ai_signals = []
        ai_dist_ma50 = []
        ai_dist_ma200 = []
        ai_vol_spike = []
        ai_atr_vol = []
        ai_st_dir = []

        with st.spinner("Scoring AI probabilities..."):
            for _, row in out.iterrows():
                feat = ai_feature_rows.get(row["Symbol"])
                if model is None or feat is None or feat[feature_cols].isna().any():
                    prob = 0.5
                else:
                    features = feat[feature_cols].astype("float64").to_numpy().reshape(1, -1)
                    dtest = xgb.DMatrix(features)
                    prob = float(model.predict(dtest)[0])

                ai_probs_pct.append(round(prob * 100.0))
                ai_signals.append(score_to_signal(prob))
                ai_dist_ma50.append(
                    round(float(feat["dist_ma50"]), 2) if feat is not None and pd.notna(feat.get("dist_ma50")) else None
                )
                ai_dist_ma200.append(
                    round(float(feat["dist_ma200"]), 2) if feat is not None and pd.notna(feat.get("dist_ma200")) else None
                )
                ai_vol_spike.append(
                    round(float(feat["vol_spike"]), 2) if feat is not None and pd.notna(feat.get("vol_spike")) else None
                )
                ai_atr_vol.append(
                    round(float(feat["atr_volatility"]), 2)
                    if feat is not None and pd.notna(feat.get("atr_volatility"))
                    else None
                )
                ai_st_dir.append(
                    int(feat["supertrend_dir"]) if feat is not None and pd.notna(feat.get("supertrend_dir")) else None
                )

        out["AI Probability %"] = ai_probs_pct
        out["AI Signal"] = ai_signals
        out["Dist MA50 %"] = ai_dist_ma50
        out["Dist MA200 %"] = ai_dist_ma200
        out["Volume Spike"] = ai_vol_spike
        out["ATR Volatility %"] = ai_atr_vol
        out["Supertrend Signal"] = ai_st_dir

    out["RSI"] = pd.to_numeric(out["RSI"], errors="coerce")
    out.sort_values(["Pass", "RSI"], ascending=[False, False], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def backtest_universe(
    universe_name: str,
    symbols: tuple[str, ...],
    interval: str,
    period: str,
    timeframe_label: str,
    resample_rule: str | None,
    start_date: date | None,
    end_date: date | None,
    trade_direction: str,
    use_ema: bool,
    ema_period: int,
    ema_direction: str,
    use_rsi: bool,
    rsi_length: int,
    rsi_threshold: float,
    rsi_mode: str,
    use_supertrend: bool,
    supertrend_mode: str,
    atr_period: int,
    atr_multiplier: float,
    exit_mode: str,
    exit_indicator: str,
    target_pct: float,
    stop_pct: float,
    hold_candles: int,
    entry_expr: str | None,
    exit_expr: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trades: list[dict] = []

    if not symbols:
        return pd.DataFrame(), pd.DataFrame()

    bulk_data = fetch_bulk_history(symbols, interval=interval, period=period, start=start_date, end=end_date)
    ema_is_above = ema_direction == "Above EMA"

    for symbol in symbols:
        try:
            if isinstance(bulk_data.columns, pd.MultiIndex):
                if symbol not in bulk_data.columns.get_level_values(0):
                    continue
                data = bulk_data[symbol].dropna(how="all")
            else:
                data = bulk_data.copy()
        except Exception:
            continue

        if data.empty:
            continue

        if resample_rule:
            data = resample_ohlcv(data, resample_rule)

        if start_date is not None:
            data = data.loc[data.index >= pd.Timestamp(start_date)]
        if end_date is not None:
            data = data.loc[data.index <= pd.Timestamp(end_date)]

        min_required = max(ema_period + 5, atr_period + 5, rsi_length + 20, 105)
        if data.empty or len(data) < min_required + 2:
            continue

        close = ensure_series(data["Close"]).astype("float64")
        high = ensure_series(data["High"]).astype("float64")
        low = ensure_series(data["Low"]).astype("float64")

        ohlc = pd.concat([high, low, close], axis=1, keys=["High", "Low", "Close"]).dropna()
        if len(ohlc) < min_required + 2:
            continue

        high = ohlc["High"]
        low = ohlc["Low"]
        close = ohlc["Close"]

        indicator_cache: dict[str, pd.Series] = {}

        def ema_fn(length: int) -> pd.Series:
            key = f"ema_{length}"
            if key not in indicator_cache:
                indicator_cache[key] = close.ewm(span=length, adjust=False).mean()
            return indicator_cache[key]

        def rsi_fn(length: int) -> pd.Series:
            key = f"rsi_{length}"
            if key not in indicator_cache:
                indicator_cache[key] = compute_rsi(close, period=length)
            return indicator_cache[key]

        def supertrend_fn(atr_len: int, mult: float) -> pd.Series:
            key = f"st_{atr_len}_{mult}"
            if key not in indicator_cache:
                _, st_local = compute_supertrend(
                    high=high,
                    low=low,
                    close=close,
                    atr_period=atr_len,
                    multiplier=mult,
                )
                indicator_cache[key] = st_local
            return indicator_cache[key]

        def to_series(value: pd.Series | float | int) -> pd.Series:
            if isinstance(value, pd.Series):
                return value
            return pd.Series(value, index=close.index, dtype="float64")

        def cross_above(a: pd.Series | float | int, b: pd.Series | float | int) -> pd.Series:
            a_s = to_series(a)
            b_s = to_series(b)
            return (a_s.shift(1) <= b_s.shift(1)) & (a_s > b_s)

        def cross_below(a: pd.Series | float | int, b: pd.Series | float | int) -> pd.Series:
            a_s = to_series(a)
            b_s = to_series(b)
            return (a_s.shift(1) >= b_s.shift(1)) & (a_s < b_s)

        ema = ema_fn(ema_period)
        rsi = rsi_fn(rsi_length)
        _, st_dir = compute_supertrend(
            high=high,
            low=low,
            close=close,
            atr_period=atr_period,
            multiplier=atr_multiplier,
        )

        def build_signal(expr: str) -> pd.Series:
            expr_clean = expr.strip()
            if not expr_clean:
                raise ValueError("Rule cannot be empty.")
            expr_clean = re.sub(
                r"([A-Za-z0-9_\\.()]+)\\s+crosses\\s+above\\s+([A-Za-z0-9_\\.()]+)",
                r"cross_above(\\1, \\2)",
                expr_clean,
                flags=re.IGNORECASE,
            )
            expr_clean = re.sub(
                r"([A-Za-z0-9_\\.()]+)\\s+crosses\\s+below\\s+([A-Za-z0-9_\\.()]+)",
                r"cross_below(\\1, \\2)",
                expr_clean,
                flags=re.IGNORECASE,
            )
            expr_clean = re.sub(r"\\band\\b", "&", expr_clean, flags=re.IGNORECASE)
            expr_clean = re.sub(r"\\bor\\b", "|", expr_clean, flags=re.IGNORECASE)
            expr_clean = re.sub(r"\\bgreen\\b", "1", expr_clean, flags=re.IGNORECASE)
            expr_clean = re.sub(r"\\bred\\b", "-1", expr_clean, flags=re.IGNORECASE)
            expr_clean = re.sub(r"\\s*&\\s*", " & ", expr_clean)
            expr_clean = re.sub(r"\\s*\\|\\s*", " | ", expr_clean)
            if " & " in expr_clean or " | " in expr_clean:
                expr_clean = "(" + expr_clean.replace(" & ", ") & (").replace(" | ", ") | (") + ")"
            env = {
                "close": close,
                "open": ensure_series(data["Open"]).astype("float64"),
                "high": high,
                "low": low,
                "volume": ensure_series(data["Volume"]).astype("float64"),
                "ema": ema_fn,
                "rsi": rsi_fn,
                "supertrend": supertrend_fn,
                "cross_above": cross_above,
                "cross_below": cross_below,
            }
            try:
                result = eval(expr_clean, {"__builtins__": {}}, env)
            except Exception as exc:
                raise ValueError(f"Invalid rule: {exc}. Parsed: {expr_clean}") from exc
            if isinstance(result, pd.Series):
                return result.fillna(False).astype(bool)
            raise ValueError("Rule did not produce a series result.")

        if entry_expr:
            signal = build_signal(entry_expr)
            exit_signal = build_signal(exit_expr) if exit_expr else None
        else:
            ema_cond = close > ema if ema_is_above else close < ema

            if rsi_mode == "RSI > Threshold":
                rsi_cond = rsi > rsi_threshold
            elif rsi_mode == "RSI < Threshold":
                rsi_cond = rsi < rsi_threshold
            elif rsi_mode == "RSI Crosses Above":
                rsi_cond = (rsi.shift(1) <= rsi_threshold) & (rsi > rsi_threshold)
            else:
                rsi_cond = (rsi.shift(1) >= rsi_threshold) & (rsi < rsi_threshold)

            supertrend_cond = st_dir == (1 if supertrend_mode == "Green (Bullish)" else -1)

            signal = pd.Series(True, index=close.index)
            if use_ema:
                signal &= ema_cond
            if use_rsi:
                signal &= rsi_cond
            if use_supertrend:
                signal &= supertrend_cond
            exit_signal = None

        symbol_clean = symbol.replace(".NS", "")
        in_trade = False
        entry_idx = None
        entry_price = None
        entry_is_short = False

        for idx in signal.index:
            if not in_trade:
                if bool(signal.loc[idx]) and pd.notna(close.loc[idx]):
                    in_trade = True
                    entry_idx = idx
                    entry_price = float(close.loc[idx])
                    if trade_direction == "Auto (Supertrend)" and use_supertrend:
                        st_value = st_dir.loc[idx] if idx in st_dir.index else 1
                        entry_is_short = int(st_value) == -1
                    else:
                        entry_is_short = trade_direction == "Short"
                continue

            if entry_idx is None or entry_price is None:
                in_trade = False
                continue

            exit_now = False
            exit_reason = ""
            is_short = entry_is_short if trade_direction == "Auto (Supertrend)" else trade_direction == "Short"

            if exit_signal is not None and bool(exit_signal.loc[idx]):
                exit_now = True
                exit_reason = "Custom Exit"
            elif exit_mode == "Fixed Target/SL":
                if is_short:
                    curr_ret = (entry_price / float(close.loc[idx]) - 1.0) * 100.0
                else:
                    curr_ret = (float(close.loc[idx]) / entry_price - 1.0) * 100.0
                if curr_ret >= target_pct:
                    exit_now = True
                    exit_reason = f"Target {target_pct:g}%"
                elif curr_ret <= -abs(stop_pct):
                    exit_now = True
                    exit_reason = f"Stop {-abs(stop_pct):g}%"
            elif exit_mode == "Indicator Flip":
                if exit_indicator == "EMA":
                    exit_now = not bool(ema_cond.loc[idx])
                elif exit_indicator == "RSI":
                    exit_now = not bool(rsi_cond.loc[idx])
                else:
                    exit_now = not bool(supertrend_cond.loc[idx])
                exit_reason = f"{exit_indicator} Flip"
            else:
                if entry_idx is not None:
                    held = (signal.index.get_loc(idx) - signal.index.get_loc(entry_idx))
                    if held >= max(hold_candles, 1):
                        exit_now = True
                        exit_reason = f"Time {hold_candles} candles"

            if exit_now:
                exit_price = float(close.loc[idx])
                if is_short:
                    ret_pct = (entry_price / exit_price - 1.0) * 100.0
                else:
                    ret_pct = (exit_price / entry_price - 1.0) * 100.0
                trades.append(
                    {
                        "Universe": universe_name,
                        "Symbol": symbol_clean,
                        "Entry Date": pd.Timestamp(entry_idx).date().isoformat(),
                        "Exit Date": pd.Timestamp(idx).date().isoformat(),
                        "Entry Close": round(entry_price, 2),
                        "Exit Close": round(exit_price, 2),
                        "Return %": round(ret_pct, 2),
                        "Exit Reason": exit_reason,
                        "Timeframe": timeframe_label,
                        "Direction": "Short" if is_short else "Long",
                    }
                )
                in_trade = False
                entry_idx = None
                entry_price = None

    if not trades:
        return pd.DataFrame(), pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    trades_df.sort_values(["Entry Date", "Symbol"], inplace=True, ignore_index=True)

    stats = {
        "Total Signals": int(len(trades_df)),
        "Win Rate %": round(float((trades_df["Return %"] > 0).mean() * 100.0), 2),
        "Avg Return %": round(float(trades_df["Return %"].mean()), 2),
        "Median Return %": round(float(trades_df["Return %"].median()), 2),
        "Best Return %": round(float(trades_df["Return %"].max()), 2),
        "Worst Return %": round(float(trades_df["Return %"].min()), 2),
        "Cumulative Return %": round(float(((trades_df["Return %"] / 100.0 + 1.0).prod() - 1.0) * 100.0), 2),
    }
    stats_df = pd.DataFrame([stats])
    return stats_df, trades_df


def main() -> None:
    st.set_page_config(page_title="Indian Index Daily Scanner", layout="wide", initial_sidebar_state="expanded")
    apply_ui_style()
    set_yf_cache()

    st.session_state.enable_ai = True

    market_data = fetch_market_data()
    stock_scanner_banner(market_data)

    nav_choice = st.radio(
        "Navigation",
        options=["Scanner", "AI Mode", "Backtest", "Portfolio", "Charts"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if nav_choice == "AI Mode":
        st.header("AI Mode")
        st.caption("AI scoring is always enabled and runs when you scan in the Scanner page.")

        table = st.session_state.get("scan_table", pd.DataFrame())
        if table is None or table.empty or "AI Probability %" not in table.columns:
            st.info("Run a scan in the Scanner page to view AI results here.")
            return

        st.subheader("AI Top 5")
        top_ai = table.sort_values("AI Probability %", ascending=False).head(5)
        for _, row in top_ai.iterrows():
            prob_pct = float(row["AI Probability %"]) if pd.notna(row.get("AI Probability %")) else 50.0
            signal_label = row.get("AI Signal", "Neutral")
            symbol = row.get("Symbol", "-")
            dist_ma50 = row.get("Dist MA50 %", "-")
            dist_ma200 = row.get("Dist MA200 %", "-")

            st.markdown(f"**{symbol}** - {signal_label} ({prob_pct:.0f}%)")
            st.progress(min(max(prob_pct, 0.0), 100.0) / 100.0)
            st.caption(f"MA50%: {dist_ma50} | MA200%: {dist_ma200}")

        st.divider()
        st.subheader("AI Feature Table")
        ai_cols = [
            "Symbol",
            "AI Probability %",
            "AI Signal",
            "Dist MA50 %",
            "Dist MA200 %",
            "Volume Spike",
            "ATR Volatility %",
            "Supertrend Signal",
        ]
        ai_cols = [c for c in ai_cols if c in table.columns]
        ai_table = table[ai_cols].copy()
        st.dataframe(ai_table, use_container_width=True, height=520, hide_index=True)
        return
    if nav_choice == "Backtest":
        st.header("Backtest")
        st.caption("Run a custom-rule backtest over a date range or max history.")

        backtest_source_bt = st.selectbox(
            "Backtest Source",
            options=["Single Stock", "Universe", "Upload File"],
            index=0,
            key="bt_source_mode",
        )
        universe_bt = st.selectbox(
            "Select Index/Universe",
            options=UNIVERSE_OPTIONS,
            index=0,
            key="bt_universe",
            disabled=backtest_source_bt != "Universe",
        )
        uploaded_file_bt = st.file_uploader(
            "Upload Symbols (CSV/XLSX)",
            type=["csv", "xlsx", "xls"],
            key="bt_upload",
            disabled=backtest_source_bt != "Upload File",
        )
        if backtest_source_bt == "Upload File" and uploaded_file_bt is not None:
            universe_symbols_bt = load_symbols_from_file(uploaded_file_bt)
            universe_source_bt = f"Uploaded file: {uploaded_file_bt.name}"
        elif backtest_source_bt == "Universe":
            universe_symbols_bt, universe_source_bt = resolve_universe(universe_bt)
        else:
            universe_symbols_bt = []
            universe_source_bt = "Single Stock"

        tf_col, date_col = st.columns([1, 2])
        with tf_col:
            timeframe_choice_bt = st.selectbox(
                "Timeframe",
                options=["Custom"],
                index=0,
            )
            custom_tf_raw = ""
            if timeframe_choice_bt == "Custom":
                custom_tf_raw = st.text_input(
                    "Custom Timeframe",
                    placeholder="Examples: 3M, 1D, 1W, 1H",
                    help="Use M for minutes, H for hours, D for days, W for weeks (e.g., 3M, 1H, 1D, 1W).",
                ).strip().upper()

            interval_bt = "1d"
            period_bt = "max"
            resample_rule_bt = None
            timeframe_label_bt = "1D"
            intraday_bt = False
            custom_invalid_bt = False

            if not re.fullmatch(r"[0-9]+[MDWH]", custom_tf_raw or ""):
                custom_invalid_bt = True
                timeframe_label_bt = "Custom"
            else:
                custom_value = int(custom_tf_raw[:-1])
                custom_unit = custom_tf_raw[-1]
                if custom_unit == "W":
                    interval_bt = "1d"
                    resample_rule_bt = f"{custom_value}W-FRI"
                    timeframe_label_bt = f"{custom_value}W"
                elif custom_unit == "D":
                    interval_bt = "1d"
                    resample_rule_bt = None if custom_value == 1 else f"{custom_value}D"
                    timeframe_label_bt = f"{custom_value}D"
                elif custom_unit == "H":
                    intraday_bt = True
                    interval_bt = "60m"
                    resample_rule_bt = None if custom_value == 1 else f"{custom_value}H"
                    timeframe_label_bt = f"{custom_value}H"
                else:
                    intraday_bt = True
                    minutes = custom_value
                    supported_minutes = [90, 60, 30, 15, 5, 2, 1]
                    base_interval = 1
                    for base in supported_minutes:
                        if minutes % base == 0:
                            base_interval = base
                            break
                    interval_bt = f"{base_interval}m"
                    if minutes % 60 == 0:
                        timeframe_label_bt = f"{minutes // 60}H"
                        resample_rule_bt = None if minutes == base_interval else f"{minutes // 60}H"
                    else:
                        timeframe_label_bt = f"{minutes}M"
                        resample_rule_bt = None if minutes == base_interval else f"{minutes}T"
            if custom_invalid_bt:
                st.error("Enter a valid timeframe like 3M, 1H, 1D, or 1W.")
        with date_col:
            use_max_history = st.checkbox("Use Max History (from beginning)", value=False)
            if use_max_history:
                start_date = None
                end_date = None
                st.caption("Using full available history.")
            else:
                today = datetime.now().date()
                default_start = today.replace(year=today.year - 2)
                date_range = st.date_input(
                    "Select Date Range",
                    value=(default_start, today),
                )
                if isinstance(date_range, tuple) and len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date = None
                    end_date = None

        if intraday_bt:
            max_days = 7 if interval_bt == "1m" else 60
            if use_max_history:
                period_bt = f"{max_days}d"
                st.info(
                    f"Intraday data is limited to the last {max_days} days. "
                    "If you want to test full history, use daily or weekly."
                )
            else:
                today = datetime.now().date()
                if end_date is None:
                    end_date = today
                if start_date is None:
                    start_date = end_date - pd.Timedelta(days=max_days)
                if (end_date - start_date).days > max_days:
                    start_date = end_date - pd.Timedelta(days=max_days)
                    st.info(
                        f"Intraday data is limited to the last {max_days} days, so the range was capped. "
                        "If you want to test full history, use daily or weekly."
                    )

        st.subheader("Rules")
        st.caption("Example: rsi(14) crosses above 50 and close > ema(200) and supertrend(10,3) == green")
        presets = {
            "Trend Long": (
                "rsi(14) crosses above 50 and close > ema(200) and supertrend(10,3) == green",
                "rsi(14) crosses below 50 or close < ema(200) or supertrend(10,3) == red",
            ),
            "Trend Short": (
                "rsi(14) crosses below 50 and close < ema(200) and supertrend(10,3) == red",
                "rsi(14) crosses above 50 or close > ema(200) or supertrend(10,3) == green",
            ),
        }
        preset_name = st.selectbox(
            "Rule Presets",
            options=["Custom"] + list(presets.keys()),
            index=0,
            key="bt_rule_preset",
        )
        if preset_name in presets:
            preset_entry, preset_exit = presets[preset_name]
            st.session_state["bt_entry_rule"] = preset_entry
            st.session_state["bt_exit_rule"] = preset_exit
        entry_expr = st.text_area("Entry Rule", height=80, key="bt_entry_rule")
        exit_expr = st.text_area("Exit Rule", height=80, key="bt_exit_rule")

        st.subheader("Stock Selection")
        stock_input = None
        selected_symbol = None
        if backtest_source_bt == "Single Stock":
            stock_input = st.text_input(
                "Enter Stock Symbol (must end with .NS)",
                placeholder="Example: TCS.NS",
            ).strip().upper()

            if stock_input:
                if not stock_input.endswith(".NS"):
                    st.error("Please type correct name with .NS (e.g., TCS.NS).")
                else:
                    selected_symbol = stock_input

        if st.button("Run Backtest", type="primary"):
            if timeframe_choice_bt == "Custom" and custom_invalid_bt:
                st.error("Please enter a valid custom timeframe (e.g., 3M, 1H, 1D, 1W).")
                return
            if backtest_source_bt == "Single Stock":
                if not selected_symbol:
                    st.error("Please enter a valid stock symbol with .NS.")
                    return
                symbols_bt = (selected_symbol,)
                universe_name_bt = selected_symbol.replace(".NS", "")
            else:
                if not universe_symbols_bt:
                    st.error("No symbols available for backtest.")
                    return
                symbols_bt = tuple(universe_symbols_bt)
                if backtest_source_bt == "Universe":
                    universe_name_bt = universe_bt
                else:
                    universe_name_bt = uploaded_file_bt.name if uploaded_file_bt else "Uploaded File"
            if not entry_expr.strip():
                st.error("Please enter an Entry Rule.")
                return
            with st.spinner("Running backtest..."):
                try:
                    stats_df, trades_df = backtest_universe(
                        universe_name=universe_name_bt,
                        symbols=symbols_bt,
                        interval=interval_bt,
                        period=period_bt,
                        timeframe_label=timeframe_label_bt,
                        resample_rule=resample_rule_bt,
                        start_date=start_date,
                        end_date=end_date,
                        trade_direction="Long",
                        use_ema=False,
                        ema_period=200,
                        ema_direction="Above EMA",
                        use_rsi=False,
                        rsi_length=14,
                        rsi_threshold=50.0,
                        rsi_mode="RSI > Threshold",
                        use_supertrend=False,
                        supertrend_mode="Green (Bullish)",
                        atr_period=10,
                        atr_multiplier=3.0,
                        exit_mode="Fixed Target/SL",
                        exit_indicator="EMA",
                        target_pct=5.0,
                        stop_pct=3.0,
                        hold_candles=5,
                        entry_expr=entry_expr,
                        exit_expr=exit_expr if exit_expr.strip() else None,
                    )
                except ValueError as exc:
                    st.error(str(exc))
                    return
            st.session_state["bt_stats"] = stats_df
            st.session_state["bt_trades"] = trades_df
            st.session_state["bt_source"] = universe_source_bt

        stats_df = st.session_state.get("bt_stats", pd.DataFrame())
        trades_df = st.session_state.get("bt_trades", pd.DataFrame())
        if not stats_df.empty:
            st.subheader("Summary")
            row = stats_df.iloc[0]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Signals", f"{int(row.get('Total Signals', 0))}")
            m2.metric("Win Rate", f"{row.get('Win Rate %', 0):.1f}%")
            m3.metric("Avg Return", f"{row.get('Avg Return %', 0):.2f}%")
            m4.metric("Cumulative", f"{row.get('Cumulative Return %', 0):.2f}%")
            if not trades_df.empty and "Return %" in trades_df.columns:
                equity = (trades_df["Return %"] / 100.0 + 1.0).cumprod()
                equity_df = pd.DataFrame({"Equity": equity.values})
                st.line_chart(equity_df, height=220)

        if not trades_df.empty:
            st.subheader("Trades")
            trades_view = trades_df.copy()
            display_cols = [
                "Entry Date",
                "Exit Date",
                "Symbol",
                "Direction",
                "Entry Close",
                "Exit Close",
                "Return %",
                "Exit Reason",
                "Timeframe",
            ]
            display_cols = [c for c in display_cols if c in trades_view.columns]
            trades_view = trades_view[display_cols]
            st.dataframe(trades_view, use_container_width=True, height=520, hide_index=True)
            st.download_button(
                "Download Trades CSV",
                data=trades_view.to_csv(index=False).encode("utf-8"),
                file_name="backtest_trades.csv",
                mime="text/csv",
                use_container_width=True,
            )
        return
    if nav_choice != "Scanner":
        st.info(f"{nav_choice} page is coming soon.")
        return

    st.sidebar.header("⚙️ Scanner Configuration")
    
    with st.sidebar.expander("ℹ️ About This Scanner", expanded=False):
        st.caption("✓ Timeframe: Selectable (1D/1W/1M/6M/12M)")
        st.caption("✓ Data: Full available history from stock listing")
        st.caption("✓ Updated: Real-time with market close")
        st.caption("✓ Not for backtesting - signals generated on latest candle")
    
    universe_mode = st.sidebar.radio(
        "📦 Stock Source",
        options=["Universe", "Upload File"],
        index=0,
        help="Choose a predefined universe or upload your own symbols file",
        horizontal=True,
    )

    universe = st.sidebar.selectbox(
        "📊 Select Index/Universe",
        options=UNIVERSE_OPTIONS,
        index=0,
        help="Choose from major Indian indices, IT sector, or market cap segments",
        disabled=universe_mode != "Universe",
    )

    uploaded_file = st.sidebar.file_uploader(
        "📥 Upload Symbols (CSV/XLSX)",
        type=["csv", "xlsx", "xls"],
        help="File should contain a column like Symbol/Ticker, or the first column will be used.",
        disabled=universe_mode != "Upload File",
    )

    if universe_mode == "Upload File" and uploaded_file is not None:
        universe_symbols = load_symbols_from_file(uploaded_file)
        universe_source = f"Uploaded file: {uploaded_file.name}"
    else:
        universe_symbols, universe_source = resolve_universe(universe)

    if universe_symbols:
        st.sidebar.caption(f"Universe symbols loaded: {len(universe_symbols)} ({universe_source})")
    else:
        if universe_mode == "Upload File":
            st.sidebar.error("No symbols found in uploaded file.")
        else:
            st.sidebar.error("Could not load symbols for selected universe.")


    st.sidebar.subheader("Timeframe")
    timeframe_choice = st.sidebar.selectbox(
        "Select Timeframe",
        options=list(TIMEFRAME_OPTIONS.keys()),
        index=0,
        help="Choose candle timeframe for the scan",
    )
    timeframe_cfg = TIMEFRAME_OPTIONS[timeframe_choice]
    interval = timeframe_cfg["interval"]
    period = timeframe_cfg["period"]
    resample_rule = timeframe_cfg["resample"]
    timeframe_label = timeframe_cfg["label"]

    st.sidebar.subheader("Candle Data")
    candle_mode = st.sidebar.radio(
        "Candle Mode",
        options=["Live (Latest Candle)", "Last Closed Candle"],
        index=0,
        horizontal=True,
        help="Choose whether to use the latest candle or the last fully closed candle",
    )
    use_last_closed_candle = candle_mode == "Last Closed Candle"
    st.sidebar.subheader("🔍 Filter Controls")
    enable_ai = True
    selected_filters = st.sidebar.multiselect(
        "Select Filters",
        options=[
            "EMA Filter (Trend)",
            "RSI Filter (Momentum)",
            "Supertrend Filter (Support/Resistance)",
        ],
        default=[],
        help="Choose one or more filters to apply",
        key="selected_filters_v2",
    )
    use_ema = "EMA Filter (Trend)" in selected_filters
    use_rsi = "RSI Filter (Momentum)" in selected_filters
    use_supertrend = "Supertrend Filter (Support/Resistance)" in selected_filters

    st.sidebar.divider()
    st.sidebar.subheader("🧭 HalfTrend Display")
    show_halftrend = st.sidebar.toggle(
        "Show HalfTrend Signal",
        value=False,
        help="Display HalfTrend Buy/Sell/Neutral in the tables",
    )
    st.sidebar.subheader("⚙️ Indicator Parameters")
    if use_ema:
        ema_direction = st.sidebar.radio(
            "EMA Condition",
            options=["Below EMA", "Above EMA"],
            index=0,
            horizontal=True,
            help="Choose whether price must be below or above EMA to qualify",
        )
        ema_period = st.sidebar.number_input(
            "EMA Period",
            min_value=50,
            max_value=400,
            value=200,
            step=5,
            help="Exponential Moving Average period (larger = longer trend)"
        )
    else:
        ema_direction = "Above EMA"
        ema_period = 200

    if use_rsi:
        rsi_length = st.sidebar.number_input(
            "RSI Length",
            min_value=2,
            max_value=200,
            value=14,
            step=1,
            help="Relative Strength Index period (standard: 14)"
        )
        rsi_threshold = st.sidebar.number_input(
            "RSI Threshold",
            min_value=1.0,
            max_value=99.0,
            value=50.0,
            step=1.0,
            help="Stock must have RSI above this value (neutral: 50, strong: 70+)"
        )
    else:
        rsi_length = 14
        rsi_threshold = 50.0

    if use_supertrend:
        atr_period = st.sidebar.number_input(
            "Supertrend ATR Period",
            min_value=5,
            max_value=50,
            value=10,
            step=1,
            help="Average True Range period for volatility-based bands"
        )
        atr_multiplier = st.sidebar.number_input(
            "Supertrend Multiplier",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Band width multiplier (3.0 is standard)"
        )
    else:
        atr_period = 10
        atr_multiplier = 3.0

    st.sidebar.subheader("➕ HalfTrend Parameters")
    if show_halftrend:
        halftrend_amplitude = st.sidebar.number_input(
            "HalfTrend Amplitude",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            help="Amplitude for HalfTrend (higher = smoother)",
        )
        halftrend_channel_deviation = st.sidebar.number_input(
            "HalfTrend Channel Deviation",
            min_value=1,
            max_value=20,
            value=2,
            step=1,
            help="Channel deviation multiplier",
        )
    else:
        halftrend_amplitude = 2
        halftrend_channel_deviation = 2

    st.sidebar.divider()
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🖭️ Clear Cache", use_container_width=True, help="Clear cached data and reload"):
            st.cache_data.clear()
            st.sidebar.success("✅ Cache cleared!")
    
    with col2:
        run_scan = st.button(
            "🔍 Run Scan",
            type="primary",
            use_container_width=True,
            key="run_scan_button",
            help="Scan selected universe for qualifying stocks"
        )

    if "scan_table" not in st.session_state:
        st.session_state.scan_table = pd.DataFrame()
    if "run_timestamp" not in st.session_state:
        st.session_state.run_timestamp = ""
    if "scan_meta" not in st.session_state:
        st.session_state.scan_meta = {
            "universe": universe if universe_mode == "Universe" else "Uploaded Symbols",
            "universe_source": universe_source,
            "universe_size": len(universe_symbols),
            "enable_ai": enable_ai,
            "timeframe": timeframe_label,
            "candle_mode": candle_mode,
            "use_ema": use_ema,
            "use_rsi": use_rsi,
            "use_supertrend": use_supertrend,
            "use_last_closed_candle": use_last_closed_candle,
            "rsi_length": rsi_length,
            "show_halftrend": show_halftrend,
            "ema_direction": ema_direction,
        }

    if run_scan:
        if not universe_symbols:
            st.error(
                f"Could not load constituents for {universe}. "
                "Please check internet connection or try another universe."
            )
            return
        with st.spinner(f"Scanning {len(universe_symbols)} stocks in {universe} on {timeframe_label} timeframe..."):
            ai_cache_key = (
                f"{universe}|{timeframe_label}|{interval}|{resample_rule}|"
                f"{rsi_length}|{atr_period}|{atr_multiplier}|{len(universe_symbols)}|"
                f"{use_last_closed_candle}|{enable_ai}"
            )
            table = scan_universe(
                universe_name=universe,
                symbols=tuple(universe_symbols),
                interval=interval,
                period=period,
                resample_rule=resample_rule,
                timeframe_label=timeframe_label,
                rsi_threshold=rsi_threshold,
                rsi_length=rsi_length,
                ema_period=ema_period,
                ema_direction=ema_direction,
                atr_period=atr_period,
                atr_multiplier=atr_multiplier,
                halftrend_amplitude=int(halftrend_amplitude),
                halftrend_channel_deviation=int(halftrend_channel_deviation),
                show_halftrend=show_halftrend,
                use_ema=use_ema,
                use_rsi=use_rsi,
                use_supertrend=use_supertrend,
                use_last_closed_candle=use_last_closed_candle,
                enable_ai=enable_ai,
                ai_cache_key=ai_cache_key,
            )
        st.session_state.scan_table = table
        st.session_state.run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.scan_meta = {
            "universe": universe if universe_mode == "Universe" else "Uploaded Symbols",
            "universe_source": universe_source,
            "universe_size": len(universe_symbols),
            "enable_ai": enable_ai,
            "timeframe": timeframe_label,
            "candle_mode": candle_mode,
            "use_ema": use_ema,
            "use_rsi": use_rsi,
            "use_supertrend": use_supertrend,
            "use_last_closed_candle": use_last_closed_candle,
            "rsi_length": rsi_length,
            "show_halftrend": show_halftrend,
            "ema_direction": ema_direction,
        }

    table = st.session_state.scan_table
    if table.empty:
        st.info("Click 'Run Scan' from the sidebar to generate signals.")
        return

    scan_meta = st.session_state.get("scan_meta", {})
    rsi_len_meta = int(scan_meta.get("rsi_length", rsi_length))
    ema_col = next((c for c in table.columns if c.startswith("EMA")), f"EMA{ema_period}")
    ema_direction_meta = scan_meta.get("ema_direction", "Above EMA")
    ema_op = ">" if ema_direction_meta == "Above EMA" else "<"
    cond_ema_col = next(
        (c for c in table.columns if c.startswith("Close ") and "EMA" in c),
        f"Close {ema_op} EMA{ema_period}",
    )
    cond_rsi_col = next((c for c in table.columns if c.startswith("RSI(")), f"RSI({rsi_len_meta}) > {rsi_threshold:g}")
    condition_cols = [
        col
        for col in [cond_ema_col, cond_rsi_col, "Supertrend Green", "Pass"]
        if col in table.columns
    ]
    passed = table[table["Pass"]].copy()

    total_scanned = len(table)
    passed_count = len(passed)
    pass_rate = (passed_count / total_scanned) * 100 if total_scanned else 0.0
    signal_candle = table["Signal Candle"].max() if "Signal Candle" in table.columns else "-"
    latest_available = table["Latest Available Candle"].max() if "Latest Available Candle" in table.columns else "-"
    candle_mode = table["Candle Mode"].iloc[0] if "Candle Mode" in table.columns and not table.empty else "Live"
    timeframe_meta = table["Timeframe"].iloc[0] if "Timeframe" in table.columns and not table.empty else "-"
    universe_meta = scan_meta.get("universe", "Universe")
    universe_source_meta = scan_meta.get("universe_source", "-")
    universe_size_meta = int(scan_meta.get("universe_size", total_scanned))

    active_filters = []
    use_ema_meta = bool(scan_meta.get("use_ema", True))
    use_rsi_meta = bool(scan_meta.get("use_rsi", True))
    use_supertrend_meta = bool(scan_meta.get("use_supertrend", True))
    ai_enabled_meta = bool(scan_meta.get("enable_ai", False))

    if use_ema_meta:
        active_filters.append(cond_ema_col)
    if use_rsi_meta:
        active_filters.append(cond_rsi_col)
    if use_supertrend_meta:
        active_filters.append("Supertrend Green")
    active_filter_text = ", ".join(active_filters) if active_filters else "No filter active"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📊 Universe", universe_meta)
    m2.metric("✅ Scanned", f"{total_scanned}/{universe_size_meta}")
    m3.metric("🎯 Matched", str(passed_count))
    m4.metric("📈 Hit Rate", f"{pass_rate:.1f}%")
    
    failed_count = universe_size_meta - total_scanned
    if failed_count > 0:
        st.warning(
            f"⚠️ **{failed_count} symbols** could not be loaded due to data unavailability (404/401 errors). "
            f"This is normal for delisted stocks or API rate limits. Scan completed with {total_scanned} available stocks."
        )

    st.caption(
        f"🕐 Run time: {st.session_state.run_timestamp} | 📅 Signal candle: {signal_candle} | "
        f"📊 Latest: {latest_available} | ⏱️ Mode: {candle_mode} | "
        f"📈 Timeframe: {timeframe_meta} | 📥 Source: {universe_source_meta} | 🎛️ Filters: {active_filter_text}"
    )

    

    tab_labels = ["Qualified Stocks", "All Stocks"]
    tab1, tab2 = st.tabs(tab_labels)
    show_halftrend_meta = bool(scan_meta.get("show_halftrend", True))
    hide_columns = []
    if not use_ema_meta:
        hide_columns.extend([ema_col, "EMA Filter Enabled", cond_ema_col])
    if not use_rsi_meta:
        hide_columns.extend(["RSI", "RSI Filter Enabled", cond_rsi_col])
    if not use_supertrend_meta:
        hide_columns.extend(["Supertrend", "Supertrend Filter Enabled", "Supertrend Green"])

    with tab1:
        title_col, filter_col = st.columns([2, 1])
        with title_col:
            st.subheader("Stocks Passing Active Conditions")
        with filter_col:
            st.text_input(
                "Filter Name",
                value=st.session_state.get("filter_symbol", ""),
                key="filter_symbol",
                help="Type to filter the tables. Example: TCS, RELIANCE",
            )
        filter_symbol = st.session_state.get("filter_symbol", "").strip()
        if passed.empty:
            st.warning(f"No {universe_meta} stock is matching all selected daily conditions right now.")
        else:
            pass_cols = [
                "Universe",
                "Symbol",
                "Close",
                ema_col,
                "RSI",
                "Supertrend",
            ]
            if show_halftrend_meta:
                pass_cols.insert(pass_cols.index("Supertrend") + 1, "HalfTrend Signal")
            pass_cols = [col for col in pass_cols if col not in hide_columns]
            pass_cols = [col for col in pass_cols if col in passed.columns]
            pass_table = passed[pass_cols].copy()
            if filter_symbol:
                pass_table = pass_table[
                    pass_table["Symbol"].str.contains(filter_symbol, case=False, na=False)
                ]
            pass_table.insert(0, "Sr. No", range(1, len(pass_table) + 1))
            for col in condition_cols:
                if col in pass_table.columns:
                    pass_table[col] = pass_table[col].map(bool_to_text)
            st.dataframe(pass_table, use_container_width=True, height=520, hide_index=True)

            st.download_button(
                "Download Qualified CSV",
                data=pass_table.to_csv(index=False).encode("utf-8"),
                file_name="nifty50_daily_scanner_qualified.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with tab2:
        title_col2, filter_col2 = st.columns([2, 1])
        with title_col2:
            st.subheader("Complete Daily Scan Table")
        with filter_col2:
            st.text_input(
                "Filter Name",
                value=st.session_state.get("filter_symbol", ""),
                key="filter_symbol_tab2",
                help="Type to filter the tables. Example: TCS, RELIANCE",
            )
        filter_symbol = st.session_state.get("filter_symbol_tab2", "").strip() or st.session_state.get("filter_symbol", "").strip()
        full_export = table.copy()
        full_table = table.copy()
        ai_columns = [
            "AI Probability %",
            "AI Signal",
            "Dist MA50 %",
            "Dist MA200 %",
            "Volume Spike",
            "ATR Volatility %",
            "Supertrend Signal",
        ]
        full_export = full_export.drop(columns=ai_columns, errors="ignore")
        full_table = full_table.drop(columns=ai_columns, errors="ignore")
        if not show_halftrend_meta:
            full_table = full_table.drop(columns=["HalfTrend Signal"], errors="ignore")
        full_table = full_table.drop(
            columns=[
                "Signal Candle",
                "Latest Available Candle",
                "Candle Mode",
                "Timeframe",
                cond_ema_col,
                cond_rsi_col,
                "Supertrend Green",
                "Pass",
            ],
            errors="ignore",
        )
        full_table = full_table.drop(
            columns=["EMA Filter Enabled", "RSI Filter Enabled", "Supertrend Filter Enabled"],
            errors="ignore",
        )
        if hide_columns:
            full_table = full_table.drop(columns=hide_columns, errors="ignore")
        if filter_symbol:
            full_table = full_table[
                full_table["Symbol"].str.contains(filter_symbol, case=False, na=False)
            ]
        for col in condition_cols:
            if col in full_table.columns:
                full_table[col] = full_table[col].map(bool_to_text)
        full_table = full_table.copy()
        full_table.insert(0, "Sr. No", range(1, len(full_table) + 1))
        st.dataframe(full_table, use_container_width=True, height=520, hide_index=True)

        st.download_button(
            "Download Full Results CSV",
            data=full_export.to_csv(index=False).encode("utf-8"),
            file_name="nifty50_daily_scanner_full.csv",
            mime="text/csv",
            use_container_width=True,
        )

if __name__ == "__main__":
    main()


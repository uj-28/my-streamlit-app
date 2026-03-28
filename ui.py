import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yfinance as yf

from config import (
    BANNER_H,
    COL_HEADER_Y,
    IFRAME_PAD,
    LIVE_Y,
    MARKET_DATA,
    MARKET_TICKERS,
    PANEL_DIVIDER_Y,
    PANEL_HEIGHT,
    PANEL_TITLE_Y,
    PANEL_TOP,
    ROW_Y,
)


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


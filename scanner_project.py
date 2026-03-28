from datetime import datetime
import re

import pandas as pd
import streamlit as st

from config import AI_CONFIDENCE_DEFAULT, TIMEFRAME_OPTIONS, UNIVERSE_OPTIONS
from data_fetch import load_symbols_from_file, resolve_universe, set_yf_cache
from scanner_core import backtest_universe, scan_universe
from ui import apply_ui_style, fetch_market_data, stock_scanner_banner
from utils import bool_to_text, configure_logging


def main() -> None:
    st.set_page_config(page_title="Indian Index Daily Scanner", layout="wide", initial_sidebar_state="expanded")
    apply_ui_style()
    set_yf_cache()
    configure_logging()

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
            "AI Confident",
            "Dist MA50 %",
            "Dist MA200 %",
            "EMA Dist %",
            "Volume Ratio",
            "ATR %",
            "ADX",
            "52W High %",
            "52W Low %",
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

            if not custom_tf_raw:
                custom_tf_raw = "1D"

            if not re.fullmatch(r"[0-9]+[MDWH]", custom_tf_raw):
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
                st.error("Enter a valid timeframe like 3M, 1H, 1D, or 1W. Leave blank for 1D.")
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
    st.sidebar.header("?? Scanner Configuration")

    with st.sidebar.expander("?? About This Scanner", expanded=False):
        st.caption("? Timeframe: Selectable (1D/1W/1M/6M/12M)")
        st.caption("? Data: Full available history from stock listing")
        st.caption("? Updated: Real-time with market close")
        st.caption("? Not for backtesting - signals generated on latest candle")

    universe_mode = st.sidebar.radio(
        "?? Stock Source",
        options=["Universe", "Upload File"],
        index=0,
        help="Choose a predefined universe or upload your own symbols file",
        horizontal=True,
    )

    universe = st.sidebar.selectbox(
        "?? Select Index/Universe",
        options=UNIVERSE_OPTIONS,
        index=0,
        help="Choose from major Indian indices, IT sector, or market cap segments",
        disabled=universe_mode != "Universe",
    )

    uploaded_file = st.sidebar.file_uploader(
        "?? Upload Symbols (CSV/XLSX)",
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
    st.sidebar.subheader("?? Filter Controls")
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
    st.sidebar.subheader("?? HalfTrend Display")
    show_halftrend = st.sidebar.toggle(
        "Show HalfTrend Signal",
        value=False,
        help="Display HalfTrend Buy/Sell/Neutral in the tables",
    )
    st.sidebar.subheader("?? Indicator Parameters")
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
            help="Exponential Moving Average period (larger = longer trend)",
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
            help="Relative Strength Index period (standard: 14)",
        )
        rsi_threshold = st.sidebar.number_input(
            "RSI Threshold",
            min_value=1.0,
            max_value=99.0,
            value=50.0,
            step=1.0,
            help="Stock must have RSI above this value (neutral: 50, strong: 70+)",
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
            help="Average True Range period for volatility-based bands",
        )
        atr_multiplier = st.sidebar.number_input(
            "Supertrend Multiplier",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="Band width multiplier (3.0 is standard)",
        )
    else:
        atr_period = 10
        atr_multiplier = 3.0

    st.sidebar.subheader("? HalfTrend Parameters")
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

    st.sidebar.subheader("?? AI Confidence")
    ai_conf_threshold = st.sidebar.slider(
        "AI Confidence Threshold",
        min_value=50,
        max_value=90,
        value=int(AI_CONFIDENCE_DEFAULT * 100),
        help="Only mark AI as confident when probability exceeds this threshold",
    ) / 100.0

    st.sidebar.divider()

    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("?? Clear Cache", use_container_width=True, help="Clear cached data and reload"):
            st.cache_data.clear()
            st.sidebar.success("? Cache cleared!")

    with col2:
        run_scan = st.button(
            "?? Run Scan",
            type="primary",
            use_container_width=True,
            key="run_scan_button",
            help="Scan selected universe for qualifying stocks",
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
            "ai_conf_threshold": ai_conf_threshold,
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
                ai_conf_threshold=ai_conf_threshold,
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
            "ai_conf_threshold": ai_conf_threshold,
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
    m1.metric("?? Universe", universe_meta)
    m2.metric("? Scanned", f"{total_scanned}/{universe_size_meta}")
    m3.metric("?? Matched", str(passed_count))
    m4.metric("?? Hit Rate", f"{pass_rate:.1f}%")

    failed_count = universe_size_meta - total_scanned
    if failed_count > 0:
        st.warning(
            f"?? **{failed_count} symbols** could not be loaded due to data unavailability (404/401 errors). "
            f"This is normal for delisted stocks or API rate limits. Scan completed with {total_scanned} available stocks."
        )

    st.caption(
        f"?? Run time: {st.session_state.run_timestamp} | ?? Signal candle: {signal_candle} | "
        f"?? Latest: {latest_available} | ?? Mode: {candle_mode} | "
        f"?? Timeframe: {timeframe_meta} | ?? Source: {universe_source_meta} | ??? Filters: {active_filter_text}"
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
            "AI Confident",
            "Dist MA50 %",
            "Dist MA200 %",
            "EMA Dist %",
            "Volume Ratio",
            "ATR %",
            "Supertrend Signal",
            "ADX",
            "52W High %",
            "52W Low %",
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

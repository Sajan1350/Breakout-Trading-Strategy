"""
Breakout Trading Strategy for Micro Futures - 1 Minute Timeframe
================================================================
Based on the fundamental principles from "The BreakOut Trading Revolution"
by Tomas Nesnidal (Mr. Breakouts Formula).

Formula Components:
  1. Point of Initiation (POI): Previous session's close
  2. Space: POI ± (ATR * FRACT) — volatility-adaptive breakout levels
  3. Filter: ADX trend strength filter + RSI momentum filter
  4. Time Parameter: Restrict entries to high-activity windows
  5. Exits: USD-based stop-loss + USD-based profit target + end-of-day exit
  6. Entry: Stop orders (buy stop above / sell stop below breakout levels)

Optimization Inputs (kept to max 6 per the book's guidance):
  1. fract — ATR multiplier for breakout distance
  2. stop_loss_usd — USD-based protective stop
  3. profit_target_usd — USD-based profit target
  4. adx_threshold — ADX filter threshold
  5. rsi_period — RSI filter lookback
  6. rsi_threshold — RSI overbought/oversold threshold

Designed for: Micro E-mini futures (MES, MNQ, MYM, M2K)
Timeframe: 1-minute bars
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Contract specifications for Micro Futures
# ─────────────────────────────────────────────────────────────
MICRO_CONTRACTS = {
    "MES": {"name": "Micro E-mini S&P 500", "tick_size": 0.25, "tick_value": 1.25, "point_value": 5.0},
    "MNQ": {"name": "Micro E-mini NASDAQ", "tick_size": 0.25, "tick_value": 0.50, "point_value": 2.0},
    "MYM": {"name": "Micro E-mini Dow Jones", "tick_size": 1.0, "tick_value": 0.50, "point_value": 0.50},
    "M2K": {"name": "Micro E-mini Russell 2000", "tick_size": 0.10, "tick_value": 0.50, "point_value": 5.0},
}


class Direction(Enum):
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class StrategyParams:
    """All optimizable and configurable parameters for the breakout strategy."""

    # --- Optimization Inputs (6 max per Mr. Breakouts rule) ---
    fract: float = 1.5                  # ATR multiplier for breakout distance
    stop_loss_usd: float = 50.0         # USD-based stop loss
    profit_target_usd: float = 100.0    # USD-based profit target
    adx_threshold: float = 30.0         # ADX filter: enter only when ADX < threshold (avoid choppy/overextended)
    rsi_period: int = 14                # RSI lookback period
    rsi_threshold: float = 30.0         # RSI threshold for directional filter

    # --- Fixed Parameters (not optimized) ---
    atr_period: int = 25                # ATR lookback (book recommends 5, 25, or 40)
    adx_period: int = 14                # ADX lookback
    poi_type: str = "prev_close"        # Point of Initiation type

    # --- Time Filter (session times in HHMM format, exchange local time) ---
    entry_start_time: int = 930         # Earliest entry time (9:30 AM ET)
    entry_end_time: int = 1130          # Latest entry time (11:30 AM ET — before lunch lull)
    session_close_time: int = 1600      # Force exit by session close (4:00 PM ET)

    # --- Contract ---
    symbol: str = "MES"
    commission_per_trade: float = 1.24  # Round-trip commission for micro futures


@dataclass
class Trade:
    """Record of a completed trade."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: Direction
    entry_price: float
    exit_price: float
    pnl_usd: float
    exit_reason: str


@dataclass
class Position:
    """Tracks an open position."""
    direction: Direction
    entry_price: float
    entry_time: pd.Timestamp
    stop_price: float
    target_price: float


# ─────────────────────────────────────────────────────────────
# Technical Indicators
# ─────────────────────────────────────────────────────────────
def compute_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range = max(H-L, |H-prevC|, |L-prevC|)."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range — the 'Holy Grail Indicator' per the book."""
    tr = compute_true_range(high, low, close)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average Directional Index — trend strength filter."""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = compute_true_range(high, low, close)
    atr = tr.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    plus_di = 100.0 * (plus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr)
    minus_di = 100.0 * (minus_dm.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean() / atr)

    dx = 100.0 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    adx = dx.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    return adx


# ─────────────────────────────────────────────────────────────
# Point of Initiation (POI)
# ─────────────────────────────────────────────────────────────
def compute_poi(df: pd.DataFrame, poi_type: str) -> pd.Series:
    """
    Calculate the Point of Initiation for each bar.
    The POI is recalculated once per session (carried forward intraday).

    Supported types:
      - 'prev_close': Previous session's closing price
      - 'today_open': Current session's opening price
      - 'max_open_prev_close': Max of today's open and prev close
      - 'min_open_prev_close': Min of today's open and prev close
    """
    dates = df.index.date
    date_series = pd.Series(dates, index=df.index)
    unique_dates = np.unique(dates)

    poi = pd.Series(np.nan, index=df.index, dtype=float)

    for i, date in enumerate(unique_dates):
        mask = date_series == date
        day_data = df.loc[mask]

        if poi_type == "prev_close":
            if i > 0:
                prev_mask = date_series == unique_dates[i - 1]
                prev_close = df.loc[prev_mask, "close"].iloc[-1]
                poi.loc[mask] = prev_close

        elif poi_type == "today_open":
            today_open = day_data["open"].iloc[0]
            poi.loc[mask] = today_open

        elif poi_type == "max_open_prev_close":
            today_open = day_data["open"].iloc[0]
            if i > 0:
                prev_mask = date_series == unique_dates[i - 1]
                prev_close = df.loc[prev_mask, "close"].iloc[-1]
                poi.loc[mask] = max(today_open, prev_close)

        elif poi_type == "min_open_prev_close":
            today_open = day_data["open"].iloc[0]
            if i > 0:
                prev_mask = date_series == unique_dates[i - 1]
                prev_close = df.loc[prev_mask, "close"].iloc[-1]
                poi.loc[mask] = min(today_open, prev_close)

    return poi


# ─────────────────────────────────────────────────────────────
# Core Strategy Engine
# ─────────────────────────────────────────────────────────────
class BreakoutStrategy:
    """
    Mr. Breakouts Formula implementation for Micro Futures (1-min bars).

    Entry Logic:
      - Compute breakout levels: POI ± (ATR * fract)
      - LONG: price breaks above upper level via stop order
      - SHORT: price breaks below lower level via stop order
      - Filters must confirm: ADX < threshold (avoid overextended trends)
                              RSI > (100 - rsi_threshold) for longs
                              RSI < rsi_threshold for shorts
      - Time window must be active

    Exit Logic:
      - USD-based stop loss
      - USD-based profit target
      - End-of-day forced exit at session close
    """

    def __init__(self, params: StrategyParams):
        self.params = params
        self.contract = MICRO_CONTRACTS[params.symbol]
        self.point_value = self.contract["point_value"]
        self.tick_size = self.contract["tick_size"]

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all indicator columns to the dataframe."""
        df = df.copy()
        df["atr"] = compute_atr(df["high"], df["low"], df["close"], self.params.atr_period)
        df["rsi"] = compute_rsi(df["close"], self.params.rsi_period)
        df["adx"] = compute_adx(df["high"], df["low"], df["close"], self.params.adx_period)
        df["poi"] = compute_poi(df, self.params.poi_type)

        # Breakout levels: POI + SPACE (long) / POI - SPACE (short)
        df["space"] = df["atr"] * self.params.fract
        df["breakout_long"] = df["poi"] + df["space"]
        df["breakout_short"] = df["poi"] - df["space"]

        # Time filter
        bar_time = df.index.hour * 100 + df.index.minute
        df["time_filter"] = (bar_time >= self.params.entry_start_time) & (bar_time <= self.params.entry_end_time)
        df["is_session_close"] = bar_time >= self.params.session_close_time

        # Directional filters
        df["filter_long"] = (df["adx"] < self.params.adx_threshold) & (df["rsi"] > (100 - self.params.rsi_threshold))
        df["filter_short"] = (df["adx"] < self.params.adx_threshold) & (df["rsi"] < self.params.rsi_threshold)

        return df

    def _price_to_usd(self, price_diff: float) -> float:
        """Convert a price difference to USD using point value."""
        return price_diff * self.point_value

    def _usd_to_price(self, usd_amount: float) -> float:
        """Convert a USD amount to price distance."""
        return usd_amount / self.point_value

    def run_backtest(self, df: pd.DataFrame) -> dict:
        """
        Run the full backtest on prepared 1-minute bar data.

        Parameters
        ----------
        df : pd.DataFrame
            Must have columns: open, high, low, close
            Index must be DatetimeIndex

        Returns
        -------
        dict with keys: trades, equity_curve, stats
        """
        df = self.prepare_data(df)

        trades: list[Trade] = []
        position: Optional[Position] = None
        equity = [0.0]
        equity_times = [df.index[0]]

        stop_distance = self._usd_to_price(self.params.stop_loss_usd)
        target_distance = self._usd_to_price(self.params.profit_target_usd)
        commission = self.params.commission_per_trade

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i - 1]

            # ── Check exits for open position ──
            if position is not None:
                closed = False
                exit_price = 0.0
                exit_reason = ""

                if position.direction == Direction.LONG:
                    # Check stop loss (hit if low <= stop)
                    if row["low"] <= position.stop_price:
                        exit_price = position.stop_price
                        exit_reason = "stop_loss"
                        closed = True
                    # Check profit target (hit if high >= target)
                    elif row["high"] >= position.target_price:
                        exit_price = position.target_price
                        exit_reason = "profit_target"
                        closed = True
                    # End of day exit
                    elif row["is_session_close"]:
                        exit_price = row["close"]
                        exit_reason = "end_of_day"
                        closed = True

                elif position.direction == Direction.SHORT:
                    # Check stop loss (hit if high >= stop)
                    if row["high"] >= position.stop_price:
                        exit_price = position.stop_price
                        exit_reason = "stop_loss"
                        closed = True
                    # Check profit target (hit if low <= target)
                    elif row["low"] <= position.target_price:
                        exit_price = position.target_price
                        exit_reason = "profit_target"
                        closed = True
                    # End of day exit
                    elif row["is_session_close"]:
                        exit_price = row["close"]
                        exit_reason = "end_of_day"
                        closed = True

                if closed:
                    if position.direction == Direction.LONG:
                        pnl = self._price_to_usd(exit_price - position.entry_price) - commission
                    else:
                        pnl = self._price_to_usd(position.entry_price - exit_price) - commission

                    trade = Trade(
                        entry_time=position.entry_time,
                        exit_time=df.index[i],
                        direction=position.direction,
                        entry_price=position.entry_price,
                        exit_price=exit_price,
                        pnl_usd=pnl,
                        exit_reason=exit_reason,
                    )
                    trades.append(trade)
                    equity.append(equity[-1] + pnl)
                    equity_times.append(df.index[i])
                    position = None
                    continue

            # ── Check entries (only if flat) ──
            if position is None and row["time_filter"]:
                # Check for LONG breakout via stop order:
                # Price must trade through the breakout level (high >= breakout_long)
                if (not pd.isna(prev["breakout_long"])) and prev["filter_long"] and row["high"] >= prev["breakout_long"]:
                    entry_price = prev["breakout_long"]
                    # If bar opened above breakout, entry is at open (gap scenario)
                    if row["open"] >= prev["breakout_long"]:
                        entry_price = row["open"]

                    position = Position(
                        direction=Direction.LONG,
                        entry_price=entry_price,
                        entry_time=df.index[i],
                        stop_price=entry_price - stop_distance,
                        target_price=entry_price + target_distance,
                    )

                # Check for SHORT breakout via stop order:
                elif (not pd.isna(prev["breakout_short"])) and prev["filter_short"] and row["low"] <= prev["breakout_short"]:
                    entry_price = prev["breakout_short"]
                    # If bar opened below breakout, entry is at open (gap scenario)
                    if row["open"] <= prev["breakout_short"]:
                        entry_price = row["open"]

                    position = Position(
                        direction=Direction.SHORT,
                        entry_price=entry_price,
                        entry_time=df.index[i],
                        stop_price=entry_price + stop_distance,
                        target_price=entry_price - target_distance,
                    )

        # Close any remaining position at last bar
        if position is not None:
            last = df.iloc[-1]
            if position.direction == Direction.LONG:
                pnl = self._price_to_usd(last["close"] - position.entry_price) - commission
            else:
                pnl = self._price_to_usd(position.entry_price - last["close"]) - commission
            trades.append(Trade(
                entry_time=position.entry_time,
                exit_time=df.index[-1],
                direction=position.direction,
                entry_price=position.entry_price,
                exit_price=last["close"],
                pnl_usd=pnl,
                exit_reason="end_of_data",
            ))
            equity.append(equity[-1] + pnl)
            equity_times.append(df.index[-1])

        # Build results
        equity_curve = pd.Series(equity, index=equity_times)
        stats = self._compute_stats(trades, equity_curve)

        return {"trades": trades, "equity_curve": equity_curve, "stats": stats}

    def _compute_stats(self, trades: list[Trade], equity_curve: pd.Series) -> dict:
        """Compute performance statistics."""
        if not trades:
            return {"total_trades": 0, "net_profit": 0.0, "message": "No trades generated"}

        pnls = [t.pnl_usd for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]

        net_profit = sum(pnls)
        gross_profit = sum(winners) if winners else 0
        gross_loss = sum(losers) if losers else 0
        win_rate = len(winners) / len(pnls) * 100
        avg_trade = net_profit / len(pnls)
        avg_winner = np.mean(winners) if winners else 0
        avg_loser = np.mean(losers) if losers else 0
        profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else float("inf")

        # Max drawdown
        running_max = equity_curve.cummax()
        drawdown = equity_curve - running_max
        max_drawdown = drawdown.min()

        # Max drawdown duration
        in_dd = drawdown < 0
        dd_groups = (~in_dd).cumsum()
        if in_dd.any():
            dd_durations = in_dd.groupby(dd_groups).sum()
            max_dd_bars = dd_durations.max()
        else:
            max_dd_bars = 0

        # Exit reason breakdown
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        # Direction breakdown
        long_trades = [t for t in trades if t.direction == Direction.LONG]
        short_trades = [t for t in trades if t.direction == Direction.SHORT]
        long_pnl = sum(t.pnl_usd for t in long_trades)
        short_pnl = sum(t.pnl_usd for t in short_trades)

        return {
            "total_trades": len(pnls),
            "winners": len(winners),
            "losers": len(losers),
            "win_rate_pct": round(win_rate, 2),
            "net_profit_usd": round(net_profit, 2),
            "gross_profit_usd": round(gross_profit, 2),
            "gross_loss_usd": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 3),
            "avg_trade_usd": round(avg_trade, 2),
            "avg_winner_usd": round(avg_winner, 2),
            "avg_loser_usd": round(avg_loser, 2),
            "max_drawdown_usd": round(max_drawdown, 2),
            "max_dd_bars": int(max_dd_bars),
            "net_profit_dd_ratio": round(abs(net_profit / max_drawdown), 2) if max_drawdown != 0 else float("inf"),
            "long_trades": len(long_trades),
            "short_trades": len(short_trades),
            "long_pnl_usd": round(long_pnl, 2),
            "short_pnl_usd": round(short_pnl, 2),
            "exit_reasons": exit_reasons,
        }


# ─────────────────────────────────────────────────────────────
# Parameter Optimization (Grid Search)
# ─────────────────────────────────────────────────────────────
def optimize_strategy(df: pd.DataFrame, symbol: str = "MES",
                      param_grid: Optional[dict] = None) -> list[dict]:
    """
    Run a grid search over parameter combinations.
    Returns results sorted by Net Profit / Max Drawdown ratio (descending).
    """
    if param_grid is None:
        param_grid = {
            "fract": [1.0, 1.5, 2.0, 2.5],
            "stop_loss_usd": [30, 50, 75],
            "profit_target_usd": [60, 100, 150],
            "adx_threshold": [25, 35, 45],
        }

    # Build all combinations
    from itertools import product
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combos = list(product(*values))

    results = []
    total = len(combos)
    logger.info(f"Optimizing {total} parameter combinations for {symbol}...")

    for idx, combo in enumerate(combos):
        param_dict = dict(zip(keys, combo))
        params = StrategyParams(symbol=symbol, **param_dict)
        strategy = BreakoutStrategy(params)
        result = strategy.run_backtest(df)
        stats = result["stats"]

        if stats["total_trades"] > 0:
            results.append({
                "params": param_dict,
                "total_trades": stats["total_trades"],
                "net_profit": stats["net_profit_usd"],
                "max_drawdown": stats["max_drawdown_usd"],
                "profit_factor": stats["profit_factor"],
                "win_rate": stats["win_rate_pct"],
                "avg_trade": stats["avg_trade_usd"],
                "np_dd_ratio": stats["net_profit_dd_ratio"],
            })

        if (idx + 1) % 25 == 0:
            logger.info(f"  Progress: {idx + 1}/{total}")

    results.sort(key=lambda x: x["np_dd_ratio"], reverse=True)
    logger.info(f"Optimization complete. {len(results)} viable combinations found.")
    return results


# ─────────────────────────────────────────────────────────────
# Sample Data Generator (for testing without live data)
# ─────────────────────────────────────────────────────────────
def generate_sample_data(symbol: str = "MES", days: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic 1-minute OHLC data for testing.
    Simulates realistic intraday price movement with session open/close.
    """
    np.random.seed(seed)
    contract = MICRO_CONTRACTS[symbol]
    tick = contract["tick_size"]

    base_prices = {"MES": 5000.0, "MNQ": 17500.0, "MYM": 38000.0, "M2K": 2000.0}
    base = base_prices.get(symbol, 5000.0)

    records = []
    price = base

    for day in range(days):
        date = pd.Timestamp("2024-01-02") + pd.Timedelta(days=day)
        if date.weekday() >= 5:  # Skip weekends
            continue

        # Regular trading hours: 9:30 AM - 4:00 PM ET (391 one-minute bars)
        session_start = date.replace(hour=9, minute=30)

        for minute in range(391):
            ts = session_start + pd.Timedelta(minutes=minute)
            current_hour = ts.hour + ts.minute / 60.0

            # Higher volatility at open and close, lower at lunch
            if current_hour < 10.5:
                vol_mult = 2.0  # Opening volatility
            elif current_hour < 12.0:
                vol_mult = 1.0
            elif current_hour < 13.5:
                vol_mult = 0.5  # Lunch lull
            elif current_hour < 15.0:
                vol_mult = 1.0
            else:
                vol_mult = 1.5  # Closing volatility

            # Simulate micro-level price movement
            noise = np.random.normal(0, tick * vol_mult * 3)
            trend = tick * 0.02 * np.random.choice([-1, 1])
            price += noise + trend

            bar_range = abs(np.random.normal(0, tick * vol_mult * 4))
            o = round(price / tick) * tick
            h = o + abs(np.random.normal(0, bar_range))
            l = o - abs(np.random.normal(0, bar_range))
            c = round((l + np.random.random() * (h - l)) / tick) * tick
            h = round(h / tick) * tick
            l = round(l / tick) * tick

            if h < max(o, c):
                h = max(o, c)
            if l > min(o, c):
                l = min(o, c)

            price = c
            records.append({"datetime": ts, "open": o, "high": h, "low": l, "close": c})

    df = pd.DataFrame(records)
    df.set_index("datetime", inplace=True)
    df.index = pd.DatetimeIndex(df.index)
    return df


# ─────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────
def print_report(stats: dict, params: StrategyParams):
    """Print a formatted backtest report."""
    print("\n" + "=" * 65)
    print(f"  BREAKOUT STRATEGY BACKTEST REPORT — {params.symbol}")
    print(f"  Mr. Breakouts Formula | 1-Minute Timeframe")
    print("=" * 65)

    print(f"\n  Contract:          {MICRO_CONTRACTS[params.symbol]['name']}")
    print(f"  POI Type:          {params.poi_type}")
    print(f"  ATR Period:        {params.atr_period}")
    print(f"  FRACT:             {params.fract}")
    print(f"  ADX Threshold:     {params.adx_threshold}")
    print(f"  RSI Period:        {params.rsi_period}")
    print(f"  RSI Threshold:     {params.rsi_threshold}")
    print(f"  Stop Loss:         ${params.stop_loss_usd}")
    print(f"  Profit Target:     ${params.profit_target_usd}")
    print(f"  Entry Window:      {params.entry_start_time} - {params.entry_end_time}")
    print(f"  Commission:        ${params.commission_per_trade}/trade")

    if stats["total_trades"] == 0:
        print("\n  No trades generated.")
        print("=" * 65)
        return

    print(f"\n  --- Performance ---")
    print(f"  Total Trades:      {stats['total_trades']}")
    print(f"  Winners:           {stats['winners']}  ({stats['win_rate_pct']}%)")
    print(f"  Losers:            {stats['losers']}")
    print(f"  Net Profit:        ${stats['net_profit_usd']:,.2f}")
    print(f"  Gross Profit:      ${stats['gross_profit_usd']:,.2f}")
    print(f"  Gross Loss:        ${stats['gross_loss_usd']:,.2f}")
    print(f"  Profit Factor:     {stats['profit_factor']}")
    print(f"  Avg Trade:         ${stats['avg_trade_usd']:.2f}")
    print(f"  Avg Winner:        ${stats['avg_winner_usd']:.2f}")
    print(f"  Avg Loser:         ${stats['avg_loser_usd']:.2f}")
    print(f"  Max Drawdown:      ${stats['max_drawdown_usd']:,.2f}")
    print(f"  NP/DD Ratio:       {stats['net_profit_dd_ratio']}")

    print(f"\n  --- Direction Breakdown ---")
    print(f"  Long Trades:       {stats['long_trades']}   (PnL: ${stats['long_pnl_usd']:,.2f})")
    print(f"  Short Trades:      {stats['short_trades']}   (PnL: ${stats['short_pnl_usd']:,.2f})")

    print(f"\n  --- Exit Reasons ---")
    for reason, count in stats["exit_reasons"].items():
        print(f"  {reason:20s} {count}")

    print("=" * 65)


# ─────────────────────────────────────────────────────────────
# Main — Demo Run
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating sample 1-minute data for MES (Micro E-mini S&P 500)...")
    df = generate_sample_data(symbol="MES", days=60, seed=42)
    print(f"Data: {len(df)} bars, from {df.index[0]} to {df.index[-1]}")

    # Run with default parameters
    params = StrategyParams(
        symbol="MES",
        fract=1.5,
        stop_loss_usd=50.0,
        profit_target_usd=100.0,
        adx_threshold=30.0,
        rsi_period=14,
        rsi_threshold=30.0,
        atr_period=25,
        poi_type="prev_close",
        entry_start_time=930,
        entry_end_time=1130,
    )

    strategy = BreakoutStrategy(params)
    result = strategy.run_backtest(df)
    print_report(result["stats"], params)

    # Show a few sample trades
    if result["trades"]:
        print(f"\n  Sample Trades (first 10):")
        print(f"  {'Entry Time':22s} {'Dir':6s} {'Entry':>10s} {'Exit':>10s} {'PnL':>10s} {'Reason'}")
        print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*10} {'-'*10} {'-'*15}")
        for t in result["trades"][:10]:
            d = "LONG" if t.direction == Direction.LONG else "SHORT"
            print(f"  {str(t.entry_time):22s} {d:6s} {t.entry_price:10.2f} {t.exit_price:10.2f} {t.pnl_usd:10.2f} {t.exit_reason}")

    # Quick optimization demo
    print("\n\nRunning parameter optimization...")
    opt_results = optimize_strategy(df, symbol="MES", param_grid={
        "fract": [1.0, 1.5, 2.0],
        "stop_loss_usd": [40, 60],
        "profit_target_usd": [80, 120],
        "adx_threshold": [25, 35],
    })

    if opt_results:
        print(f"\n  Top 5 Parameter Combinations (by NP/DD Ratio):")
        print(f"  {'Rank':>4s} {'Trades':>7s} {'Net PnL':>10s} {'MaxDD':>10s} {'PF':>6s} {'WR%':>6s} {'Params'}")
        print(f"  {'-'*4} {'-'*7} {'-'*10} {'-'*10} {'-'*6} {'-'*6} {'-'*40}")
        for i, r in enumerate(opt_results[:5]):
            print(f"  {i+1:4d} {r['total_trades']:7d} {r['net_profit']:10.2f} "
                  f"{r['max_drawdown']:10.2f} {r['profit_factor']:6.2f} "
                  f"{r['win_rate']:6.1f} {r['params']}")

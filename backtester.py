import pandas as pd
import numpy as np

def get_trading_day(current_date, offset, unique_trading_days):
    matches = np.where(unique_trading_days == current_date)[0]
    if len(matches) == 0:
        raise ValueError(f"current_date {current_date} not found in unique_trading_days.")
    current_idx = matches[0]
    target_idx = current_idx + offset
    if target_idx >= len(unique_trading_days):
        return unique_trading_days[-1]
    if target_idx < 0:
        return unique_trading_days[0]
    return unique_trading_days[target_idx]

def get_atm_straddle(chain_t0, target_ttm=30/365.0):
    chain_t0 = chain_t0.copy()
    chain_t0['ttm_diff'] = abs(chain_t0['TTM'] - target_ttm)
    best_expiration = chain_t0.loc[chain_t0['ttm_diff'].idxmin()]['exdate']
    
    target_chain = chain_t0[chain_t0['exdate'] == best_expiration].copy()
    target_chain['moneyness_diff'] = abs(target_chain['Moneyness'] - 1.0)
    best_strike = target_chain.loc[target_chain['moneyness_diff'].idxmin()]['strike_price']
    
    atm_call = target_chain[(target_chain['strike_price'] == best_strike) & (target_chain['cp_flag'] == 'C')]
    atm_put = target_chain[(target_chain['strike_price'] == best_strike) & (target_chain['cp_flag'] == 'P')]

    if atm_call.empty or atm_put.empty:
        raise ValueError(
            f"Missing call or put leg at ATM strike {best_strike} for expiration {best_expiration}."
        )

    return atm_call.iloc[0], atm_put.iloc[0]

def calculate_5_day_straddle_pnl(signal_df, option_data, stock_data, hold_days=5, target_notional=None):
    """
    Backtests a delta-hedged straddle held for `hold_days` trading days.

    target_notional: if given, position size (contracts) is scaled per
    trade as target_notional / (entry_stock_price * 100), so dollar
    exposure is roughly constant across tickers of very different share
    prices instead of a flat 1 contract every time (which lets a handful
    of expensive/volatile names dominate tail risk). If None, sizing is
    a flat 1 contract, matching the original behavior.

    signal_df may optionally include a 'Residual' column (actual minus
    model-reconstructed implied vol at the near-ATM, near-term grid
    point). When present, a trade is shorted instead of bought whenever
    Residual > 0 -- i.e. the flagged surface looks richer than the
    autoencoder's typical reconstruction, so selling vol rather than
    buying it is the theoretically consistent side. In this mid-price,
    no-transaction-cost model, a short straddle's PnL is exactly the
    negative of the equivalent long straddle's PnL for the same trade
    parameters, so the sign is simply flipped rather than re-deriving
    the hedge mechanics. Without a 'Residual' column, every trade is
    long, matching the original behavior.
    """
    trade_results = []
    unique_trading_days = np.sort(stock_data['date'].unique())

    # Pre-index by (date, ticker) once, instead of re-scanning the full
    # option_data / stock_data tables on every lookup inside the trade
    # loop below. On a multi-million-row options table, a fresh boolean
    # mask over the whole frame per lookup is the dominant cost of this
    # function; a dict lookup is effectively free by comparison.
    option_chains = {key: group for key, group in option_data.groupby(['date', 'ticker'])}
    stock_prices = stock_data.set_index(['date', 'ticker'])['close'].to_dict()
    has_direction = 'Residual' in signal_df.columns

    for _, signal in signal_df.iterrows():
        entry_date = signal['Date']
        ticker = signal['Ticker']

        chain_t0 = option_chains.get((entry_date, ticker))
        if chain_t0 is None or chain_t0.empty:
            continue

        try:
            call_t0, put_t0 = get_atm_straddle(chain_t0)
        except ValueError:
            continue

        stock_price_t0 = stock_prices.get((entry_date, ticker))
        if stock_price_t0 is None:
            continue

        contracts = 1.0 if target_notional is None else target_notional / (stock_price_t0 * 100.0)

        initial_cost = (call_t0['Premium'] + put_t0['Premium']) * contracts
        net_delta = call_t0['delta'] + put_t0['delta']
        shares_held = -net_delta * 100 * contracts
        cash_flow = -(shares_held * stock_price_t0)

        for i in range(1, hold_days):
            current_date = get_trading_day(entry_date, offset=i, unique_trading_days=unique_trading_days)
            chain_t = option_chains.get((current_date, ticker))
            if chain_t is None or chain_t.empty:
                continue

            call_t = chain_t[(chain_t['strike_price'] == call_t0['strike_price']) & (chain_t['cp_flag'] == 'C')]
            put_t = chain_t[(chain_t['strike_price'] == put_t0['strike_price']) & (chain_t['cp_flag'] == 'P')]
            if call_t.empty or put_t.empty:
                continue

            stock_price_t = stock_prices.get((current_date, ticker))
            if stock_price_t is None:
                continue

            new_net_delta = call_t['delta'].values[0] + put_t['delta'].values[0]
            target_shares = -new_net_delta * 100 * contracts

            shares_traded = target_shares - shares_held
            cash_flow -= (shares_traded * stock_price_t)
            shares_held = target_shares

        exit_date = get_trading_day(entry_date, offset=hold_days, unique_trading_days=unique_trading_days)
        chain_texit = option_chains.get((exit_date, ticker))
        if chain_texit is None or chain_texit.empty:
            continue

        call_texit = chain_texit[(chain_texit['strike_price'] == call_t0['strike_price']) & (chain_texit['cp_flag'] == 'C')]
        put_texit = chain_texit[(chain_texit['strike_price'] == put_t0['strike_price']) & (chain_texit['cp_flag'] == 'P')]
        if call_texit.empty or put_texit.empty:
            continue

        stock_price_texit = stock_prices.get((exit_date, ticker))
        if stock_price_texit is None:
            continue

        final_option_revenue = (call_texit['Premium'].values[0] + put_texit['Premium'].values[0]) * contracts
        cash_flow += (shares_held * stock_price_texit)

        pnl_long = final_option_revenue - initial_cost + cash_flow

        if has_direction and signal['Residual'] > 0:
            direction = 'short'
            total_pnl = -pnl_long
        else:
            direction = 'long'
            total_pnl = pnl_long

        # Capital required to put the trade on: the straddle premium plus
        # the dollar notional of the stock hedge established alongside it.
        # A simplification (no formal margin model for the short side), but
        # consistent with this backtester's mid-price, no-transaction-cost
        # approach elsewhere.
        capital_allocated = initial_cost + abs(shares_held * stock_price_t0)

        trade_results.append({
            'Date': entry_date, 'Ticker': ticker, 'PnL': total_pnl, 'Direction': direction,
            'Capital_Allocated': capital_allocated,
        })

    return pd.DataFrame(trade_results)

def calculate_performance_metrics(trade_results, risk_free_rate=0.04, total_portfolio_capital=None):
    """
    Vectorized performance tear sheet for a completed set of trades.

    trade_results columns:
        date              -- liquidation (exit) date
        ticker            -- underlying ticker
        sector            -- GICS sector
        net_pnl           -- dollar PnL of the trade
        capital_allocated -- capital at risk for the trade (straddle + hedge)

    total_portfolio_capital: if given, every day's return is computed as
    that day's dollar PnL divided by this fixed capital base, instead of
    dividing by that day's own pooled capital_allocated. This matters
    because capital_allocated is a notional exposure estimate, not a hard
    loss cap -- a delta hedge can slip badly enough on a gap day that
    realized losses exceed the notional tied up in that day's trades,
    especially on a day with only one or two low-notional trades (small
    denominator). That produces a daily return below -100%, which flips
    the sign of the compounding equity curve and makes Total Cumulative
    Return / Max Drawdown meaningless from that day forward. A fixed
    capital base (reflecting a real account's total capital, not just
    what happened to be in that specific day's trades) avoids this. If
    None (default), uses each day's own pooled capital_allocated, which
    can still hit that breakdown.

    A day's portfolio return pools every trade closing that day (capital-
    weighted when total_portfolio_capital is None, since multiple trades
    can liquidate on the same date). Return recognition happens on the
    exit day, not as true day-by-day mark-to-market over the hold period
    -- an approximation forced by the schema (only a liquidation date is
    available, no separate entry-date capital-lockup column).

    Returns (summary, sector_summary):
        summary: dict with Total Cumulative Return (%), Win Rate (%),
            Average PnL per Trade ($), Annualized Sharpe Ratio, and
            Max Drawdown (%) (reported as a positive magnitude).
        sector_summary: DataFrame indexed by sector with Sharpe Ratio,
            Win Rate (%), Total PnL ($), and Trade Count per sector.
    """
    df = trade_results.copy()
    df['date'] = pd.to_datetime(df['date'])
    daily_rf = risk_free_rate / 252.0

    def _sharpe(mean_return, std_return, n):
        if n <= 1 or std_return == 0 or pd.isna(std_return):
            return np.nan
        return (mean_return - daily_rf) / std_return * np.sqrt(252)

    def _pooled_daily_returns(group_keys):
        daily = df.groupby(group_keys).agg(pnl=('net_pnl', 'sum'), capital=('capital_allocated', 'sum'))
        if total_portfolio_capital is None:
            daily = daily[daily['capital'] != 0]
            returns = daily['pnl'] / daily['capital']
        else:
            returns = daily['pnl'] / total_portfolio_capital
        return returns.sort_index()

    # --- Portfolio-level daily returns, equity curve, and drawdown -------
    daily_returns = _pooled_daily_returns('date')

    equity_curve = (1.0 + daily_returns).cumprod()
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max

    total_cumulative_return_pct = (equity_curve.iloc[-1] - 1.0) * 100 if len(equity_curve) else 0.0
    max_drawdown_pct = abs(drawdown.min()) * 100 if len(drawdown) else 0.0
    sharpe_ratio = _sharpe(daily_returns.mean(), daily_returns.std(), len(daily_returns))

    summary = {
        'Total Cumulative Return (%)': total_cumulative_return_pct,
        'Win Rate (%)': (df['net_pnl'] > 0).mean() * 100,
        'Average PnL per Trade ($)': df['net_pnl'].mean(),
        'Annualized Sharpe Ratio': sharpe_ratio,
        'Max Drawdown (%)': max_drawdown_pct,
    }

    # --- Sector breakdown, fully vectorized (no explicit per-sector loop) --
    sector_daily_returns = _pooled_daily_returns(['sector', 'date'])

    sector_stats = sector_daily_returns.groupby(level='sector').agg(mean='mean', std='std', n='count')
    sector_sharpe = pd.Series(
        np.vectorize(_sharpe, otypes=[float])(sector_stats['mean'], sector_stats['std'], sector_stats['n']),
        index=sector_stats.index,
    )

    sector_summary = df.groupby('sector').agg(**{
        'Win Rate (%)': ('net_pnl', lambda s: (s > 0).mean() * 100),
        'Total PnL ($)': ('net_pnl', 'sum'),
        'Trade Count': ('net_pnl', 'size'),
    })
    sector_summary['Sharpe Ratio'] = sector_sharpe
    sector_summary = sector_summary[['Sharpe Ratio', 'Win Rate (%)', 'Total PnL ($)', 'Trade Count']]

    return summary, sector_summary
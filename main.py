import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import components from our modular files
from data_pipeline import build_options_environment, interpolate_surface_arrays, atm_flat_index
from models import VolatilityAutoencoder, EarlyStopping
from backtester import calculate_5_day_straddle_pnl, calculate_performance_metrics

# Backtest configuration, tuned via historical analysis (see project notes):
# - HOLD_DAYS shortened from 5 to 3: most tail losses came from gap/jump
#   risk accumulating over the hold window that daily rehedging can't
#   track; a shorter window cuts that exposure substantially.
# - TARGET_NOTIONAL_PER_TRADE: dollar notional per trade, sized so
#   expensive/volatile names (e.g. TSLA at $1,500+/share) don't dominate
#   tail risk the way a flat 1-contract (100 share hedge) position does.
#   Chosen to match the median per-trade notional a flat 1-contract
#   position implied historically, for a fair, validated comparison.
HOLD_DAYS = 3
TARGET_NOTIONAL_PER_TRADE = 6887.75

# Exclude surfaces where standardize_surface had to fall back to flat
# nearest-neighbor extrapolation across nearly the entire grid (empirically,
# fallback_fraction >= 0.95 for only ~2% of days -- these have essentially
# no real quote coverage and can't represent a genuine structural anomaly).
# Note: at this grid's resolution (10 moneyness x 5 TTM points out to 180
# days), *some* fallback reliance is normal, not a data-quality problem --
# the median day across the full dataset sits around 40% fallback, so a
# more aggressive threshold would discard the majority of usable signal.
MAX_FALLBACK_FRACTION = 0.95

# Without a fixed seed, VolatilityAutoencoder() starts from different
# random weights on every run, which changes reconstruction error, which
# changes anomaly rankings and the sign of Residual (long vs. short) --
# so two runs on identical data can silently produce different trades and
# different PnL even with zero code changes. Fixing the seed makes runs
# reproducible, which matters for comparing changes (hold length, sizing,
# etc.) apples-to-apples instead of partly measuring random init noise.
RANDOM_SEED = 42

# Cap worker processes rather than claiming every core. This matters on
# WSL/remote-dev boxes where .vscode-server and other tooling run on the
# same machine and compete for CPU with the pool -- saturating every core
# can starve the editor's server process badly enough that it looks like
# a crash. Leave a couple cores free and don't scale unboundedly on big
# machines either (more workers than ~6-8 buys little once each is
# spending most of its time on ~1ms interpolation calls anyway).
RESERVED_CORES = 2
MAX_WORKERS = 6

def process_single_surface(task):
    """
    Top-level worker for ProcessPoolExecutor -- must stay at module scope
    so it can be pickled and imported by worker processes (required on
    spawn-based platforms; harmless on fork-based ones like Linux).

    `task` is a lightweight tuple of plain numpy arrays and scalars, never
    a pandas DataFrame slice: with ~tens of thousands of tasks, pickling
    whole DataFrame groups (index, dtypes, block manager) per task would
    make inter-process serialization overhead dwarf the ~1ms of actual
    interpolation work each one does.

    Returns (metadata_dict, surface_array), the same shape of information
    the old sequential loop produced per group, so the caller can build
    metadata_df / master_tensors identically either way.
    """
    date, ticker, moneyness_arr, ttm_arr, vol_arr, target_moneyness, target_ttm = task
    known_points = np.column_stack([moneyness_arr, ttm_arr])
    surface_array, fallback_fraction = interpolate_surface_arrays(
        known_points, vol_arr, target_moneyness, target_ttm, return_quality=True,
    )
    metadata_entry = {'Date': date, 'Ticker': ticker, 'Fallback_Fraction': fallback_fraction}
    return metadata_entry, surface_array

def main(options_path='wrds_options_raw.csv', stock_path='wrds_stock_raw.csv',
         sector_map_path='ticker_sector_map.csv'):
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Building options environment...")
    clean_options_df = build_options_environment(options_path, stock_path)

    target_moneyness = np.linspace(0.8, 1.2, 10)
    target_ttm = np.array([30, 60, 90, 120, 180]) / 365.0

    print("Preparing interpolation tasks...")
    # ngroups alone is cheap (just the grouping keys, not the per-group
    # column data), so we can size the pool/chunksize without building
    # every task up front.
    grouped = clean_options_df.groupby(['date', 'ticker'])
    n_groups = grouped.ngroups

    def build_tasks():
        # A generator, not a materialized list: clean_options_df is still
        # needed later (for the backtest), so it can't be freed here --
        # the memory win is not holding a second near-complete copy of its
        # numeric columns (as a list of per-group numpy arrays) alongside
        # it at the same time. executor.map() below consumes this
        # incrementally in chunksize-sized batches rather than all at once.
        for (date, ticker), group in grouped:
            yield (date, ticker, group['Moneyness'].to_numpy(), group['TTM'].to_numpy(),
                   group['impl_volatility'].to_numpy(), target_moneyness, target_ttm)

    n_workers = max(1, min(MAX_WORKERS, (os.cpu_count() or 1) - RESERVED_CORES))
    # Each task is only ~1ms of actual work, so dispatching one task per
    # IPC round-trip would let pool overhead dominate; batch many tasks
    # per worker fetch instead.
    chunksize = max(1, n_groups // (n_workers * 4))
    print(f"Interpolating {n_groups} surfaces across {n_workers} worker processes "
          f"(chunksize={chunksize})...")

    tensor_list = []
    metadata = []
    skipped_low_quality = 0

    # Use 'spawn' rather than the platform default ('fork' on Linux):
    # this process is multi-threaded (PyTorch/BLAS run their own internal
    # thread pools), and fork()ing a multi-threaded process can silently
    # deadlock a child if another thread held a lock at fork time. spawn
    # starts each worker as a genuinely fresh interpreter -- slightly
    # slower to launch (paid once for the whole pool, not per task), but
    # immune to that failure mode.
    mp_context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
        # .map preserves input order in its results, so master_tensors and
        # metadata_df stay aligned exactly as they would from the
        # sequential loop -- no extra bookkeeping needed to re-sort.
        for metadata_entry, surface_array in executor.map(process_single_surface, build_tasks(), chunksize=chunksize):
            if metadata_entry['Fallback_Fraction'] >= MAX_FALLBACK_FRACTION:
                skipped_low_quality += 1
                continue
            tensor_list.append(surface_array)
            metadata.append(metadata_entry)

    if skipped_low_quality:
        print(f"Skipped {skipped_low_quality} near-fully-extrapolated surfaces "
              f"(fallback_fraction >= {MAX_FALLBACK_FRACTION}).")

    if len(tensor_list) == 0:
        raise ValueError(
            "No usable surfaces remain after quality filtering -- every chain had "
            f"fallback_fraction >= {MAX_FALLBACK_FRACTION}. Check the input data "
            "(e.g. missing expirations causing collinear chains) or loosen "
            "MAX_FALLBACK_FRACTION."
        )

    master_tensors = torch.FloatTensor(np.array(tensor_list))
    metadata_df = pd.DataFrame(metadata)
    
    print("Training autoencoder...")
    split_idx = int(len(master_tensors) * 0.8)
    train_dataset = TensorDataset(master_tensors[:split_idx])
    val_dataset = TensorDataset(master_tensors[split_idx:])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = VolatilityAutoencoder()
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    stopper = EarlyStopping(patience=10)
    
    for epoch in range(500):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            output = model(batch[0])
            loss = criterion(output, batch[0])
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_output = model(batch[0])
                val_loss += criterion(val_output, batch[0]).item()
                
        stopper(val_loss, model)
        if stopper.early_stop:
            print(f"Early stopping triggered at epoch {epoch}")
            break
            
    model.load_state_dict(torch.load('optimal_autoencoder.pth', weights_only=True))
    
    print("Scoring anomalies...")
    model.eval()
    scoring_criterion = nn.MSELoss(reduction='none')
    flat_idx = atm_flat_index(target_moneyness, target_ttm)

    with torch.no_grad():
        reconstructed = model(master_tensors)
        mse_scores = scoring_criterion(reconstructed, master_tensors).mean(dim=1).numpy()
        # Signed residual at the near-ATM, near-term grid point: actual IV
        # minus what the model typically reconstructs there. Positive means
        # the flagged surface looks richer than the model's norm (sell
        # vol); negative means it looks cheaper (buy vol). See backtester's
        # calculate_5_day_straddle_pnl for how this drives trade direction.
        atm_residuals = (master_tensors[:, flat_idx] - reconstructed[:, flat_idx]).numpy()

    metadata_df['Anomaly_Score'] = mse_scores
    metadata_df['Residual'] = atm_residuals

    print("Sorting portfolios...")
    signals = []
    for date, group in metadata_df.groupby('Date'):
        threshold = group['Anomaly_Score'].quantile(0.90)
        signals.append(group[group['Anomaly_Score'] >= threshold])

    signal_df = pd.concat(signals)[['Date', 'Ticker', 'Residual']]

    print(f"Running {HOLD_DAYS}-day delta-hedged, direction-aware straddle backtest...")
    stock_data = pd.read_csv(stock_path)
    stock_data['date'] = pd.to_datetime(stock_data['date'])

    final_pnl_df = calculate_5_day_straddle_pnl(
        signal_df, clean_options_df, stock_data,
        hold_days=HOLD_DAYS, target_notional=TARGET_NOTIONAL_PER_TRADE,
    )

    average_trade_pnl = final_pnl_df['PnL'].mean()
    win_rate = (final_pnl_df['PnL'] > 0).mean()

    print(f"Average PnL per Trade: ${average_trade_pnl:.2f}")
    print(f"Strategy Win Rate: {win_rate * 100:.2f}%")
    final_pnl_df.to_csv('strategy_results.csv', index=False)

    print("Computing performance tear sheet...")
    trade_results = final_pnl_df.rename(columns={
        'Date': 'date', 'Ticker': 'ticker', 'PnL': 'net_pnl', 'Capital_Allocated': 'capital_allocated',
    })
    sector_map = pd.read_csv(sector_map_path)
    trade_results = trade_results.merge(sector_map, on='ticker', how='left')

    unmapped_tickers = sorted(trade_results.loc[trade_results['sector'].isna(), 'ticker'].unique())
    if unmapped_tickers:
        print(f"WARNING: no sector mapping for {len(unmapped_tickers)} ticker(s): {unmapped_tickers}. "
              "Grouping them under 'Unknown' in the sector breakdown.")
        trade_results['sector'] = trade_results['sector'].fillna('Unknown')

    summary, sector_summary = calculate_performance_metrics(trade_results)

    print()
    print("=== Performance Summary ===")
    for key, value in summary.items():
        if isinstance(value, float) and pd.isna(value):
            print(f"{key}: N/A")
        else:
            print(f"{key}: {value:.4f}")

    print()
    print("=== Sector Breakdown ===")
    print(sector_summary.round(4))

    sector_summary.to_csv('sector_performance.csv')
    print("Saved sector breakdown to sector_performance.csv")

if __name__ == "__main__":
    main()
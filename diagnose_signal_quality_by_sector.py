"""
Checks whether reconstruction error (Anomaly_Score) and anomaly-flagging
rate are elevated in specific sectors -- particularly the low-liquidity
ones (Utilities, Materials, Real Estate) that diagnose_surface_quality.py
found have much higher average fallback_fraction. If those sectors also
get flagged as "anomalies" disproportionately often relative to their
share of the candidate population, that's evidence the signal there is
partly driven by interpolation noise rather than genuine mispricing.

Reuses the real trained model checkpoint (optimal_autoencoder.pth) and
the real pipeline code (process_single_surface, build_options_environment,
atm_flat_index) rather than retraining or reimplementing any of it.

Usage: python diagnose_signal_quality_by_sector.py
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_pipeline import build_options_environment, atm_flat_index
from models import VolatilityAutoencoder
from main import process_single_surface, MAX_FALLBACK_FRACTION, RESERVED_CORES, MAX_WORKERS

OPTIONS_PATH = 'wrds_options_raw.csv'
STOCK_PATH = 'wrds_stock_raw.csv'
SECTOR_MAP_PATH = 'ticker_sector_map.csv'
CHECKPOINT_PATH = 'optimal_autoencoder.pth'

def main():
    print("Building options environment...")
    clean_options_df = build_options_environment(OPTIONS_PATH, STOCK_PATH)

    target_moneyness = np.linspace(0.8, 1.2, 10)
    target_ttm = np.array([30, 60, 90, 120, 180]) / 365.0

    grouped = clean_options_df.groupby(['date', 'ticker'])
    n_groups = grouped.ngroups

    def build_tasks():
        for (date, ticker), group in grouped:
            yield (date, ticker, group['Moneyness'].to_numpy(), group['TTM'].to_numpy(),
                   group['impl_volatility'].to_numpy(), target_moneyness, target_ttm)

    n_workers = max(1, min(MAX_WORKERS, (os.cpu_count() or 1) - RESERVED_CORES))
    chunksize = max(1, n_groups // (n_workers * 4))
    print(f"Interpolating {n_groups} surfaces across {n_workers} worker processes "
          f"(chunksize={chunksize})...")

    tensor_list = []
    metadata = []
    mp_context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
        for metadata_entry, surface_array in executor.map(process_single_surface, build_tasks(), chunksize=chunksize):
            if metadata_entry['Fallback_Fraction'] >= MAX_FALLBACK_FRACTION:
                continue
            tensor_list.append(surface_array)
            metadata.append(metadata_entry)

    master_tensors = torch.FloatTensor(np.array(tensor_list))
    metadata_df = pd.DataFrame(metadata)
    print(f"Kept {len(metadata_df)} surfaces after quality filtering.")

    print("Scoring with the trained model...")
    model = VolatilityAutoencoder()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, weights_only=True))
    model.eval()

    scoring_criterion = nn.MSELoss(reduction='none')
    with torch.no_grad():
        reconstructed = model(master_tensors)
        mse_scores = scoring_criterion(reconstructed, master_tensors).mean(dim=1).numpy()

    metadata_df['Anomaly_Score'] = mse_scores

    # Replicate main.py's exact per-day top-decile flagging logic.
    print("Replicating per-day top-decile signal flagging...")
    is_signal = pd.Series(False, index=metadata_df.index)
    for date, group in metadata_df.groupby('Date'):
        threshold = group['Anomaly_Score'].quantile(0.90)
        is_signal.loc[group.index[group['Anomaly_Score'] >= threshold]] = True
    metadata_df['Is_Signal'] = is_signal

    sector_map = pd.read_csv(SECTOR_MAP_PATH)
    metadata_df = metadata_df.merge(sector_map, left_on='Ticker', right_on='ticker', how='left')
    metadata_df['sector'] = metadata_df['sector'].fillna('Unknown')

    overall_signal_rate = metadata_df['Is_Signal'].mean() * 100
    print(f"\nOverall signal rate (should be ~10% by construction): {overall_signal_rate:.2f}%")

    by_sector = metadata_df.groupby('sector').agg(
        n_candidates=('Is_Signal', 'size'),
        n_flagged=('Is_Signal', 'sum'),
        signal_rate_pct=('Is_Signal', lambda s: s.mean() * 100),
        avg_anomaly_score=('Anomaly_Score', 'mean'),
        avg_fallback_fraction=('Fallback_Fraction', 'mean'),
    ).sort_values('signal_rate_pct', ascending=False)
    by_sector['population_share_pct'] = by_sector['n_candidates'] / by_sector['n_candidates'].sum() * 100
    by_sector['signal_share_pct'] = by_sector['n_flagged'] / by_sector['n_flagged'].sum() * 100
    by_sector['over_representation'] = by_sector['signal_share_pct'] / by_sector['population_share_pct']

    print("\n=== Signal flagging by sector (over_representation > 1 means flagged more than its fair share) ===")
    print(by_sector.round(3))

    corr = by_sector['avg_fallback_fraction'].corr(by_sector['avg_anomaly_score'])
    print(f"\nCorrelation (sector avg fallback_fraction, sector avg anomaly score): {corr:.3f}")

    by_sector.to_csv('signal_quality_by_sector.csv')
    print("\nSaved signal_quality_by_sector.csv")

if __name__ == "__main__":
    main()

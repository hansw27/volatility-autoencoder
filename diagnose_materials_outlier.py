"""
Targeted follow-up to diagnose_signal_quality_by_sector.py: Materials'
average anomaly score (0.0285) was ~54x every other sector's median
(~0.00053) -- find out whether that's a handful of pathological
surfaces or genuinely sector-wide, and which ticker(s)/date(s) are
responsible.

Filters to just the Materials-sector tickers *before* the expensive
interpolation step, so this only redoes ~1/11th of the work a full
diagnose_signal_quality_by_sector.py run does, instead of a full re-run.

Usage: python diagnose_materials_outlier.py
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_pipeline import build_options_environment
from models import VolatilityAutoencoder
from main import process_single_surface, MAX_FALLBACK_FRACTION, RESERVED_CORES, MAX_WORKERS

OPTIONS_PATH = 'wrds_options_raw.csv'
STOCK_PATH = 'wrds_stock_raw.csv'
SECTOR_MAP_PATH = 'ticker_sector_map.csv'
CHECKPOINT_PATH = 'optimal_autoencoder.pth'
TARGET_SECTOR = 'Materials'

def main():
    sector_map = pd.read_csv(SECTOR_MAP_PATH)
    target_tickers = set(sector_map.loc[sector_map['sector'] == TARGET_SECTOR, 'ticker'])
    print(f"{TARGET_SECTOR} tickers: {sorted(target_tickers)}")

    print("Building options environment...")
    clean_options_df = build_options_environment(OPTIONS_PATH, STOCK_PATH)
    clean_options_df = clean_options_df[clean_options_df['ticker'].isin(target_tickers)]
    print(f"Filtered to {len(clean_options_df)} rows for {TARGET_SECTOR} tickers only.")

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
    metadata_df = metadata_df.sort_values('Anomaly_Score', ascending=False)

    print(f"\nAnomaly_Score distribution for {TARGET_SECTOR}:")
    print(metadata_df['Anomaly_Score'].describe())

    print(f"\n=== Worst 30 {TARGET_SECTOR} surfaces by Anomaly_Score ===")
    print(metadata_df[['Date', 'Ticker', 'Anomaly_Score', 'Fallback_Fraction']].head(30).to_string(index=False))

    print(f"\n=== Anomaly_Score by ticker within {TARGET_SECTOR} ===")
    by_ticker = metadata_df.groupby('Ticker').agg(
        n=('Anomaly_Score', 'size'),
        mean_score=('Anomaly_Score', 'mean'),
        median_score=('Anomaly_Score', 'median'),
        max_score=('Anomaly_Score', 'max'),
        mean_fallback=('Fallback_Fraction', 'mean'),
    ).sort_values('mean_score', ascending=False)
    print(by_ticker.round(6))

    metadata_df.to_csv('materials_outlier_surfaces.csv', index=False)
    print("\nSaved full sorted breakdown to materials_outlier_surfaces.csv")

if __name__ == "__main__":
    main()

"""
Standalone diagnostic: breaks down standardize_surface's fallback_fraction
(how much of a day's grid had no real cubic interpolation coverage, per
data_pipeline.interpolate_surface_arrays) by ticker and by GICS sector, to
see whether the surfaces MAX_FALLBACK_FRACTION skips concentrate in
specific names/sectors (e.g. thin-volume Utilities/Real Estate/Materials
names) or are spread evenly across the universe.

Reuses the real pipeline code (process_single_surface, build_options_environment,
and main.py's worker-count/quality-threshold constants) rather than
reimplementing any of the interpolation or filtering logic -- this is meant
to describe what the real run actually did, not a separate approximation
of it.

Usage: python diagnose_surface_quality.py
"""
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from data_pipeline import build_options_environment
from main import process_single_surface, MAX_FALLBACK_FRACTION, RESERVED_CORES, MAX_WORKERS

OPTIONS_PATH = 'wrds_options_raw.csv'
STOCK_PATH = 'wrds_stock_raw.csv'
SECTOR_MAP_PATH = 'ticker_sector_map.csv'

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
    print(f"Scoring quality for {n_groups} surfaces across {n_workers} worker processes "
          f"(chunksize={chunksize})...")

    records = []
    mp_context = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp_context) as executor:
        for metadata_entry, _ in executor.map(process_single_surface, build_tasks(), chunksize=chunksize):
            records.append(metadata_entry)

    quality_df = pd.DataFrame(records)
    quality_df['Skipped'] = quality_df['Fallback_Fraction'] >= MAX_FALLBACK_FRACTION

    sector_map = pd.read_csv(SECTOR_MAP_PATH)
    quality_df = quality_df.merge(sector_map, left_on='Ticker', right_on='ticker', how='left')
    quality_df['sector'] = quality_df['sector'].fillna('Unknown')

    n_skipped = int(quality_df['Skipped'].sum())
    print()
    print(f"Total surfaces: {len(quality_df)}   Skipped: {n_skipped} "
          f"({quality_df['Skipped'].mean() * 100:.2f}%)")

    def _breakdown(group_col):
        return quality_df.groupby(group_col).agg(
            n_days=('Skipped', 'size'),
            n_skipped=('Skipped', 'sum'),
            skip_rate_pct=('Skipped', lambda s: s.mean() * 100),
            avg_fallback_fraction=('Fallback_Fraction', 'mean'),
        ).sort_values('skip_rate_pct', ascending=False)

    by_ticker = _breakdown('Ticker')
    by_sector = _breakdown('sector')

    print()
    print("=== Skip rate by ticker (worst 20) ===")
    print(by_ticker.head(20).round(2))

    print()
    print("=== Skip rate by sector ===")
    print(by_sector.round(2))

    by_ticker.to_csv('surface_quality_by_ticker.csv')
    by_sector.to_csv('surface_quality_by_sector.csv')
    print()
    print("Saved surface_quality_by_ticker.csv and surface_quality_by_sector.csv")

if __name__ == "__main__":
    main()

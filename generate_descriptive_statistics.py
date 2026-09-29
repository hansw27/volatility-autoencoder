"""
Produces descriptive statistics (including percentiles needed to evaluate
a 10%-90% trim/winsorization) for the raw option-chain input variables
that feed surface interpolation: strike_price, Moneyness, TTM,
impl_volatility, delta, Premium, and underlying_price.

Reuses the real build_options_environment pipeline function rather than
reimplementing any of its filtering/scaling logic, so these statistics
describe the actual cleaned data the model trains on.

Usage: python generate_descriptive_statistics.py
Output: descriptive_statistics.csv
"""
import numpy as np
import pandas as pd

from data_pipeline import build_options_environment

OPTIONS_PATH = 'wrds_options_raw.csv'
STOCK_PATH = 'wrds_stock_raw.csv'
OUTPUT_PATH = 'descriptive_statistics.csv'

VARIABLES = ['strike_price', 'Moneyness', 'TTM', 'impl_volatility', 'delta', 'Premium', 'underlying_price']

PERCENTILES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]

def main():
    print("Building options environment...")
    df = build_options_environment(OPTIONS_PATH, STOCK_PATH)

    # underlying_price isn't in build_options_environment's final output
    # columns, but is recoverable exactly from Moneyness = strike_price /
    # underlying_price (no approximation, just algebra on columns already
    # returned) rather than re-deriving it from a separate merge.
    df['underlying_price'] = df['strike_price'] / df['Moneyness']

    print(f"Computing descriptive statistics over {len(df):,} rows...")
    rows = []
    for var in VARIABLES:
        s = df[var].astype('float64')
        stats = {
            'variable': var,
            'count': s.count(),
            'mean': s.mean(),
            'std': s.std(),
            'min': s.min(),
            'max': s.max(),
            'skew': s.skew(),
            'kurtosis': s.kurtosis(),
        }
        quantiles = s.quantile(PERCENTILES)
        for p in PERCENTILES:
            stats[f'p{int(p * 100)}'] = quantiles.loc[p]
        rows.append(stats)

    stats_df = pd.DataFrame(rows).set_index('variable')

    column_order = ['count', 'mean', 'std', 'min'] + \
                    [f'p{int(p * 100)}' for p in PERCENTILES] + \
                    ['max', 'skew', 'kurtosis']
    stats_df = stats_df[column_order]

    pd.set_option('display.width', 160)
    pd.set_option('display.max_columns', None)
    print()
    print(stats_df.round(4))

    stats_df.to_csv(OUTPUT_PATH)
    print(f"\nSaved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()

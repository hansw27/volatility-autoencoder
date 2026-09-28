import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import the functions from your data pipeline module
from data_pipeline import build_options_environment, standardize_surface

def reshape_surface_grid(surface_array, target_moneyness, target_ttm):
    # standardize_surface flattens a grid built via
    # np.meshgrid(target_moneyness, target_ttm), which has shape
    # (len(target_ttm), len(target_moneyness)). Reshape to that layout
    # first, then transpose to (Moneyness, TTM) so it lines up with
    # X, Y = np.meshgrid(target_ttm, target_moneyness) in the caller.
    return surface_array.reshape(len(target_ttm), len(target_moneyness)).T

def build_surface_figure(target_moneyness, target_ttm, surface_array, ticker, date):
    Z = reshape_surface_grid(surface_array, target_moneyness, target_ttm)
    X, Y = np.meshgrid(target_ttm, target_moneyness)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', linewidth=0.1)

    ax.set_title(f'Volatility Surface: {ticker} on {date}')
    ax.set_xlabel('Time to Maturity (Years)')
    ax.set_ylabel('Moneyness (Strike / Underlying Price)')
    ax.set_zlabel('Implied Volatility')

    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Implied Vol')
    return fig

def plot_volatility_surface(target_moneyness, target_ttm, surface_array, ticker, date):
    build_surface_figure(target_moneyness, target_ttm, surface_array, ticker, date)
    plt.show()

def save_volatility_surface(target_moneyness, target_ttm, surface_array, ticker, date, out_path):
    fig = build_surface_figure(target_moneyness, target_ttm, surface_array, ticker, date)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)

def save_monthly_surfaces(df, target_moneyness, target_ttm, out_dir='surfaces'):
    """
    For every ticker, picks the first available trading day in each
    calendar month and saves its standardized volatility surface as a PNG
    under out_dir/<ticker>/<ticker>_<YYYY-MM>.png. Returns the number of
    surfaces saved.
    """
    os.makedirs(out_dir, exist_ok=True)

    df = df.copy()
    df['year_month'] = df['date'].dt.to_period('M')

    saved = 0
    skipped = 0
    for (ticker, year_month), _ in df.groupby(['ticker', 'year_month']):
        month_slice = df[(df['ticker'] == ticker) & (df['year_month'] == year_month)]
        sample_date = month_slice['date'].min()
        daily_chain = month_slice[month_slice['date'] == sample_date]

        try:
            surface = standardize_surface(daily_chain, target_moneyness, target_ttm)
        except ValueError:
            # e.g. an empty chain slipped through -- skip rather than crash
            # the whole run over one bad month.
            skipped += 1
            continue

        ticker_dir = os.path.join(out_dir, str(ticker))
        os.makedirs(ticker_dir, exist_ok=True)
        out_path = os.path.join(ticker_dir, f'{ticker}_{year_month}.png')
        save_volatility_surface(target_moneyness, target_ttm, surface, ticker, sample_date.date(), out_path)
        saved += 1

    if skipped:
        print(f"Skipped {skipped} ticker-months with no usable quotes.")
    return saved

if __name__ == "__main__":
    print("Loading WRDS datasets...")
    # Ensure your downloaded files are named exactly this and sit in the same folder
    df = build_options_environment('wrds_options_raw.csv', 'wrds_stock_raw.csv')

    # Define the 10x5 grid parameters
    target_moneyness = np.linspace(0.8, 1.2, 10)
    target_ttm = np.array([30, 60, 90, 120, 180]) / 365.0

    print("Saving one surface per ticker per month to ./surfaces ...")
    n_saved = save_monthly_surfaces(df, target_moneyness, target_ttm, out_dir='surfaces')
    print(f"Saved {n_saved} surface plots to ./surfaces")

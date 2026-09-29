import pandas as pd
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import QhullError

# Only the columns build_options_environment actually reads. Real WRDS
# OptionMetrics exports carry several columns this pipeline never touches
# (optionid, issuer, index_flag, exercise_style, secid); loading them just
# to discard them wastes memory, especially 'issuer' (free-text company
# name, an object-dtype column -- among the most expensive per row).
OPTIONS_USECOLS = ['date', 'exdate', 'ticker', 'secid', 'cp_flag', 'strike_price',
                    'volume', 'open_interest', 'impl_volatility', 'delta',
                    'best_bid', 'best_offer']
OPTIONS_DTYPES = {
    'ticker': 'category', 'cp_flag': 'category', 'secid': 'int32',
    'strike_price': 'float32', 'volume': 'int32', 'open_interest': 'int32',
    'impl_volatility': 'float32', 'delta': 'float32',
    'best_bid': 'float32', 'best_offer': 'float32',
}
STOCK_USECOLS = ['date', 'ticker', 'secid', 'close']
STOCK_DTYPES = {'ticker': 'category', 'secid': 'int32', 'close': 'float32'}

def build_options_environment(filepath, underlying_filepath):
    options_df = pd.read_csv(filepath, usecols=OPTIONS_USECOLS, dtype=OPTIONS_DTYPES)
    stock_df = pd.read_csv(underlying_filepath, usecols=STOCK_USECOLS, dtype=STOCK_DTYPES)

    options_df['date'] = pd.to_datetime(options_df['date'])
    options_df['exdate'] = pd.to_datetime(options_df['exdate'])
    stock_df['date'] = pd.to_datetime(stock_df['date'])

    # CRSP/WRDS convention: a negative close price means no trade actually
    # executed that day, and the stored value is the bid-ask midpoint
    # instead, flagged with a negative sign to mark it as an estimate
    # rather than a traded price. The magnitude is still a legitimate price
    # -- take the absolute value rather than dropping or zeroing these rows
    # (zeroing would divide-by-zero into Moneyness below).
    stock_df['close'] = stock_df['close'].abs()

    # Join on (date, secid) rather than (date, ticker). Ticker symbols get
    # reused/shared across unrelated companies over a multi-decade dataset
    # (e.g. two different securities both traded as 'LIN' on overlapping
    # dates in the mid-2000s, one of them unrelated to the modern Linde
    # plc) -- joining on ticker alone fans a single option row out against
    # every same-ticker stock row that day, pairing real option contracts
    # with a different company's stock price. secid is the actual unique
    # security identifier and is populated in both files.
    df = pd.merge(options_df, stock_df[['date', 'secid', 'close']],
                  on=['date', 'secid'], how='left')
    df.rename(columns={'close': 'underlying_price'}, inplace=True)
    df = df[df['underlying_price'].notna()]

    df = df[(df['volume'] > 0) & (df['open_interest'] > 0)]
    df = df[df['impl_volatility'].notna()]
    df = df[df['best_bid'] > 0]
    df = df[df['best_bid'] < df['best_offer']]

    df['TTM'] = ((df['exdate'] - df['date']).dt.days / 365.0).astype('float32')
    df = df[df['TTM'] > 0]

    df['strike_price'] = df['strike_price'] / 1000.0
    df['Moneyness'] = df['strike_price'] / df['underlying_price']
    df['Premium'] = (df['best_bid'] + df['best_offer']) / 2.0

    final_columns = ['date', 'ticker', 'exdate', 'cp_flag', 'strike_price',
                     'Moneyness', 'TTM', 'impl_volatility', 'delta', 'Premium']
    return df[final_columns]

def interpolate_surface_arrays(known_points, known_vols, target_moneyness, target_ttm, return_quality=False):
    """
    Pure-numpy core of standardize_surface: takes raw (Moneyness, TTM)
    points and implied vols directly rather than a DataFrame. Kept
    separate so callers that need to ship work to another process (e.g.
    a multiprocessing pool) can pass small numpy arrays instead of
    pickling pandas DataFrame slices.
    """
    grid_x, grid_y = np.meshgrid(target_moneyness, target_ttm)

    if len(known_points) == 0:
        raise ValueError("Cannot standardize an empty options chain: no (Moneyness, TTM) quotes available.")

    nearest_surface = griddata(points=known_points, values=known_vols, xi=(grid_x, grid_y), method='nearest')

    try:
        cubic_surface = griddata(points=known_points, values=known_vols, xi=(grid_x, grid_y), method='cubic')
    except QhullError:
        # Too few (or degenerate/collinear) points to build a cubic
        # interpolant, e.g. thin/illiquid chains. Fall back to flat
        # nearest-neighbor extrapolation across the whole grid instead
        # of crashing the pipeline.
        cubic_surface = np.full_like(nearest_surface, np.nan)

    fallback_mask = np.isnan(cubic_surface)
    final_surface = np.where(fallback_mask, nearest_surface, cubic_surface)

    if return_quality:
        # Fraction of the grid that had no real cubic coverage and was
        # extrapolated via flat nearest-neighbor instead -- a free-to-compute
        # proxy for how much of this surface is genuinely interpolated from
        # nearby quotes vs. guessed from the nearest one. 0.0 = fully
        # interpolated, 1.0 = the whole grid is a flat nearest-neighbor guess.
        fallback_fraction = float(np.mean(fallback_mask))
        return final_surface.flatten(), fallback_fraction

    return final_surface.flatten()

def standardize_surface(df_daily, target_moneyness, target_ttm, return_quality=False):
    known_points = df_daily[['Moneyness', 'TTM']].values
    known_vols = df_daily['impl_volatility'].values
    return interpolate_surface_arrays(known_points, known_vols, target_moneyness, target_ttm, return_quality=return_quality)

def atm_flat_index(target_moneyness, target_ttm):
    """
    Index into a standardize_surface() flattened array for the grid point
    nearest ATM (moneyness closest to 1.0) at the shortest available
    time-to-maturity. Used to read a single representative implied-vol
    point off a standardized surface, e.g. to compare actual vs.
    autoencoder-reconstructed vol for signal direction.
    """
    target_moneyness = np.asarray(target_moneyness)
    target_ttm = np.asarray(target_ttm)
    moneyness_idx = np.argmin(np.abs(target_moneyness - 1.0))
    ttm_idx = np.argmin(target_ttm)
    return ttm_idx * len(target_moneyness) + moneyness_idx
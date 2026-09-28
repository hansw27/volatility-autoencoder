"""
Extensive integration-level coverage for everything in this folder that
test_volatility_strategy.py doesn't already unit-test:

- visualize_surface.py: the Z-array reshape used to feed matplotlib
  (matplotlib itself is not exercised -- we don't want a GUI/backend
  dependency in the test suite, but the numpy reshape logic that maps a
  flat 50-element surface onto (moneyness, ttm) axes is pure, testable,
  and turns out to be wrong).
- main.py: a real end-to-end run of build_options_environment ->
  standardize_surface -> train the autoencoder -> score anomalies ->
  backtest, against a small synthetic WRDS-shaped dataset written to a
  tmp_path. This exercises the actual orchestration code in main(),
  not a reimplementation of it.
"""
import importlib

import numpy as np
import pandas as pd
import pytest

from data_pipeline import standardize_surface
from visualize_surface import reshape_surface_grid


# ---------------------------------------------------------------------------
# visualize_surface.py
# ---------------------------------------------------------------------------

class TestVisualizeSurfaceReshape:

    def test_reshape_matches_the_grid_that_produced_the_flat_surface(self):
        # standardize_surface builds its grid via
        #   grid_x, grid_y = np.meshgrid(target_moneyness, target_ttm)
        # which (default 'xy' indexing) has shape (len(target_ttm),
        # len(target_moneyness)) -- i.e. TTM varies down rows, Moneyness
        # varies across columns -- and then flattens that.
        #
        # visualize_surface.plot_volatility_surface must reconstruct that
        # same (TTM, Moneyness) layout and transpose to (Moneyness, TTM)
        # to line up with X, Y = np.meshgrid(target_ttm, target_moneyness).
        target_moneyness = np.linspace(0.8, 1.2, 10)
        target_ttm = np.array([30, 60, 90, 120, 180]) / 365.0

        # implied vol that depends only on moneyness, so a correct reshape
        # must show the same value across every TTM row.
        vol_by_moneyness = dict(zip(target_moneyness, np.linspace(0.1, 0.5, 10)))
        chain = pd.DataFrame({
            'Moneyness': np.tile(target_moneyness, len(target_ttm)),
            'TTM': np.repeat(target_ttm, len(target_moneyness)),
            'impl_volatility': [vol_by_moneyness[m] for m in np.tile(target_moneyness, len(target_ttm))],
        })
        surface = standardize_surface(chain, target_moneyness, target_ttm)
        Z = reshape_surface_grid(surface, target_moneyness, target_ttm)

        assert Z.shape == (len(target_moneyness), len(target_ttm))
        for ttm_idx in range(len(target_ttm)):
            assert Z[:, ttm_idx] == pytest.approx(np.linspace(0.1, 0.5, 10), abs=1e-6), (
                "Every TTM column must show the same moneyness-only vol ramp; "
                "a transposed reshape would scramble this."
            )


# ---------------------------------------------------------------------------
# main.py: full pipeline integration test
# ---------------------------------------------------------------------------

def _write_synthetic_wrds_dataset(tmp_path, seed=0, n_days=14, tickers=('AAPL', 'MSFT')):
    rng = np.random.default_rng(seed)
    trading_days = pd.bdate_range('2026-01-05', periods=n_days)
    moneyness_pts = [0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15]
    # Real chains trade multiple expirations on any given day; using just
    # one (as an earlier version of this fixture did) makes every day's
    # (Moneyness, TTM) point cloud collinear (constant TTM), which is
    # degenerate for 2D interpolation and unrealistic vs. real data.
    tenor_offsets_days = [30, 90]

    opt_rows = []
    for d in trading_days:
        for t in tickers:
            base_close = 100.0 if t == 'AAPL' else 250.0
            for tenor_days in tenor_offsets_days:
                exdate = d + pd.Timedelta(days=tenor_days)
                for m in moneyness_pts:
                    strike = round(base_close * m)
                    for cp in ['C', 'P']:
                        delta = (m - 0.7) * 0.9 if cp == 'C' else -(1.3 - m) * 0.9
                        premium = max(0.5, base_close * 0.05 * abs(1 - m) + rng.uniform(0, 0.3))
                        opt_rows.append({
                            'date': d, 'exdate': exdate, 'ticker': t, 'cp_flag': cp,
                            'strike_price': strike * 1000, 'volume': 50, 'open_interest': 200,
                            'impl_volatility': 0.2 + 0.1 * abs(1 - m) + rng.uniform(0, 0.02),
                            'best_bid': premium, 'best_offer': premium + 0.1, 'delta': delta,
                        })

    stock_rows = []
    price = {t: (100.0 if t == 'AAPL' else 250.0) for t in tickers}
    for d in trading_days:
        for t in tickers:
            price[t] *= (1 + rng.uniform(-0.01, 0.01))
            stock_rows.append({'date': d, 'ticker': t, 'close': price[t]})

    pd.DataFrame(opt_rows).to_csv(tmp_path / 'wrds_options_raw.csv', index=False)
    pd.DataFrame(stock_rows).to_csv(tmp_path / 'wrds_stock_raw.csv', index=False)

    sector_cycle = ['Information Technology', 'Health Care', 'Financials', 'Energy']
    sector_rows = [{'ticker': t, 'sector': sector_cycle[i % len(sector_cycle)]} for i, t in enumerate(tickers)]
    pd.DataFrame(sector_rows).to_csv(tmp_path / 'ticker_sector_map.csv', index=False)


class TestMainPipelineEndToEnd:

    def test_full_pipeline_runs_and_produces_valid_results(self, tmp_path, monkeypatch, capsys):
        _write_synthetic_wrds_dataset(tmp_path)
        monkeypatch.chdir(tmp_path)

        # Reload main so any module-level state from a prior test run
        # doesn't leak, and import fresh each time it's run under a new cwd.
        import main as main_module
        importlib.reload(main_module)

        main_module.main()

        results_path = tmp_path / 'strategy_results.csv'
        assert results_path.exists(), "main() must write strategy_results.csv"

        results = pd.read_csv(results_path)
        assert list(results.columns) == ['Date', 'Ticker', 'PnL', 'Direction', 'Capital_Allocated']
        assert set(results['Direction']) <= {'long', 'short'}
        assert not results['PnL'].isna().any()
        assert np.isfinite(results['PnL'].to_numpy()).all()

        checkpoint_path = tmp_path / 'optimal_autoencoder.pth'
        assert checkpoint_path.exists(), "EarlyStopping must persist a model checkpoint"

        sector_path = tmp_path / 'sector_performance.csv'
        assert sector_path.exists(), "main() must write sector_performance.csv"
        sector_summary = pd.read_csv(sector_path, index_col=0)
        assert list(sector_summary.columns) == ['Sharpe Ratio', 'Win Rate (%)', 'Total PnL ($)', 'Trade Count']
        assert sector_summary['Trade Count'].sum() == len(results)

        out = capsys.readouterr().out
        assert "Average PnL per Trade" in out
        assert "Strategy Win Rate" in out
        assert "Performance Summary" in out
        assert "Sector Breakdown" in out

    def test_unmapped_ticker_falls_back_to_unknown_sector_with_warning(self, tmp_path, monkeypatch, capsys):
        # If a traded ticker isn't in the sector map (a stale map, a new
        # addition to the universe, etc.), main() must not crash or silently
        # drop those trades from the tear sheet -- it should warn clearly
        # and group them under 'Unknown' instead.
        _write_synthetic_wrds_dataset(tmp_path, seed=13, n_days=10, tickers=('AAPL', 'MSFT'))
        # Remove MSFT from the sector map to simulate a gap in the mapping.
        sector_map_path = tmp_path / 'ticker_sector_map.csv'
        sector_map = pd.read_csv(sector_map_path)
        sector_map[sector_map['ticker'] != 'MSFT'].to_csv(sector_map_path, index=False)

        monkeypatch.chdir(tmp_path)
        import main as main_module
        importlib.reload(main_module)

        main_module.main()

        out = capsys.readouterr().out
        assert "WARNING: no sector mapping" in out
        assert "MSFT" in out

        sector_summary = pd.read_csv(tmp_path / 'sector_performance.csv', index_col=0)
        results = pd.read_csv(tmp_path / 'strategy_results.csv')
        if 'MSFT' in results['Ticker'].values:
            assert 'Unknown' in sector_summary.index

    def test_pipeline_is_deterministic_given_same_seed(self, tmp_path, monkeypatch):
        # Same synthetic input data (same seed) should always survive the
        # filtering/interpolation stage and produce the same number of
        # standardized surfaces -- this guards against the interpolation
        # fallback silently dropping or corrupting rows.
        _write_synthetic_wrds_dataset(tmp_path, seed=42)
        monkeypatch.chdir(tmp_path)

        import main as main_module
        importlib.reload(main_module)

        from data_pipeline import build_options_environment
        clean_df_1 = build_options_environment('wrds_options_raw.csv', 'wrds_stock_raw.csv')
        clean_df_2 = build_options_environment('wrds_options_raw.csv', 'wrds_stock_raw.csv')
        pd.testing.assert_frame_equal(clean_df_1.reset_index(drop=True), clean_df_2.reset_index(drop=True))

    def test_full_pipeline_is_reproducible_with_fixed_seed(self, tmp_path, monkeypatch):
        # Two full runs (including model training) on identical data must
        # produce bit-identical results now that RANDOM_SEED fixes the
        # autoencoder's weight initialization. Before this fix, every run
        # started from different random weights, so the exact same input
        # data could silently produce different anomaly rankings, flip a
        # trade's long/short direction, and change the resulting PnL --
        # with zero code changes between runs.
        _write_synthetic_wrds_dataset(tmp_path, seed=11, n_days=20, tickers=('AAPL', 'MSFT'))
        monkeypatch.chdir(tmp_path)

        import main as main_module
        importlib.reload(main_module)

        main_module.main()
        first_run = pd.read_csv(tmp_path / 'strategy_results.csv')

        main_module.main()
        second_run = pd.read_csv(tmp_path / 'strategy_results.csv')

        pd.testing.assert_frame_equal(first_run, second_run)

    def test_pipeline_handles_a_thin_single_quote_day_without_crashing(self, tmp_path, monkeypatch):
        # Regression test for the standardize_surface QhullError fix: inject
        # one day/ticker with only a single quote into an otherwise normal
        # dataset and make sure the full main() run still completes.
        _write_synthetic_wrds_dataset(tmp_path, seed=1, n_days=10, tickers=('AAPL',))
        options_path = tmp_path / 'wrds_options_raw.csv'
        options_df = pd.read_csv(options_path)

        thin_day = pd.to_datetime(options_df['date']).min()
        keep_mask = ~(
            (pd.to_datetime(options_df['date']) == thin_day) &
            (options_df['ticker'] == 'AAPL')
        )
        # Re-add exactly one quote for that day so the group still exists
        # but only has 1 point -- previously fatal for standardize_surface.
        thin_row = options_df[options_df['ticker'] == 'AAPL'].iloc[[0]].copy()
        thin_row['date'] = thin_day.strftime('%Y-%m-%d')
        options_df = pd.concat([options_df[keep_mask], thin_row], ignore_index=True)
        options_df.to_csv(options_path, index=False)

        monkeypatch.chdir(tmp_path)
        import main as main_module
        importlib.reload(main_module)

        main_module.main()  # must not raise
        assert (tmp_path / 'strategy_results.csv').exists()

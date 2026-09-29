import numpy as np
import pandas as pd
import pytest
import torch
from scipy.interpolate import griddata

from data_pipeline import build_options_environment, standardize_surface, atm_flat_index
from models import VolatilityAutoencoder, EarlyStopping
from backtester import get_trading_day, get_atm_straddle, calculate_5_day_straddle_pnl, calculate_performance_metrics


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _write_raw_csvs(tmp_path, options_rows, stock_rows):
    options_path = tmp_path / "options.csv"
    stock_path = tmp_path / "stock.csv"
    pd.DataFrame(options_rows).to_csv(options_path, index=False)
    pd.DataFrame(stock_rows).to_csv(stock_path, index=False)
    return str(options_path), str(stock_path)


def _base_option_row(**overrides):
    row = {
        'date': '2026-01-05',
        'exdate': '2026-02-04',
        'ticker': 'AAPL',
        'secid': 1001,
        'cp_flag': 'C',
        'strike_price': 100000,
        'volume': 100,
        'open_interest': 500,
        'impl_volatility': 0.25,
        'best_bid': 2.50,
        'best_offer': 2.70,
        'delta': 0.52,
    }
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Data pipeline: ingestion / filtering / scaling
# ---------------------------------------------------------------------------

class TestBuildOptionsEnvironment:

    def test_valid_row_survives_and_is_scaled_correctly(self, tmp_path):
        options_rows = [_base_option_row()]
        stock_rows = [{'date': '2026-01-05', 'ticker': 'AAPL', 'secid': 1001, 'close': 100.0}]
        opt_path, stock_path = _write_raw_csvs(tmp_path, options_rows, stock_rows)

        df = build_options_environment(opt_path, stock_path)

        assert len(df) == 1
        assert df['strike_price'].iloc[0] == pytest.approx(100.0), "strike_price must be divided by 1000"
        assert df['Moneyness'].iloc[0] == pytest.approx(1.0)
        assert df['Premium'].iloc[0] == pytest.approx(2.60)
        assert df['TTM'].iloc[0] == pytest.approx(30 / 365.0, abs=1e-4)

    def test_negative_close_price_is_taken_as_absolute_value(self, tmp_path):
        # CRSP/WRDS convention: a negative close means no trade executed
        # that day and the stored value is the bid-ask midpoint instead,
        # sign-flipped to flag it as an estimate. The magnitude is still a
        # real price -- Moneyness must be computed from abs(close), not
        # left negative (which would also make Moneyness itself negative
        # and nonsensical) or zeroed (which would divide-by-zero).
        options_rows = [_base_option_row()]
        stock_rows = [{'date': '2026-01-05', 'ticker': 'AAPL', 'secid': 1001, 'close': -100.0}]
        opt_path, stock_path = _write_raw_csvs(tmp_path, options_rows, stock_rows)

        df = build_options_environment(opt_path, stock_path)

        assert len(df) == 1
        assert df['Moneyness'].iloc[0] == pytest.approx(1.0), "Moneyness must use abs(close), not the raw negative price"

    def test_zero_volume_contract_filtered_out(self, tmp_path):
        options_rows = [_base_option_row(volume=0)]
        stock_rows = [{'date': '2026-01-05', 'ticker': 'AAPL', 'secid': 1001, 'close': 100.0}]
        opt_path, stock_path = _write_raw_csvs(tmp_path, options_rows, stock_rows)

        df = build_options_environment(opt_path, stock_path)
        assert df.empty, "Zero-volume contracts must be filtered"

    def test_zero_open_interest_filtered_out(self, tmp_path):
        options_rows = [_base_option_row(open_interest=0)]
        stock_rows = [{'date': '2026-01-05', 'ticker': 'AAPL', 'secid': 1001, 'close': 100.0}]
        opt_path, stock_path = _write_raw_csvs(tmp_path, options_rows, stock_rows)

        df = build_options_environment(opt_path, stock_path)
        assert df.empty, "Zero open-interest contracts must be filtered"

    def test_crossed_market_filtered_out(self, tmp_path):
        # best_bid > best_offer: a crossed/broken quote
        options_rows = [_base_option_row(best_bid=3.00, best_offer=2.70)]
        stock_rows = [{'date': '2026-01-05', 'ticker': 'AAPL', 'secid': 1001, 'close': 100.0}]
        opt_path, stock_path = _write_raw_csvs(tmp_path, options_rows, stock_rows)

        df = build_options_environment(opt_path, stock_path)
        assert df.empty, "Crossed markets (bid > ask) must be filtered"

    def test_locked_market_at_equal_bid_ask_is_filtered(self, tmp_path):
        # best_bid == best_offer is NOT strictly "crossed" but the current
        # filter uses a strict '<' so it is dropped too. Documents actual
        # behavior rather than assuming it.
        options_rows = [_base_option_row(best_bid=2.70, best_offer=2.70)]
        stock_rows = [{'date': '2026-01-05', 'ticker': 'AAPL', 'secid': 1001, 'close': 100.0}]
        opt_path, stock_path = _write_raw_csvs(tmp_path, options_rows, stock_rows)

        df = build_options_environment(opt_path, stock_path)
        assert df.empty

    def test_zero_bid_filtered_out(self, tmp_path):
        options_rows = [_base_option_row(best_bid=0.0)]
        stock_rows = [{'date': '2026-01-05', 'ticker': 'AAPL', 'secid': 1001, 'close': 100.0}]
        opt_path, stock_path = _write_raw_csvs(tmp_path, options_rows, stock_rows)

        df = build_options_environment(opt_path, stock_path)
        assert df.empty, "Zero-bid quotes must be filtered"

    def test_zero_dte_expiration_filtered_out(self, tmp_path):
        # exdate == date -> TTM == 0, must be dropped (TTM > 0 strict)
        options_rows = [_base_option_row(exdate='2026-01-05')]
        stock_rows = [{'date': '2026-01-05', 'ticker': 'AAPL', 'secid': 1001, 'close': 100.0}]
        opt_path, stock_path = _write_raw_csvs(tmp_path, options_rows, stock_rows)

        df = build_options_environment(opt_path, stock_path)
        assert df.empty, "Zero-DTE (TTM<=0) contracts must be filtered"

    def test_missing_implied_vol_filtered_out(self, tmp_path):
        options_rows = [_base_option_row(impl_volatility=np.nan)]
        stock_rows = [{'date': '2026-01-05', 'ticker': 'AAPL', 'secid': 1001, 'close': 100.0}]
        opt_path, stock_path = _write_raw_csvs(tmp_path, options_rows, stock_rows)

        df = build_options_environment(opt_path, stock_path)
        assert df.empty, "Rows with NaN implied vol must be filtered"

    def test_no_matching_underlying_price_is_dropped(self, tmp_path):
        # left-merge: a date/ticker with no matching stock row used to
        # produce NaN underlying_price and silently propagate into
        # Moneyness (NaN). Fixed: such rows are now filtered out.
        options_rows = [_base_option_row()]
        stock_rows = [{'date': '2026-01-06', 'ticker': 'AAPL', 'secid': 1001, 'close': 100.0}]  # wrong date
        opt_path, stock_path = _write_raw_csvs(tmp_path, options_rows, stock_rows)

        df = build_options_environment(opt_path, stock_path)
        assert df.empty, "Rows with no matching underlying price must be dropped, not kept with NaN Moneyness"

    def test_mixed_good_and_bad_rows(self, tmp_path):
        options_rows = [
            _base_option_row(cp_flag='C', volume=100),
            _base_option_row(cp_flag='P', volume=0),          # filtered: zero volume
            _base_option_row(cp_flag='C', strike_price=105000, best_bid=5.0, best_offer=1.0),  # crossed
        ]
        stock_rows = [{'date': '2026-01-05', 'ticker': 'AAPL', 'secid': 1001, 'close': 100.0}]
        opt_path, stock_path = _write_raw_csvs(tmp_path, options_rows, stock_rows)

        df = build_options_environment(opt_path, stock_path)
        assert len(df) == 1
        assert df['cp_flag'].iloc[0] == 'C'
        assert df['strike_price'].iloc[0] == pytest.approx(100.0)

    def test_same_day_ticker_collision_between_two_secids_does_not_cross_contaminate(self, tmp_path):
        # BUG (found via a real WRDS pull): ticker symbols get reused across
        # unrelated companies over a multi-decade dataset. Two different
        # secids traded as the same ticker on the same date (e.g. an
        # unrelated 1990s-2000s company and the modern post-merger Linde
        # both as 'LIN'). Joining on ('date', 'ticker') alone fans each
        # option row out against every same-ticker stock row that day,
        # pairing real contracts with a different company's stock price and
        # producing nonsensical Moneyness. Joining on ('date', 'secid')
        # instead must keep each option priced against its own company.
        options_rows = [
            _base_option_row(secid=1001, strike_price=25000),   # belongs to the $25 company
            _base_option_row(secid=2002, strike_price=40000),   # belongs to the $40 company
        ]
        stock_rows = [
            {'date': '2026-01-05', 'ticker': 'LIN', 'secid': 1001, 'close': 25.0},
            {'date': '2026-01-05', 'ticker': 'LIN', 'secid': 2002, 'close': 40.0},
        ]
        for row in options_rows:
            row['ticker'] = 'LIN'
        opt_path, stock_path = _write_raw_csvs(tmp_path, options_rows, stock_rows)

        df = build_options_environment(opt_path, stock_path)

        assert len(df) == 2, "Each option row must match exactly one stock row, not fan out against both"
        moneyness_by_strike = dict(zip(df['strike_price'], df['Moneyness']))
        assert moneyness_by_strike[25.0] == pytest.approx(1.0), "the $25 strike must be priced against the $25 company"
        assert moneyness_by_strike[40.0] == pytest.approx(1.0), "the $40 strike must be priced against the $40 company"


# ---------------------------------------------------------------------------
# Surface standardization / interpolation
# ---------------------------------------------------------------------------

class TestAtmFlatIndex:

    def test_matches_manual_flatten_of_meshgrid(self):
        target_moneyness = np.linspace(0.8, 1.2, 10)
        target_ttm = np.array([30, 60, 90, 120, 180]) / 365.0

        idx = atm_flat_index(target_moneyness, target_ttm)

        grid_x, grid_y = np.meshgrid(target_moneyness, target_ttm)
        flat_moneyness = grid_x.flatten()
        flat_ttm = grid_y.flatten()

        assert flat_ttm[idx] == target_ttm.min()
        assert abs(flat_moneyness[idx] - 1.0) == pytest.approx(np.min(np.abs(target_moneyness - 1.0)))

    def test_picks_lowest_ttm_when_ttm_not_sorted(self):
        target_moneyness = np.array([0.9, 1.0, 1.1])
        target_ttm = np.array([90, 30, 60]) / 365.0  # 30 is the shortest, at index 1
        idx = atm_flat_index(target_moneyness, target_ttm)
        # flat_idx = ttm_idx * len(moneyness) + moneyness_idx = 1*3 + 1 = 4
        assert idx == 4


class TestStandardizeSurface:

    TARGET_MONEYNESS = np.linspace(0.8, 1.2, 10)
    TARGET_TTM = np.array([30, 60, 90, 120, 180]) / 365.0

    def test_output_is_flat_50_element_array(self):
        np.random.seed(1)
        chain = pd.DataFrame({
            'Moneyness': np.random.uniform(0.7, 1.3, 60),
            'TTM': np.random.uniform(10 / 365.0, 200 / 365.0, 60),
            'impl_volatility': np.random.uniform(0.1, 0.5, 60),
        })
        surface = standardize_surface(chain, self.TARGET_MONEYNESS, self.TARGET_TTM)
        assert surface.shape == (50,)

    def test_no_nans_with_dense_well_spread_data(self):
        np.random.seed(7)
        chain = pd.DataFrame({
            'Moneyness': np.random.uniform(0.7, 1.3, 80),
            'TTM': np.random.uniform(10 / 365.0, 200 / 365.0, 80),
            'impl_volatility': np.random.uniform(0.1, 0.5, 80),
        })
        surface = standardize_surface(chain, self.TARGET_MONEYNESS, self.TARGET_TTM)
        assert not np.isnan(surface).any()

    def test_extrapolated_tails_use_nearest_neighbor_flat_value(self):
        # Known data occupies only the interior of the target grid, so grid
        # corners fall outside the convex hull and must come from the
        # 'nearest' fallback rather than a (possibly wild) cubic extrapolation.
        chain = pd.DataFrame({
            'Moneyness': [0.95, 1.00, 1.05, 0.95, 1.00, 1.05, 0.95, 1.00, 1.05],
            'TTM': np.repeat([40 / 365.0, 90 / 365.0, 150 / 365.0], 3),
            'impl_volatility': [0.20, 0.21, 0.22, 0.25, 0.26, 0.27, 0.30, 0.31, 0.32],
        })
        surface = standardize_surface(chain, self.TARGET_MONEYNESS, self.TARGET_TTM)

        known_points = chain[['Moneyness', 'TTM']].values
        known_vols = chain['impl_volatility'].values
        grid_x, grid_y = np.meshgrid(self.TARGET_MONEYNESS, self.TARGET_TTM)
        expected_nearest = griddata(known_points, known_vols, (grid_x, grid_y), method='nearest').flatten()

        # Corner of the grid (min moneyness, min TTM) is far outside the hull.
        corner_idx = 0
        assert surface[corner_idx] == pytest.approx(expected_nearest[corner_idx]), (
            "Deep OTM / far-tenor grid points should be flat-extrapolated via "
            "nearest-neighbor, not left to cubic overshoot."
        )
        assert not np.isnan(surface).any()

    def test_sparse_chain_falls_back_to_nearest_neighbor(self):
        # Illiquid tickers/days with fewer than 4 (non-collinear) quotes are
        # common in real OptionMetrics data. Cubic interpolation can't run
        # (Qhull needs >= 4 points), so the whole grid must fall back to
        # flat nearest-neighbor values instead of crashing.
        chain = pd.DataFrame({
            'Moneyness': [0.95, 1.05],
            'TTM': [30 / 365.0, 60 / 365.0],
            'impl_volatility': [0.20, 0.22],
        })
        surface = standardize_surface(chain, self.TARGET_MONEYNESS, self.TARGET_TTM)
        assert surface.shape == (50,)
        assert not np.isnan(surface).any()
        assert set(np.unique(surface)) <= {0.20, 0.22}

    def test_single_quote_chain_falls_back_to_nearest_neighbor(self):
        chain = pd.DataFrame({
            'Moneyness': [1.0],
            'TTM': [30 / 365.0],
            'impl_volatility': [0.20],
        })
        surface = standardize_surface(chain, self.TARGET_MONEYNESS, self.TARGET_TTM)
        assert surface.shape == (50,)
        assert not np.isnan(surface).any()
        assert np.all(surface == 0.20), "With a single known quote, every grid point must equal it"

    def test_empty_chain_raises_clear_error(self):
        chain = pd.DataFrame({'Moneyness': [], 'TTM': [], 'impl_volatility': []})
        with pytest.raises(ValueError, match="empty options chain"):
            standardize_surface(chain, self.TARGET_MONEYNESS, self.TARGET_TTM)

    def test_collinear_points_fall_back_to_nearest_neighbor(self):
        # All quotes share a single moneyness (e.g. only ATM strikes traded
        # that day) -> degenerate, non-2D point cloud for Delaunay triangulation.
        chain = pd.DataFrame({
            'Moneyness': [1.0, 1.0, 1.0, 1.0],
            'TTM': [30 / 365.0, 60 / 365.0, 90 / 365.0, 120 / 365.0],
            'impl_volatility': [0.20, 0.21, 0.22, 0.23],
        })
        surface = standardize_surface(chain, self.TARGET_MONEYNESS, self.TARGET_TTM)
        assert surface.shape == (50,)
        assert not np.isnan(surface).any()

    def test_return_quality_false_by_default_keeps_old_return_type(self):
        chain = pd.DataFrame({
            'Moneyness': [1.0], 'TTM': [30 / 365.0], 'impl_volatility': [0.20],
        })
        result = standardize_surface(chain, self.TARGET_MONEYNESS, self.TARGET_TTM)
        assert isinstance(result, np.ndarray)
        assert result.shape == (50,)

    def test_return_quality_true_reports_fallback_fraction(self):
        chain = pd.DataFrame({
            'Moneyness': [1.0], 'TTM': [30 / 365.0], 'impl_volatility': [0.20],
        })
        surface, fallback_fraction = standardize_surface(
            chain, self.TARGET_MONEYNESS, self.TARGET_TTM, return_quality=True,
        )
        assert surface.shape == (50,)
        # A single known quote can't build a cubic interpolant at all, so
        # (almost) every grid point falls back to nearest-neighbor.
        assert fallback_fraction > 0.9

    def test_dense_well_spread_chain_has_low_fallback_fraction(self):
        np.random.seed(7)
        chain = pd.DataFrame({
            'Moneyness': np.random.uniform(0.7, 1.3, 80),
            'TTM': np.random.uniform(10 / 365.0, 200 / 365.0, 80),
            'impl_volatility': np.random.uniform(0.1, 0.5, 80),
        })
        surface, fallback_fraction = standardize_surface(
            chain, self.TARGET_MONEYNESS, self.TARGET_TTM, return_quality=True,
        )
        assert 0.0 <= fallback_fraction < 1.0
        assert not np.isnan(surface).any()


# ---------------------------------------------------------------------------
# Autoencoder: tensor boundaries and training mechanics
# ---------------------------------------------------------------------------

class TestVolatilityAutoencoder:

    @pytest.mark.parametrize("batch_size", [1, 8, 64, 200])
    def test_forward_pass_preserves_shape(self, batch_size):
        model = VolatilityAutoencoder()
        x = torch.randn(batch_size, 50)
        out = model(x)
        assert out.shape == (batch_size, 50)

    def test_latent_bottleneck_is_three_dimensional(self):
        model = VolatilityAutoencoder()
        x = torch.randn(4, 50)
        latent = model.encoder(x)
        assert latent.shape == (4, 3)

    def test_output_has_no_nan_or_inf(self):
        model = VolatilityAutoencoder()
        x = torch.randn(32, 50)
        out = model(x)
        assert torch.isfinite(out).all()

    def test_gradients_flow_and_remain_finite(self):
        model = VolatilityAutoencoder()
        x = torch.randn(16, 50)
        out = model(x)
        loss = nn_mse(out, x)
        loss.backward()
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient reached {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient in {name}"

    def test_wrong_input_dimension_raises(self):
        model = VolatilityAutoencoder()
        x = torch.randn(4, 49)
        with pytest.raises(RuntimeError):
            model(x)


def nn_mse(a, b):
    return ((a - b) ** 2).mean()


class TestEarlyStopping:

    def test_saves_checkpoint_and_updates_best_loss_on_improvement(self, tmp_path):
        ckpt = tmp_path / "model.pth"
        stopper = EarlyStopping(patience=3, path=str(ckpt))
        model = VolatilityAutoencoder()

        stopper(1.0, model)
        assert ckpt.exists()
        assert stopper.best_loss == 1.0
        assert stopper.counter == 0
        assert stopper.early_stop is False

    def test_counter_increments_without_improvement(self, tmp_path):
        ckpt = tmp_path / "model.pth"
        stopper = EarlyStopping(patience=3, path=str(ckpt))
        model = VolatilityAutoencoder()

        stopper(1.0, model)
        stopper(1.5, model)  # worse
        stopper(1.2, model)  # still worse than best (1.0)
        assert stopper.counter == 2
        assert stopper.early_stop is False

    def test_counter_resets_on_new_improvement(self, tmp_path):
        ckpt = tmp_path / "model.pth"
        stopper = EarlyStopping(patience=3, path=str(ckpt))
        model = VolatilityAutoencoder()

        stopper(1.0, model)
        stopper(1.5, model)
        stopper(0.5, model)  # improvement -> resets counter
        assert stopper.counter == 0
        assert stopper.best_loss == 0.5

    def test_triggers_after_patience_exhausted(self, tmp_path):
        ckpt = tmp_path / "model.pth"
        stopper = EarlyStopping(patience=2, path=str(ckpt))
        model = VolatilityAutoencoder()

        stopper(1.0, model)
        stopper(1.1, model)  # counter=1
        stopper(1.2, model)  # counter=2 >= patience -> stop
        assert stopper.early_stop is True


# ---------------------------------------------------------------------------
# Backtester: temporal logic (get_trading_day)
# ---------------------------------------------------------------------------

class TestGetTradingDay:

    @pytest.fixture
    def trading_days(self):
        # Mon-Fri week 1, Mon-Fri week 2 (two weekends embedded)
        return np.array([
            '2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08', '2026-01-09',
            '2026-01-12', '2026-01-13', '2026-01-14', '2026-01-15', '2026-01-16',
        ], dtype='datetime64[D]')

    def test_skips_single_weekend(self, trading_days):
        result = get_trading_day(np.datetime64('2026-01-09'), 1, trading_days)  # Fri -> Mon
        assert str(result) == '2026-01-12'

    def test_skips_two_weekends_over_wide_offset(self, trading_days):
        result = get_trading_day(np.datetime64('2026-01-05'), 8, trading_days)
        assert str(result) == '2026-01-15'

    def test_offset_zero_returns_same_day(self, trading_days):
        result = get_trading_day(np.datetime64('2026-01-07'), 0, trading_days)
        assert str(result) == '2026-01-07'

    def test_offset_beyond_array_end_clamps_to_last_day(self, trading_days):
        result = get_trading_day(np.datetime64('2026-01-16'), 5, trading_days)
        assert str(result) == '2026-01-16'

    def test_offset_just_past_end_clamps(self, trading_days):
        result = get_trading_day(np.datetime64('2026-01-15'), 2, trading_days)
        assert str(result) == '2026-01-16'

    def test_date_not_present_in_trading_days_raises_clear_error(self, trading_days):
        # A Saturday, holiday, or any date missing from the stock-data
        # calendar (e.g. a ticker with a trading halt) now raises a
        # descriptive, catchable ValueError instead of a bare IndexError.
        with pytest.raises(ValueError, match="not found in unique_trading_days"):
            get_trading_day(np.datetime64('2026-01-10'), 1, trading_days)  # Saturday

    def test_negative_offset_before_start_clamps_to_first_day(self, trading_days):
        # Fixed: target_idx < 0 is now clamped to the first trading day
        # instead of wrapping around via numpy negative indexing.
        result = get_trading_day(np.datetime64('2026-01-05'), -3, trading_days)
        assert str(result) == '2026-01-05'


# ---------------------------------------------------------------------------
# Backtester: ATM straddle selection
# ---------------------------------------------------------------------------

class TestGetAtmStraddle:

    def test_selects_expiration_closest_to_30_days_and_atm_strike(self):
        chain = pd.DataFrame({
            'exdate': pd.to_datetime(['2026-02-04', '2026-02-04', '2026-03-06', '2026-03-06']),
            'TTM': [30 / 365.0, 30 / 365.0, 60 / 365.0, 60 / 365.0],
            'strike_price': [100.0, 100.0, 100.0, 100.0],
            'cp_flag': ['C', 'P', 'C', 'P'],
            'Moneyness': [1.0, 1.0, 1.0, 1.0],
            'delta': [0.52, -0.48, 0.55, -0.45],
        })
        call, put = get_atm_straddle(chain)
        assert call['exdate'] == pd.Timestamp('2026-02-04')
        assert put['exdate'] == pd.Timestamp('2026-02-04')

    def test_selects_moneyness_closest_to_one(self):
        chain = pd.DataFrame({
            'exdate': pd.to_datetime(['2026-02-04'] * 4),
            'TTM': [30 / 365.0] * 4,
            'strike_price': [95.0, 100.0, 95.0, 100.0],
            'cp_flag': ['C', 'C', 'P', 'P'],
            'Moneyness': [0.95, 1.0, 0.95, 1.0],
            'delta': [0.7, 0.52, -0.3, -0.48],
        })
        call, put = get_atm_straddle(chain)
        assert call['strike_price'] == 100.0
        assert put['strike_price'] == 100.0

    def test_missing_put_at_atm_strike_raises_clear_error(self):
        # If data cleaning drops one leg at the chosen ATM strike (e.g. its
        # quote was crossed or had zero volume that day), get_atm_straddle
        # now raises a descriptive ValueError instead of an opaque IndexError.
        # calculate_5_day_straddle_pnl catches this and skips the trade
        # rather than aborting the whole backtest.
        chain = pd.DataFrame({
            'exdate': pd.to_datetime(['2026-02-04', '2026-02-04']),
            'TTM': [30 / 365.0, 30 / 365.0],
            'strike_price': [100.0, 105.0],
            'cp_flag': ['C', 'C'],
            'Moneyness': [1.0, 1.05],
            'delta': [0.52, 0.40],
        })
        with pytest.raises(ValueError, match="Missing call or put leg"):
            get_atm_straddle(chain)


# ---------------------------------------------------------------------------
# Backtester: full 5-day delta-hedged PnL accounting
# ---------------------------------------------------------------------------

class TestCalculate5DayStraddlePnl:

    @pytest.fixture
    def trading_days(self):
        return pd.to_datetime([
            '2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08', '2026-01-09', '2026-01-12',
        ])

    def _option_leg(self, date, exdate, strike, cp_flag, delta, premium, ttm):
        return {
            'date': date, 'exdate': exdate, 'ticker': 'AAPL', 'cp_flag': cp_flag,
            'strike_price': strike, 'Moneyness': 1.0, 'TTM': ttm,
            'impl_volatility': 0.25, 'delta': delta, 'Premium': premium,
        }

    def test_flat_stock_price_yields_zero_net_hedge_cash_flow(self, trading_days):
        # If the underlying never moves, every rebalance trades zero shares
        # (delta unchanged) and the terminal hedge liquidation exactly
        # offsets the initial hedge trade at the same price. Net hedge cash
        # flow must be exactly zero, isolating pure options PnL and proving
        # the buy/sell sign convention nets to zero for a delta-neutral book.
        exdate = pd.Timestamp('2026-02-04')
        option_rows = []
        for d in trading_days:
            option_rows.append(self._option_leg(d, exdate, 100.0, 'C', 0.52, 3.00, 30 / 365.0))
            option_rows.append(self._option_leg(d, exdate, 100.0, 'P', -0.48, 2.50, 30 / 365.0))
        option_data = pd.DataFrame(option_rows)

        stock_data = pd.DataFrame({
            'date': trading_days,
            'ticker': ['AAPL'] * len(trading_days),
            'close': [150.0] * len(trading_days),  # flat price, flat delta
        })

        signal_df = pd.DataFrame({'Date': [trading_days[0]], 'Ticker': ['AAPL']})
        result = calculate_5_day_straddle_pnl(signal_df, option_data, stock_data)

        assert len(result) == 1
        expected_pnl = (3.00 + 2.50) - (3.00 + 2.50)  # exit revenue - entry cost, no price movement
        assert result['PnL'].iloc[0] == pytest.approx(expected_pnl)

    def test_pnl_matches_hand_computed_value_with_price_drift(self, trading_days):
        exdate = pd.Timestamp('2026-02-04')
        # delta and premium drift with the stock price across the 6 days
        deltas_c = [0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
        deltas_p = [-0.48, -0.45, -0.42, -0.40, -0.38, -0.35]
        premiums_c = [3.00, 3.10, 3.25, 3.35, 3.50, 3.70]
        premiums_p = [2.50, 2.40, 2.30, 2.20, 2.10, 1.95]
        prices = [150.0, 151.0, 152.5, 153.5, 155.0, 157.0]

        option_rows = []
        for i, d in enumerate(trading_days):
            option_rows.append(self._option_leg(d, exdate, 100.0, 'C', deltas_c[i], premiums_c[i], 30 / 365.0))
            option_rows.append(self._option_leg(d, exdate, 100.0, 'P', deltas_p[i], premiums_p[i], 30 / 365.0))
        option_data = pd.DataFrame(option_rows)
        stock_data = pd.DataFrame({
            'date': trading_days,
            'ticker': ['AAPL'] * len(trading_days),
            'close': prices,
        })

        signal_df = pd.DataFrame({'Date': [trading_days[0]], 'Ticker': ['AAPL']})
        result = calculate_5_day_straddle_pnl(signal_df, option_data, stock_data)

        # Hand-replicate the function's own accounting to pin down expected PnL.
        net_deltas = [c + p for c, p in zip(deltas_c, deltas_p)]
        shares = [-nd * 100 for nd in net_deltas]
        cash_flow = -(shares[0] * prices[0])
        for i in range(1, 5):
            traded = shares[i] - shares[i - 1]
            cash_flow -= traded * prices[i]
        cash_flow += shares[4] * prices[5]
        initial_cost = premiums_c[0] + premiums_p[0]
        final_revenue = premiums_c[5] + premiums_p[5]
        expected_pnl = final_revenue - initial_cost + cash_flow

        assert len(result) == 1
        assert result['PnL'].iloc[0] == pytest.approx(expected_pnl)

    def test_hedge_buy_is_negative_cash_flow_and_short_proceeds_positive(self, trading_days):
        # Net delta > 0 at entry => shares_held = -net_delta*100 < 0 (short
        # stock to hedge long premium). Selling stock short must be a cash
        # *inflow* at t0. We isolate this by checking sign consistency
        # rather than re-deriving the whole PnL.
        exdate = pd.Timestamp('2026-02-04')
        # strongly positive net delta at entry (call dominates put)
        option_rows = [
            self._option_leg(trading_days[0], exdate, 100.0, 'C', 0.80, 5.00, 30 / 365.0),
            self._option_leg(trading_days[0], exdate, 100.0, 'P', -0.20, 1.00, 30 / 365.0),
        ]
        for d in trading_days[1:]:
            option_rows.append(self._option_leg(d, exdate, 100.0, 'C', 0.80, 5.00, 30 / 365.0))
            option_rows.append(self._option_leg(d, exdate, 100.0, 'P', -0.20, 1.00, 30 / 365.0))
        option_data = pd.DataFrame(option_rows)
        stock_data = pd.DataFrame({
            'date': trading_days,
            'ticker': ['AAPL'] * len(trading_days),
            'close': [150.0] * len(trading_days),
        })
        signal_df = pd.DataFrame({'Date': [trading_days[0]], 'Ticker': ['AAPL']})
        result = calculate_5_day_straddle_pnl(signal_df, option_data, stock_data)

        # net_delta = 0.60 -> shares_held = -60 (short 60 shares at 150 -> +9000 cash)
        # flat price -> hedge unwinds for exactly -9000 at exit -> net hedge cash flow = 0
        expected_pnl = (5.00 + 1.00) - (5.00 + 1.00)
        assert result['PnL'].iloc[0] == pytest.approx(expected_pnl)

    def test_missing_entry_chain_is_skipped_not_raised(self, trading_days):
        option_data = pd.DataFrame(columns=[
            'date', 'exdate', 'ticker', 'cp_flag', 'strike_price',
            'Moneyness', 'TTM', 'impl_volatility', 'delta', 'Premium',
        ])
        stock_data = pd.DataFrame({'date': trading_days, 'ticker': ['AAPL'] * len(trading_days), 'close': [150.0] * len(trading_days)})
        signal_df = pd.DataFrame({'Date': [trading_days[0]], 'Ticker': ['AAPL']})

        result = calculate_5_day_straddle_pnl(signal_df, option_data, stock_data)
        assert result.empty

    def test_missing_atm_leg_on_entry_day_is_skipped_not_raised(self, trading_days):
        # A signal day where the ATM strike only has a call quote (put was
        # crossed/zero-volume and filtered upstream) must not crash the
        # whole backtest -- it should just skip that one trade.
        exdate = pd.Timestamp('2026-02-04')
        option_data = pd.DataFrame([
            self._option_leg(trading_days[0], exdate, 100.0, 'C', 0.52, 3.00, 30 / 365.0),
            # no put leg at strike 100.0 on entry day
        ])
        stock_data = pd.DataFrame({
            'date': trading_days,
            'ticker': ['AAPL'] * len(trading_days),
            'close': [150.0] * len(trading_days),
        })
        signal_df = pd.DataFrame({'Date': [trading_days[0]], 'Ticker': ['AAPL']})

        result = calculate_5_day_straddle_pnl(signal_df, option_data, stock_data)
        assert result.empty

    def test_missing_exit_chain_is_skipped_not_raised(self, trading_days):
        exdate = pd.Timestamp('2026-02-04')
        # only provide data for day 0, none for the rebalance/exit days
        option_data = pd.DataFrame([
            self._option_leg(trading_days[0], exdate, 100.0, 'C', 0.52, 3.00, 30 / 365.0),
            self._option_leg(trading_days[0], exdate, 100.0, 'P', -0.48, 2.50, 30 / 365.0),
        ])
        stock_data = pd.DataFrame({
            'date': trading_days,
            'ticker': ['AAPL'] * len(trading_days),
            'close': [150.0] * len(trading_days),
        })
        signal_df = pd.DataFrame({'Date': [trading_days[0]], 'Ticker': ['AAPL']})

        result = calculate_5_day_straddle_pnl(signal_df, option_data, stock_data)
        assert result.empty, "Trade with no exit-day data should be dropped, not crash or fabricate a PnL"

    def test_missing_stock_price_on_entry_day_is_skipped_not_raised(self, trading_days):
        # A signal day with a valid option chain but no matching row in
        # stock_data (e.g. a gap in the price feed) used to crash with an
        # uncaught IndexError from `.values[0]` on an empty match. It must
        # now be skipped like any other incomplete trade.
        exdate = pd.Timestamp('2026-02-04')
        option_data = pd.DataFrame([
            self._option_leg(trading_days[0], exdate, 100.0, 'C', 0.52, 3.00, 30 / 365.0),
            self._option_leg(trading_days[0], exdate, 100.0, 'P', -0.48, 2.50, 30 / 365.0),
        ])
        # stock_data has no row at all for the entry date
        stock_data = pd.DataFrame({
            'date': trading_days[1:],
            'ticker': ['AAPL'] * (len(trading_days) - 1),
            'close': [150.0] * (len(trading_days) - 1),
        })
        signal_df = pd.DataFrame({'Date': [trading_days[0]], 'Ticker': ['AAPL']})

        result = calculate_5_day_straddle_pnl(signal_df, option_data, stock_data)
        assert result.empty

    def test_hold_days_parameter_exits_on_shorter_horizon(self, trading_days):
        # hold_days=3 should exit using the chain 3 trading days out
        # (2026-01-08), not the default 5-day horizon (2026-01-12).
        exdate = pd.Timestamp('2026-02-04')
        premiums_c = [3.00, 3.10, 3.25, 3.35, 3.50, 3.70]
        premiums_p = [2.50, 2.40, 2.30, 2.20, 2.10, 1.95]
        deltas_c = [0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
        deltas_p = [-0.48, -0.45, -0.42, -0.40, -0.38, -0.35]
        prices = [150.0, 151.0, 152.5, 153.5, 155.0, 157.0]

        option_rows = []
        for i, d in enumerate(trading_days):
            option_rows.append(self._option_leg(d, exdate, 100.0, 'C', deltas_c[i], premiums_c[i], 30 / 365.0))
            option_rows.append(self._option_leg(d, exdate, 100.0, 'P', deltas_p[i], premiums_p[i], 30 / 365.0))
        option_data = pd.DataFrame(option_rows)
        stock_data = pd.DataFrame({'date': trading_days, 'ticker': ['AAPL'] * len(trading_days), 'close': prices})
        signal_df = pd.DataFrame({'Date': [trading_days[0]], 'Ticker': ['AAPL']})

        result = calculate_5_day_straddle_pnl(signal_df, option_data, stock_data, hold_days=3)

        # Hand-replicate using only days 0..3 (exit at index 3, rebalance at 1,2).
        net_deltas = [c + p for c, p in zip(deltas_c, deltas_p)]
        shares = [-nd * 100 for nd in net_deltas]
        cash_flow = -(shares[0] * prices[0])
        for i in range(1, 3):
            traded = shares[i] - shares[i - 1]
            cash_flow -= traded * prices[i]
        cash_flow += shares[2] * prices[3]
        initial_cost = premiums_c[0] + premiums_p[0]
        final_revenue = premiums_c[3] + premiums_p[3]
        expected_pnl = final_revenue - initial_cost + cash_flow

        assert len(result) == 1
        assert result['PnL'].iloc[0] == pytest.approx(expected_pnl)

    def test_target_notional_scales_pnl_proportionally(self, trading_days):
        exdate = pd.Timestamp('2026-02-04')
        premiums_c = [3.00, 3.10, 3.25, 3.35, 3.50, 3.70]
        premiums_p = [2.50, 2.40, 2.30, 2.20, 2.10, 1.95]
        deltas_c = [0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
        deltas_p = [-0.48, -0.45, -0.42, -0.40, -0.38, -0.35]
        prices = [150.0, 151.0, 152.5, 153.5, 155.0, 157.0]

        option_rows = []
        for i, d in enumerate(trading_days):
            option_rows.append(self._option_leg(d, exdate, 100.0, 'C', deltas_c[i], premiums_c[i], 30 / 365.0))
            option_rows.append(self._option_leg(d, exdate, 100.0, 'P', deltas_p[i], premiums_p[i], 30 / 365.0))
        option_data = pd.DataFrame(option_rows)
        stock_data = pd.DataFrame({'date': trading_days, 'ticker': ['AAPL'] * len(trading_days), 'close': prices})
        signal_df = pd.DataFrame({'Date': [trading_days[0]], 'Ticker': ['AAPL']})

        baseline = calculate_5_day_straddle_pnl(signal_df, option_data, stock_data)
        # entry price is 150, so target_notional=30000 -> contracts=2.0 (double the flat 1-contract baseline)
        scaled = calculate_5_day_straddle_pnl(signal_df, option_data, stock_data, target_notional=30000.0)

        assert scaled['PnL'].iloc[0] == pytest.approx(2.0 * baseline['PnL'].iloc[0])

    def test_direction_aware_short_flips_sign_of_long_pnl(self, trading_days):
        exdate = pd.Timestamp('2026-02-04')
        premiums_c = [3.00, 3.10, 3.25, 3.35, 3.50, 3.70]
        premiums_p = [2.50, 2.40, 2.30, 2.20, 2.10, 1.95]
        deltas_c = [0.52, 0.55, 0.58, 0.60, 0.62, 0.65]
        deltas_p = [-0.48, -0.45, -0.42, -0.40, -0.38, -0.35]
        prices = [150.0, 151.0, 152.5, 153.5, 155.0, 157.0]

        option_rows = []
        for i, d in enumerate(trading_days):
            option_rows.append(self._option_leg(d, exdate, 100.0, 'C', deltas_c[i], premiums_c[i], 30 / 365.0))
            option_rows.append(self._option_leg(d, exdate, 100.0, 'P', deltas_p[i], premiums_p[i], 30 / 365.0))
        option_data = pd.DataFrame(option_rows)
        stock_data = pd.DataFrame({'date': trading_days, 'ticker': ['AAPL'] * len(trading_days), 'close': prices})

        baseline = calculate_5_day_straddle_pnl(
            pd.DataFrame({'Date': [trading_days[0]], 'Ticker': ['AAPL']}), option_data, stock_data
        )

        shorted = calculate_5_day_straddle_pnl(
            pd.DataFrame({'Date': [trading_days[0]], 'Ticker': ['AAPL'], 'Residual': [0.05]}),
            option_data, stock_data,
        )
        assert shorted['Direction'].iloc[0] == 'short'
        assert shorted['PnL'].iloc[0] == pytest.approx(-baseline['PnL'].iloc[0])

        longed = calculate_5_day_straddle_pnl(
            pd.DataFrame({'Date': [trading_days[0]], 'Ticker': ['AAPL'], 'Residual': [-0.05]}),
            option_data, stock_data,
        )
        assert longed['Direction'].iloc[0] == 'long'
        assert longed['PnL'].iloc[0] == pytest.approx(baseline['PnL'].iloc[0])

    def test_missing_residual_column_defaults_to_long(self, trading_days):
        exdate = pd.Timestamp('2026-02-04')
        option_rows = []
        for d in trading_days:
            option_rows.append(self._option_leg(d, exdate, 100.0, 'C', 0.52, 3.00, 30 / 365.0))
            option_rows.append(self._option_leg(d, exdate, 100.0, 'P', -0.48, 2.50, 30 / 365.0))
        option_data = pd.DataFrame(option_rows)
        stock_data = pd.DataFrame({
            'date': trading_days, 'ticker': ['AAPL'] * len(trading_days), 'close': [150.0] * len(trading_days),
        })
        signal_df = pd.DataFrame({'Date': [trading_days[0]], 'Ticker': ['AAPL']})

        result = calculate_5_day_straddle_pnl(signal_df, option_data, stock_data)
        assert not result.empty
        assert (result['Direction'] == 'long').all()


# ---------------------------------------------------------------------------
# Backtester: vectorized performance tear sheet
# ---------------------------------------------------------------------------

class TestCalculatePerformanceMetrics:

    def test_win_rate_and_average_pnl(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05', '2026-01-06', '2026-01-07', '2026-01-08']),
            'ticker': ['AAPL', 'MSFT', 'GOOG', 'AMZN'],
            'sector': ['Tech', 'Tech', 'Tech', 'Consumer'],
            'net_pnl': [100.0, -50.0, 80.0, -20.0],
            'capital_allocated': [1000.0] * 4,
        })
        summary, _ = calculate_performance_metrics(df)
        assert summary['Win Rate (%)'] == pytest.approx(50.0)
        assert summary['Average PnL per Trade ($)'] == pytest.approx((100 - 50 + 80 - 20) / 4)

    def test_total_cumulative_return_matches_compounded_equity_curve(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05', '2026-01-06', '2026-01-07']),
            'ticker': ['AAPL'] * 3,
            'sector': ['Tech'] * 3,
            'net_pnl': [100.0, -50.0, 80.0],
            'capital_allocated': [1000.0] * 3,
        })
        summary, _ = calculate_performance_metrics(df)

        daily_returns = np.array([100 / 1000, -50 / 1000, 80 / 1000])
        expected_equity = np.cumprod(1 + daily_returns)
        expected_return_pct = (expected_equity[-1] - 1) * 100
        assert summary['Total Cumulative Return (%)'] == pytest.approx(expected_return_pct)

    def test_max_drawdown_matches_peak_to_trough_calc(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05', '2026-01-06', '2026-01-07']),
            'ticker': ['AAPL'] * 3,
            'sector': ['Tech'] * 3,
            'net_pnl': [100.0, -50.0, 80.0],
            'capital_allocated': [1000.0] * 3,
        })
        summary, _ = calculate_performance_metrics(df)

        daily_returns = np.array([0.10, -0.05, 0.08])
        equity = np.cumprod(1 + daily_returns)
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        expected_mdd_pct = abs(dd.min()) * 100

        assert summary['Max Drawdown (%)'] == pytest.approx(expected_mdd_pct)
        assert expected_mdd_pct == pytest.approx(5.0)  # sanity: matches the known dip after day 1

    def test_sharpe_ratio_matches_textbook_formula(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05', '2026-01-06', '2026-01-07']),
            'ticker': ['AAPL'] * 3,
            'sector': ['Tech'] * 3,
            'net_pnl': [100.0, -50.0, 80.0],
            'capital_allocated': [1000.0] * 3,
        })
        summary, _ = calculate_performance_metrics(df, risk_free_rate=0.04)

        daily_returns = pd.Series([0.10, -0.05, 0.08])
        daily_rf = 0.04 / 252
        expected_sharpe = (daily_returns.mean() - daily_rf) / daily_returns.std() * np.sqrt(252)
        assert summary['Annualized Sharpe Ratio'] == pytest.approx(expected_sharpe)

    def test_multiple_trades_same_day_are_capital_weighted_not_averaged(self):
        # Two trades close on the same day with very different capital
        # sizes -- the day's return must be pooled PnL / pooled capital,
        # not a naive mean of the two trades' individual returns.
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05', '2026-01-05']),
            'ticker': ['AAPL', 'TSLA'],
            'sector': ['Tech', 'Consumer'],
            'net_pnl': [10.0, 500.0],
            'capital_allocated': [100.0, 10000.0],
        })
        summary, _ = calculate_performance_metrics(df)

        pooled_return = (10.0 + 500.0) / (100.0 + 10000.0)
        expected_total_return_pct = pooled_return * 100
        naive_mean_of_individual_returns = ((10 / 100) + (500 / 10000)) / 2 * 100

        assert summary['Total Cumulative Return (%)'] == pytest.approx(expected_total_return_pct)
        assert summary['Total Cumulative Return (%)'] != pytest.approx(naive_mean_of_individual_returns)

    def test_sector_breakdown_isolates_trades_correctly(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05', '2026-01-06', '2026-01-05', '2026-01-06']),
            'ticker': ['AAPL', 'MSFT', 'XOM', 'CVX'],
            'sector': ['Tech', 'Tech', 'Energy', 'Energy'],
            'net_pnl': [100.0, -50.0, 30.0, 30.0],
            'capital_allocated': [1000.0, 1000.0, 500.0, 500.0],
        })
        _, sector_summary = calculate_performance_metrics(df)

        assert set(sector_summary.index) == {'Tech', 'Energy'}
        assert sector_summary.loc['Tech', 'Total PnL ($)'] == pytest.approx(50.0)
        assert sector_summary.loc['Energy', 'Total PnL ($)'] == pytest.approx(60.0)
        assert sector_summary.loc['Tech', 'Win Rate (%)'] == pytest.approx(50.0)
        assert sector_summary.loc['Energy', 'Win Rate (%)'] == pytest.approx(100.0)
        assert sector_summary.loc['Tech', 'Trade Count'] == 2
        assert sector_summary.loc['Energy', 'Trade Count'] == 2

    def test_sector_with_zero_variance_pnl_has_nan_sharpe(self):
        # A sector with wildly volatile PnL shouldn't affect another
        # sector's independently-computed Sharpe ratio.
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05', '2026-01-06', '2026-01-07',
                                     '2026-01-05', '2026-01-06', '2026-01-07']),
            'ticker': ['AAPL', 'AAPL', 'AAPL', 'TSLA', 'TSLA', 'TSLA'],
            'sector': ['Tech', 'Tech', 'Tech', 'Consumer', 'Consumer', 'Consumer'],
            'net_pnl': [10.0, 10.0, 10.0, -5000.0, 8000.0, -3000.0],
            'capital_allocated': [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 1000.0],
        })
        _, sector_summary = calculate_performance_metrics(df)

        # Tech has identical daily returns every day -> zero variance -> Sharpe undefined
        assert pd.isna(sector_summary.loc['Tech', 'Sharpe Ratio'])
        # Consumer's extreme volatility must not leak into Tech's row
        assert sector_summary.loc['Tech', 'Total PnL ($)'] == pytest.approx(30.0)

    def test_single_day_of_data_yields_zero_drawdown_and_nan_sharpe(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05']),
            'ticker': ['AAPL'],
            'sector': ['Tech'],
            'net_pnl': [50.0],
            'capital_allocated': [1000.0],
        })
        summary, _ = calculate_performance_metrics(df)
        assert summary['Max Drawdown (%)'] == pytest.approx(0.0)
        assert pd.isna(summary['Annualized Sharpe Ratio'])  # std of a single point is undefined

    def test_empty_trade_results_does_not_crash(self):
        df = pd.DataFrame(columns=['date', 'ticker', 'sector', 'net_pnl', 'capital_allocated'])
        summary, sector_summary = calculate_performance_metrics(df)
        assert summary['Total Cumulative Return (%)'] == 0.0
        assert summary['Max Drawdown (%)'] == 0.0
        assert sector_summary.empty

    def test_risk_free_rate_lowers_sharpe_ratio_as_it_increases(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05', '2026-01-06', '2026-01-07']),
            'ticker': ['AAPL'] * 3,
            'sector': ['Tech'] * 3,
            'net_pnl': [100.0, -50.0, 80.0],
            'capital_allocated': [1000.0] * 3,
        })
        summary_low_rf, _ = calculate_performance_metrics(df, risk_free_rate=0.0)
        summary_high_rf, _ = calculate_performance_metrics(df, risk_free_rate=0.20)
        assert summary_low_rf['Annualized Sharpe Ratio'] > summary_high_rf['Annualized Sharpe Ratio']

    def test_sector_summary_has_expected_shape_and_columns(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05']),
            'ticker': ['AAPL'],
            'sector': ['Tech'],
            'net_pnl': [50.0],
            'capital_allocated': [1000.0],
        })
        _, sector_summary = calculate_performance_metrics(df)
        assert isinstance(sector_summary, pd.DataFrame)
        assert list(sector_summary.columns) == ['Sharpe Ratio', 'Win Rate (%)', 'Total PnL ($)', 'Trade Count']

    def test_all_eleven_gics_sectors_produce_independent_rows(self):
        sectors = [
            'Information Technology', 'Health Care', 'Financials', 'Consumer Discretionary',
            'Communication Services', 'Industrials', 'Consumer Staples', 'Energy',
            'Utilities', 'Real Estate', 'Materials',
        ]
        rng = np.random.default_rng(0)
        dates = pd.bdate_range('2026-01-05', periods=20)
        rows = []
        for sector in sectors:
            for d in rng.choice(dates, 15):
                rows.append({
                    'date': d, 'ticker': sector[:3].upper(), 'sector': sector,
                    'net_pnl': rng.normal(0, 100), 'capital_allocated': 1000.0,
                })
        df = pd.DataFrame(rows)

        _, sector_summary = calculate_performance_metrics(df)
        assert len(sector_summary) == 11
        assert set(sector_summary.index) == set(sectors)
        assert sector_summary['Trade Count'].sum() == len(df)

    def test_default_capital_weighting_can_break_below_negative_100_pct_return(self):
        # Documents the actual failure mode found on real data: a thin day
        # (tiny pooled capital_allocated) with a loss larger than that
        # day's own notional produces a daily return < -100%, flipping the
        # sign of the compounded equity curve. This is the default
        # (total_portfolio_capital=None) behavior, kept for backward
        # compatibility -- not something to "fix" in this mode, just to
        # document so the fixed-capital-base mode below can be shown to
        # avoid it.
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05']),
            'ticker': ['AAPL'], 'sector': ['Tech'],
            'net_pnl': [-500.0], 'capital_allocated': [100.0],  # loses 5x its own notional
        })
        summary, _ = calculate_performance_metrics(df)
        assert summary['Total Cumulative Return (%)'] == pytest.approx(-500.0)  # ((1 + (-500/100)) - 1) * 100

    def test_fixed_total_portfolio_capital_avoids_the_sign_flip(self):
        # Same pathological single-trade loss as above, but normalized
        # against a fixed, realistic total capital base instead of that
        # day's own tiny notional -- the return should be a small, sane
        # negative number, not a equity-curve-breaking -600%.
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05']),
            'ticker': ['AAPL'], 'sector': ['Tech'],
            'net_pnl': [-500.0], 'capital_allocated': [100.0],
        })
        summary, _ = calculate_performance_metrics(df, total_portfolio_capital=1_000_000.0)
        expected_return_pct = (-500.0 / 1_000_000.0) * 100
        assert summary['Total Cumulative Return (%)'] == pytest.approx(expected_return_pct)
        assert summary['Total Cumulative Return (%)'] > -1.0  # sane, small magnitude

    def test_fixed_total_portfolio_capital_applies_to_sector_breakdown_too(self):
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05', '2026-01-06']),
            'ticker': ['AAPL', 'AAPL'], 'sector': ['Tech', 'Tech'],
            'net_pnl': [-500.0, 300.0], 'capital_allocated': [100.0, 100.0],
        })
        _, sector_summary = calculate_performance_metrics(df, total_portfolio_capital=1_000_000.0)
        # With a huge fixed capital base, daily returns are tiny and stable
        # enough that Sharpe is computable (finite), unlike the default
        # mode where a -500% single-day return would dominate the series.
        assert np.isfinite(sector_summary.loc['Tech', 'Sharpe Ratio'])

    def test_total_portfolio_capital_none_matches_original_default_behavior(self):
        # Regression guard: omitting the new parameter must reproduce the
        # exact numbers the function produced before it existed.
        df = pd.DataFrame({
            'date': pd.to_datetime(['2026-01-05', '2026-01-06', '2026-01-07']),
            'ticker': ['AAPL'] * 3, 'sector': ['Tech'] * 3,
            'net_pnl': [100.0, -50.0, 80.0], 'capital_allocated': [1000.0] * 3,
        })
        default_summary, _ = calculate_performance_metrics(df)
        explicit_none_summary, _ = calculate_performance_metrics(df, total_portfolio_capital=None)
        assert default_summary == explicit_none_summary


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-v']))

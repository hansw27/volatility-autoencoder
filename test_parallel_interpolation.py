"""
Coverage for the multiprocessing refactor of the surface-interpolation
stage in main.py:

- data_pipeline.interpolate_surface_arrays: the pure-numpy core extracted
  from standardize_surface so it can be shipped across a process pool
  without pickling pandas DataFrame slices.
- main.process_single_surface: the top-level, picklable worker function
  built on top of it.
- The real ProcessPoolExecutor usage in main.py itself: correctness
  (matches sequential results, preserves order), the 'spawn' start method
  (chosen specifically to avoid a fork-after-threads deadlock hazard with
  PyTorch's internal thread pool), and edge cases like more workers than
  tasks or every surface being filtered out.
"""
import multiprocessing
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import pytest

from data_pipeline import standardize_surface, interpolate_surface_arrays
import main as main_module
from test_pipeline_integration import _write_synthetic_wrds_dataset

TARGET_MONEYNESS = np.linspace(0.8, 1.2, 10)
TARGET_TTM = np.array([30, 60, 90, 120, 180]) / 365.0


def _dense_chain(seed=0, n=80):
    rng = np.random.default_rng(seed)
    moneyness = rng.uniform(0.7, 1.3, n)
    ttm = rng.uniform(10 / 365.0, 200 / 365.0, n)
    vol = rng.uniform(0.1, 0.5, n)
    return moneyness, ttm, vol


# ---------------------------------------------------------------------------
# interpolate_surface_arrays: pure-numpy core
# ---------------------------------------------------------------------------

class TestInterpolateSurfaceArrays:

    def test_matches_standardize_surface_dataframe_path(self):
        moneyness, ttm, vol = _dense_chain()
        df = pd.DataFrame({'Moneyness': moneyness, 'TTM': ttm, 'impl_volatility': vol})

        via_dataframe, frac_df = standardize_surface(df, TARGET_MONEYNESS, TARGET_TTM, return_quality=True)
        known_points = np.column_stack([moneyness, ttm])
        via_arrays, frac_arr = interpolate_surface_arrays(
            known_points, vol, TARGET_MONEYNESS, TARGET_TTM, return_quality=True,
        )

        assert np.allclose(via_dataframe, via_arrays)
        assert frac_df == pytest.approx(frac_arr)

    def test_works_without_any_dataframe_involved(self):
        # Pure numpy in, pure numpy out -- no pandas dependency at all,
        # which is exactly what makes this cheap to ship across a pool.
        known_points = np.array([[0.95, 30 / 365.0], [1.0, 30 / 365.0],
                                  [1.05, 60 / 365.0], [0.9, 90 / 365.0]])
        known_vols = np.array([0.20, 0.21, 0.22, 0.23])
        surface = interpolate_surface_arrays(known_points, known_vols, TARGET_MONEYNESS, TARGET_TTM)
        assert surface.shape == (50,)
        assert not np.isnan(surface).any()

    def test_return_quality_false_by_default(self):
        known_points = np.array([[1.0, 30 / 365.0]])
        known_vols = np.array([0.2])
        result = interpolate_surface_arrays(known_points, known_vols, TARGET_MONEYNESS, TARGET_TTM)
        assert isinstance(result, np.ndarray)

    def test_raises_clear_error_on_empty_input(self):
        with pytest.raises(ValueError, match="empty options chain"):
            interpolate_surface_arrays(np.empty((0, 2)), np.empty(0), TARGET_MONEYNESS, TARGET_TTM)

    def test_sparse_points_fall_back_fully_without_crashing(self):
        known_points = np.array([[0.95, 30 / 365.0], [1.05, 60 / 365.0]])
        known_vols = np.array([0.20, 0.22])
        surface, frac = interpolate_surface_arrays(
            known_points, known_vols, TARGET_MONEYNESS, TARGET_TTM, return_quality=True,
        )
        assert surface.shape == (50,)
        assert not np.isnan(surface).any()
        assert frac == 1.0  # too few points for cubic -> fully nearest-neighbor


# ---------------------------------------------------------------------------
# main.process_single_surface: the picklable worker
# ---------------------------------------------------------------------------

class TestProcessSingleSurfaceWorker:

    def _task(self, date=pd.Timestamp('2026-01-05'), ticker='AAPL', seed=0, n=80):
        moneyness, ttm, vol = _dense_chain(seed, n)
        return (date, ticker, moneyness, ttm, vol, TARGET_MONEYNESS, TARGET_TTM)

    def test_returns_metadata_dict_and_surface_array(self):
        metadata, surface = main_module.process_single_surface(self._task())
        assert set(metadata.keys()) == {'Date', 'Ticker', 'Fallback_Fraction'}
        assert metadata['Date'] == pd.Timestamp('2026-01-05')
        assert metadata['Ticker'] == 'AAPL'
        assert isinstance(surface, np.ndarray)
        assert surface.shape == (50,)

    def test_matches_direct_interpolate_surface_arrays_call(self):
        date, ticker, moneyness, ttm, vol, tm, tt = self._task()
        metadata, surface = main_module.process_single_surface((date, ticker, moneyness, ttm, vol, tm, tt))

        known_points = np.column_stack([moneyness, ttm])
        expected_surface, expected_frac = interpolate_surface_arrays(
            known_points, vol, tm, tt, return_quality=True,
        )

        assert np.allclose(surface, expected_surface)
        assert metadata['Fallback_Fraction'] == pytest.approx(expected_frac)

    def test_worker_function_itself_is_picklable(self):
        # Required for ProcessPoolExecutor to dispatch it to worker
        # processes at all -- especially under the 'spawn' start method
        # main.py uses, which locates the target by qualified name in each
        # fresh worker interpreter rather than inheriting parent memory.
        restored = pickle.loads(pickle.dumps(main_module.process_single_surface))
        assert restored is main_module.process_single_surface

    def test_task_tuple_is_picklable_and_round_trips(self):
        task = self._task()
        date, ticker, moneyness, ttm, vol, tm, tt = pickle.loads(pickle.dumps(task))
        assert date == task[0]
        assert ticker == task[1]
        assert np.array_equal(moneyness, task[2])
        assert np.array_equal(ttm, task[3])
        assert np.array_equal(vol, task[4])

    def test_task_payload_is_plain_arrays_not_a_dataframe_or_series(self):
        # Guards against regressing back to shipping DataFrame slices
        # across the pool, which is what this whole refactor avoids.
        task = self._task()
        for arr in task[2:5]:
            assert isinstance(arr, np.ndarray)
            assert not isinstance(arr, (pd.Series, pd.DataFrame))


# ---------------------------------------------------------------------------
# Real ProcessPoolExecutor usage, matching main.py's actual configuration
# ---------------------------------------------------------------------------

class TestParallelPoolExecution:

    def _build_tasks(self, n_groups=12, seed=1):
        tasks = []
        rng = np.random.default_rng(seed)
        for i in range(n_groups):
            date = pd.Timestamp('2026-01-05') + pd.Timedelta(days=i)
            ticker = 'AAPL' if i % 2 == 0 else 'MSFT'
            n_pts = int(rng.integers(2, 100))
            moneyness, ttm, vol = _dense_chain(seed=i, n=n_pts)
            tasks.append((date, ticker, moneyness, ttm, vol, TARGET_MONEYNESS, TARGET_TTM))
        return tasks

    def test_pool_results_match_sequential_and_preserve_order(self):
        tasks = self._build_tasks()
        sequential = [main_module.process_single_surface(t) for t in tasks]

        mp_context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=4, mp_context=mp_context) as executor:
            parallel = list(executor.map(main_module.process_single_surface, tasks, chunksize=2, timeout=60))

        assert len(parallel) == len(sequential)
        for (seq_meta, seq_surf), (par_meta, par_surf) in zip(sequential, parallel):
            assert seq_meta == par_meta
            assert np.allclose(seq_surf, par_surf)

    def test_pool_completes_promptly_with_multiple_workers(self):
        # Regression guard for the fork-vs-spawn deadlock hazard: if this
        # ever silently reverted to fork (or otherwise deadlocked), the
        # `timeout` passed to .map raises TimeoutError here instead of
        # hanging the whole test run.
        tasks = self._build_tasks(n_groups=20, seed=2)
        mp_context = multiprocessing.get_context('spawn')
        start = time.time()
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1, mp_context=mp_context) as executor:
            results = list(executor.map(main_module.process_single_surface, tasks, chunksize=3, timeout=60))
        elapsed = time.time() - start
        assert len(results) == 20
        assert elapsed < 60


# ---------------------------------------------------------------------------
# main.py end-to-end: the pool wired into the real pipeline
# ---------------------------------------------------------------------------

class TestMainParallelIntegration:

    def test_main_runs_with_more_workers_than_tasks(self, tmp_path, monkeypatch, capsys):
        # Very few (date, ticker) groups relative to CPU count exercises
        # chunksize = max(1, len(tasks) // (n_workers * 4)) -- must not
        # divide down to zero or otherwise misbehave with a tiny task count.
        _write_synthetic_wrds_dataset(tmp_path, seed=5, n_days=2, tickers=('AAPL',))
        monkeypatch.chdir(tmp_path)

        import importlib
        importlib.reload(main_module)
        main_module.main()

        assert (tmp_path / 'strategy_results.csv').exists()
        out = capsys.readouterr().out
        assert 'worker processes' in out
        assert 'chunksize=1' in out

    def test_main_prints_task_and_worker_counts(self, tmp_path, monkeypatch, capsys):
        _write_synthetic_wrds_dataset(tmp_path, seed=6, n_days=10, tickers=('AAPL', 'MSFT'))
        monkeypatch.chdir(tmp_path)

        import importlib
        importlib.reload(main_module)
        main_module.main()

        out = capsys.readouterr().out
        assert 'Interpolating 20 surfaces across' in out

    @pytest.mark.parametrize("fake_cpu_count,expected_workers", [
        (32, 6),   # big machine: capped at MAX_WORKERS, not cpu_count - RESERVED_CORES (30)
        (8, 6),    # 8 - 2 reserved = 6, exactly at the cap
        (4, 2),    # small machine: 4 - 2 reserved = 2, under the cap
        (1, 1),    # tiny/CI machine: never drops below 1 worker
    ])
    def test_worker_count_is_capped_and_leaves_reserved_cores(
        self, tmp_path, monkeypatch, capsys, fake_cpu_count, expected_workers,
    ):
        # Regression guard for the RESERVED_CORES / MAX_WORKERS logic: on a
        # WSL/remote-dev box, saturating every core starves the editor's
        # own server process. This must hold across small and large
        # machines alike, never dropping to zero workers.
        _write_synthetic_wrds_dataset(tmp_path, seed=8, n_days=2, tickers=('AAPL',))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(main_module.os, 'cpu_count', lambda: fake_cpu_count)

        import importlib
        importlib.reload(main_module)
        # reload() resets os.cpu_count() to the real function since main
        # re-imports `os` fresh -- reapply the patch after reloading.
        monkeypatch.setattr(main_module.os, 'cpu_count', lambda: fake_cpu_count)

        main_module.main()

        out = capsys.readouterr().out
        assert f'across {expected_workers} worker processes' in out

    def test_all_surfaces_filtered_out_raises_clear_error(self, tmp_path, monkeypatch):
        # Deliberately degenerate: a single fixed expiration per day makes
        # every chain collinear in TTM, so every surface's
        # fallback_fraction is exactly 1.0 and MAX_FALLBACK_FRACTION
        # filters all of them out via the *parallel* path. main() must
        # fail with a clear, actionable error here instead of crashing
        # several steps later with an opaque PyTorch shape mismatch on an
        # empty tensor.
        trading_days = pd.bdate_range('2026-01-05', periods=6)
        exdate = trading_days[-1] + pd.Timedelta(days=60)
        opt_rows = []
        for d in trading_days:
            for m in [0.9, 0.95, 1.0, 1.05, 1.1]:
                for cp in ['C', 'P']:
                    opt_rows.append({
                        'date': d, 'exdate': exdate, 'ticker': 'AAPL', 'secid': 1001, 'cp_flag': cp,
                        'strike_price': round(100 * m) * 1000, 'volume': 50, 'open_interest': 200,
                        'impl_volatility': 0.2, 'best_bid': 1.0, 'best_offer': 1.1, 'delta': 0.3,
                    })
        stock_rows = [{'date': d, 'ticker': 'AAPL', 'secid': 1001, 'close': 100.0} for d in trading_days]
        pd.DataFrame(opt_rows).to_csv(tmp_path / 'wrds_options_raw.csv', index=False)
        pd.DataFrame(stock_rows).to_csv(tmp_path / 'wrds_stock_raw.csv', index=False)

        monkeypatch.chdir(tmp_path)
        import importlib
        importlib.reload(main_module)

        with pytest.raises(ValueError, match="No usable surfaces remain"):
            main_module.main()

    def test_main_uses_spawn_start_method(self, tmp_path, monkeypatch):
        # Locks in the deliberate choice of 'spawn' over the platform
        # default ('fork' on Linux) -- see main.py's comment on why fork
        # is unsafe here (PyTorch's internal thread pool + fork-after-
        # threads deadlock hazard). If this ever gets "simplified" back to
        # the default context, this test should catch it.
        _write_synthetic_wrds_dataset(tmp_path, seed=7, n_days=4, tickers=('AAPL',))
        monkeypatch.chdir(tmp_path)

        import importlib
        importlib.reload(main_module)

        captured_contexts = []
        real_executor_init = ProcessPoolExecutor.__init__

        def spy_init(self, *args, **kwargs):
            captured_contexts.append(kwargs.get('mp_context'))
            return real_executor_init(self, *args, **kwargs)

        monkeypatch.setattr(ProcessPoolExecutor, '__init__', spy_init)
        main_module.main()

        assert len(captured_contexts) == 1
        assert captured_contexts[0] is not None
        assert captured_contexts[0].get_start_method() == 'spawn'

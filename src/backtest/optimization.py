"""Optimization module for hyperparameter tuning."""

from __future__ import annotations
from typing import Dict, Any, List, Callable, Optional, Tuple
from itertools import product
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from src.backtest.core import calculate_metrics, run_backtest
from src.strategies import StrategyFactory


def grid_search(
    strategy_builder: Callable,
    close: pd.DataFrame,
    base_params: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    run_config: Dict[str, Any],
    metric: str = "Sharpe",
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Perform grid search over hyperparameters.
    
    Args:
        strategy_builder: Function that builds strategy from params
        close: Price data
        base_params: Base parameters for strategy
        param_grid: Dictionary of parameter names to list of values to test
        run_config: Backtest configuration
        metric: Metric to optimize (Sharpe, Total Return, Calmar, etc.)
        n_jobs: Number of parallel jobs (-1 = all cores)
        verbose: Show progress bar
    
    Returns:
        DataFrame with all results sorted by metric
    """
    keys = list(param_grid.keys())
    combinations = list(product(*[param_grid[k] for k in keys]))
    
    def evaluate(params_values):
        """Evaluate a single parameter combination."""
        params = base_params.copy()
        params.update(dict(zip(keys, params_values)))
        
        # Create strategy
        strategy = StrategyFactory.create(strategy_builder, params)
        strategy.prepare(close)
        
        # Generate weights
        weights = strategy.generate_weights(
            close=close,
            max_positions=run_config.get("max_positions", 10),
            start_dt=run_config.get("start"),
            sizing=run_config.get("position_sizing", "equal"),
            vol_lookback=run_config.get("volatility_lookback", 21),
            stop_loss_pct=run_config.get("stop_loss_pct"),
            take_profit_pct=run_config.get("take_profit_pct"),
        )
        
        # Run backtest
        result = run_backtest(close, weights, run_config, "optimization")
        metrics = result.metrics
        
        # Add parameters to results
        result_dict = {**params, **metrics}
        return result_dict
    
    # Run evaluations in parallel
    if n_jobs != 1 and verbose:
        print(f"Running grid search with {len(combinations)} combinations...")
    
    if n_jobs == 1:
        # Sequential
        results = []
        iterator = tqdm(combinations, desc="Grid search") if verbose else combinations
        for combo in iterator:
            results.append(evaluate(combo))
    else:
        # Parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(evaluate)(combo) for combo in tqdm(combinations, desc="Grid search", disable=not verbose)
        )
    
    df = pd.DataFrame(results)
    
    # Sort by metric
    if metric in df.columns:
        ascending = False if run_config.get("direction", "maximize") == "maximize" else True
        df = df.sort_values(metric, ascending=ascending)
    
    return df


def walk_forward_optimization(
    close: pd.DataFrame,
    strategy_name: str,
    base_params: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
    run_config: Dict[str, Any],
    train_years: int = 3,
    test_months: int = 6,
    metric: str = "Sharpe",
    n_jobs: int = -1,
) -> pd.DataFrame:
    """
    Perform walk-forward optimization.
    
    Args:
        close: Price data
        strategy_name: Name of strategy
        base_params: Base parameters
        param_grid: Parameter grid for optimization
        run_config: Backtest configuration
        train_years: Years of training data
        test_months: Months of test data
        metric: Metric to optimize
        n_jobs: Number of parallel jobs
    
    Returns:
        DataFrame with walk-forward results
    """
    start_date = close.index.min()
    end_date = close.index.max()
    train_start = start_date
    train_end = train_start + pd.DateOffset(years=train_years)
    
    results = []
    
    while train_end < end_date:
        test_start = train_end
        test_end = min(test_start + pd.DateOffset(months=test_months), end_date)
        
        if test_end <= test_start or (test_end - test_start).days < 5:
            break
        
        print(f"\nTraining: {train_start.date()} to {train_end.date()}")
        print(f"Testing: {test_start.date()} to {test_end.date()}")
        
        # Training data
        train_data = close.loc[train_start:train_end]
        
        # Optimize on training data
        train_config = run_config.copy()
        train_config["start"] = train_start.strftime("%Y-%m-%d")
        train_config["end"] = train_end.strftime("%Y-%m-%d")
        
        opt_results = grid_search(
            strategy_builder=strategy_name,
            close=train_data,
            base_params=base_params,
            param_grid=param_grid,
            run_config=train_config,
            metric=metric,
            n_jobs=n_jobs,
            verbose=False,
        )
        
        if opt_results.empty:
            break
        
        # Best parameters
        best_params = opt_results.iloc[0].to_dict()
        best_params = {k: best_params[k] for k in param_grid}
        
        # Test on out-of-sample data
        test_data = close.loc[test_start:test_end]
        
        strategy = StrategyFactory.create(strategy_name, {**base_params, **best_params})
        strategy.prepare(test_data)
        
        weights = strategy.generate_weights(
            close=test_data,
            max_positions=run_config.get("max_positions", 10),
            start_dt=None,  # Use full test data
            sizing=run_config.get("position_sizing", "equal"),
            vol_lookback=run_config.get("volatility_lookback", 21),
            stop_loss_pct=run_config.get("stop_loss_pct"),
            take_profit_pct=run_config.get("take_profit_pct"),
        )
        
        test_config = run_config.copy()
        test_config["start"] = test_start.strftime("%Y-%m-%d")
        test_config["end"] = test_end.strftime("%Y-%m-%d")
        
        result = run_backtest(test_data, weights, test_config, strategy_name)
        metrics = result.metrics
        
        # Record results
        row = {
            "Train Start": train_start,
            "Train End": train_end,
            "Test Start": test_start,
            "Test End": test_end,
            **best_params,
            **metrics,
        }
        results.append(row)
        
        # Move window forward
        train_start = test_end
        train_end = min(train_start + pd.DateOffset(years=train_years), end_date)
    
    return pd.DataFrame(results)


def optimize_with_optuna(
    close: pd.DataFrame,
    strategy_name: str,
    base_params: Dict[str, Any],
    param_ranges: Dict[str, Tuple[float, float]],
    run_config: Dict[str, Any],
    metric: str = "Sharpe",
    n_trials: int = 100,
    direction: str = "maximize",
) -> Dict[str, Any]:
    """
    Use Optuna for Bayesian optimization.
    
    Args:
        close: Price data
        strategy_name: Name of strategy
        base_params: Base parameters
        param_ranges: Dict of parameter names to (min, max) tuples
        run_config: Backtest configuration
        metric: Metric to optimize
        n_trials: Number of optimization trials
        direction: "maximize" or "minimize"
    
    Returns:
        Best parameters found
    """
    try:
        import optuna
    except ImportError:
        raise ImportError("Optuna is required for this optimization method. Install with: pip install optuna")
    
    def objective(trial):
        # Suggest parameters
        params = base_params.copy()
        
        for param_name, (low, high) in param_ranges.items():
            if isinstance(low, int) and isinstance(high, int):
                params[param_name] = trial.suggest_int(param_name, low, high)
            else:
                params[param_name] = trial.suggest_float(param_name, low, high)
        
        # Create strategy
        strategy = StrategyFactory.create(strategy_name, params)
        strategy.prepare(close)
        
        # Generate weights
        weights = strategy.generate_weights(
            close=close,
            max_positions=run_config.get("max_positions", 10),
            start_dt=run_config.get("start"),
            sizing=run_config.get("position_sizing", "equal"),
            vol_lookback=run_config.get("volatility_lookback", 21),
            stop_loss_pct=run_config.get("stop_loss_pct"),
            take_profit_pct=run_config.get("take_profit_pct"),
        )
        
        # Run backtest
        result = run_backtest(close, weights, run_config, strategy_name)
        metrics = result.metrics
        
        return metrics.get(metric, -np.inf)
    
    # Create study
    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest parameters: {study.best_params}")
    print(f"Best {metric}: {study.best_value}")
    
    return study.best_params


def parameter_sensitivity_analysis(
    results_df: pd.DataFrame,
    param_cols: List[str],
    metric: str = "Sharpe",
) -> pd.DataFrame:
    """
    Analyze parameter sensitivity.
    
    Args:
        results_df: Results from grid search
        param_cols: List of parameter column names
        metric: Metric to analyze
    
    Returns:
        Sensitivity analysis DataFrame
    """
    sensitivity = {}
    
    for param in param_cols:
        if param not in results_df.columns:
            continue
        
        # Group by parameter and calculate statistics
        grouped = results_df.groupby(param)[metric].agg([
            'mean', 'std', 'min', 'max', 'count'
        ])
        
        # Calculate range effect
        if len(grouped) > 1:
            range_effect = grouped['max'] - grouped['min']
            grouped['range_effect'] = range_effect
        
        sensitivity[param] = grouped
    
    return sensitivity
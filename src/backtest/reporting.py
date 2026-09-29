"""Reporting and visualization module."""

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional, List
from numbers import Number
import base64
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

from src.backtest.core import calculate_metrics, monthly_returns, quarterly_returns, BacktestResult

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def format_indian_number(value: Number, decimals: int = 2) -> str:
    """Format a number with Indian lakh/crore digit grouping."""
    if pd.isna(value):
        return "N/A"

    sign = "-" if value < 0 else ""
    formatted = f"{abs(float(value)):.{decimals}f}"
    if decimals:
        integer, fraction = formatted.split(".")
    else:
        integer, fraction = formatted, ""
    if len(integer) > 3:
        last_three = integer[-3:]
        remaining = integer[:-3]
        groups = []
        while remaining:
            groups.insert(0, remaining[-2:])
            remaining = remaining[:-2]
        integer = ",".join(groups + [last_three])
    return f"{sign}{integer}.{fraction}" if decimals else f"{sign}{integer}"

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def plot_equity_curves(results: Dict[str, BacktestResult], benchmark: Optional[pd.Series] = None):
    """Plot equity curves for all strategies."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, result in results.items():
        value = result.value
        normalized = value / value.iloc[0]
        normalized = normalized.where(normalized > 0)
        normalized.plot(ax=ax, label=name, linewidth=2)
    
    if benchmark is not None:
        bench_norm = (benchmark / benchmark.iloc[0]).where(lambda values: values > 0)
        bench_norm.plot(ax=ax, label='Benchmark', linewidth=2, linestyle='--', color='black')
    
    ax.set_yscale('log')
    ax.set_title('Equity Curves (Normalized, Log Scale)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Value (Normalized to 1, Log Scale)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    return fig

def plot_drawdown(results: Dict[str, BacktestResult]):
    """Plot drawdown for all strategies."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for name, result in results.items():
        value = result.value
        dd = value / value.cummax() - 1
        dd.plot(ax=ax, label=name, linewidth=1.5)
    
    ax.set_title('Drawdown', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Drawdown')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-1, 0.05)
    
    return fig

def plot_correlation_matrix(results: Dict[str, BacktestResult]):
    """Plot correlation matrix of strategy returns."""
    returns = {}
    for name, result in results.items():
        returns[name] = result.value.pct_change()
    
    df_returns = pd.DataFrame(returns).dropna()
    
    if len(df_returns.columns) < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, 'Need at least 2 strategies\nfor correlation matrix', 
                ha='center', va='center', fontsize=12)
        ax.set_title('Strategy Return Correlation')
        return fig
    
    corr = df_returns.corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right')
    ax.set_yticklabels(corr.index)
    
    # Add text values
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                          ha='center', va='center', color='black' if abs(corr.iloc[i, j]) < 0.5 else 'white',
                          fontsize=10)
    
    ax.set_title('Strategy Return Correlation', fontsize=14, fontweight='bold')
    fig.colorbar(im, ax=ax)
    
    return fig

def plot_rolling_sharpe(results: Dict[str, BacktestResult], window: int = 126):
    """Plot rolling Sharpe ratio."""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    for name, result in results.items():
        ret = result.value.pct_change()
        rolling_sharpe = ret.rolling(window).mean() / ret.rolling(window).std() * np.sqrt(252)
        rolling_sharpe.plot(ax=ax, label=name, linewidth=1.5)
    
    ax.set_title(f'Rolling Sharpe Ratio ({window} days)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Sharpe Ratio')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    return fig

def plot_trade_distribution(results: Dict[str, BacktestResult]):
    """Plot trade P&L distribution."""
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 4 * len(results)))
    if len(results) == 1:
        axes = [axes]
    
    for idx, (name, result) in enumerate(results.items()):
        trades = result.trade_log
        if trades.empty:
            axes[idx].text(0.5, 0.5, f'{name}: No trades', ha='center', va='center')
            continue
        
        pnl = trades['PnL'].dropna()
        if pnl.empty:
            axes[idx].text(0.5, 0.5, f'{name}: No P&L data', ha='center', va='center')
            continue
        
        axes[idx].hist(pnl, bins=30, edgecolor='black', alpha=0.7)
        axes[idx].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[idx].set_title(f'{name} - Trade P&L Distribution', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('P&L')
        axes[idx].set_ylabel('Frequency')
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_monthly_heatmap(value: pd.Series, title="Monthly Returns"):
    """Plot monthly return heatmap."""
    monthly = value.resample("ME").last().pct_change().dropna()
    if not monthly.empty:
        monthly = monthly.to_frame("Return")
        monthly["Year"] = monthly.index.year
        monthly["Month"] = monthly.index.month
        monthly = monthly.pivot(index="Year", columns="Month", values="Return")
    if monthly.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No monthly data available', ha='center', va='center')
        return fig
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(monthly, annot=True, fmt='.1%', cmap='RdYlGn', center=0, 
                cbar_kws={'label': 'Return'}, ax=ax, linewidths=1, linecolor='white')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Month')
    ax.set_ylabel('Year')
    
    return fig

def generate_charts(results: Dict[str, BacktestResult], benchmark: Optional[pd.Series] = None) -> Dict[str, str]:
    """Generate all charts and return as base64 strings."""
    charts = {}
    
    # Equity curves
    charts['equity'] = fig_to_base64(plot_equity_curves(results, benchmark))
    
    # Drawdown
    charts['drawdown'] = fig_to_base64(plot_drawdown(results))
    
    # Correlation matrix
    charts['correlation'] = fig_to_base64(plot_correlation_matrix(results))
    
    # Rolling Sharpe
    charts['rolling_sharpe'] = fig_to_base64(plot_rolling_sharpe(results))
    
    # Trade distribution
    charts['trade_distribution'] = fig_to_base64(plot_trade_distribution(results))
    
    # Monthly returns for first strategy
    if results:
        first_result = next(iter(results.values()))
        charts['monthly_returns'] = fig_to_base64(
            plot_monthly_heatmap(first_result.value, 'Monthly Returns')
        )
    
    return charts

def estimate_cagr_without_top_wins(result: BacktestResult, count: int = 5) -> float:
    """Estimate CAGR after removing the largest winning trade P&Ls."""
    value = result.value
    trades = result.trade_log
    if len(value) < 2 or trades.empty or "PnL" not in trades:
        return np.nan

    pnl = pd.to_numeric(trades["PnL"], errors="coerce").dropna()
    top_wins = pnl[pnl > 0].nlargest(count).sum()
    adjusted_final = value.iloc[-1] - top_wins
    years = (value.index[-1] - value.index[0]).days / 365.25
    if years <= 0 or adjusted_final <= 0 or value.iloc[0] <= 0:
        return np.nan
    return (adjusted_final / value.iloc[0]) ** (1 / years) - 1

def generate_html_report(
    results: Dict[str, BacktestResult],
    summary: pd.DataFrame,
    output_path: Path,
    benchmark: Optional[pd.Series] = None,
    config: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate comprehensive HTML report."""
    charts = generate_charts(results, benchmark)
    
    # Prepare sections with pre-formatted numbers
    sections = []
    for name, result in results.items():
        metrics = result.metrics or calculate_metrics(result.portfolio, value=result.value)
        monthly = monthly_returns(result.portfolio, value=result.value)
        quarterly = quarterly_returns(result.portfolio, value=result.value)
        
        # Format metrics for display
        formatted_metrics = {}
        for key, value in metrics.items():
            if pd.isna(value):
                formatted_metrics[key] = "N/A"
            elif isinstance(value, Number):
                if key == 'Average Win':
                    formatted_metrics[key] = format_indian_number(value)
                elif 'Rate' in key or 'Return' in key or 'Factor' in key or 'Win' in key:
                    formatted_metrics[key] = f"{value * 100:.2f}%" if abs(value) < 10 else f"{value:.2f}%"
                elif 'Ratio' in key or 'Sharpe' in key or 'Sortino' in key or 'Calmar' in key:
                    formatted_metrics[key] = f"{value:.2f}"
                elif 'Drawdown' in key:
                    formatted_metrics[key] = f"{abs(value) * 100:.2f}%"
                elif 'Trades' in key or 'Duration' in key:
                    formatted_metrics[key] = format_indian_number(value, decimals=0)
                else:
                    formatted_metrics[key] = format_indian_number(value)
            else:
                formatted_metrics[key] = str(value)
        
        # Trade stats
        trades = result.trade_log
        trade_stats = {}
        top_trades = "No trades"
        top_losses = "No losses"
        adjusted_cagr = estimate_cagr_without_top_wins(result)
        if not trades.empty:
            pnl = trades['PnL'].dropna()
            if not pnl.empty:
                profits = pnl[pnl > 0]
                losses = pnl[pnl < 0]
                winning_trades = trades[trades['PnL'] > 0].nlargest(5, 'PnL')
                losing_trades = trades[trades['PnL'] < 0].nsmallest(5, 'PnL')
                trade_stats = {
                    'Total Trades': len(trades),
                    'Winning Trades': len(trades[trades['PnL'] > 0]),
                    'Losing Trades': len(trades[trades['PnL'] < 0]),
                    'Win Rate': f"{len(trades[trades['PnL'] > 0]) / len(trades):.2%}",
                    'Total P&L': format_indian_number(pnl.sum()),
                    'Average P&L': format_indian_number(pnl.mean()),
                    'Median Profit': format_indian_number(profits.median()) if not profits.empty else 'N/A',
                    'Median Loss': format_indian_number(losses.median()) if not losses.empty else 'N/A',
                    'Top Quartile Profit (75th percentile)': format_indian_number(profits.quantile(0.75)) if not profits.empty else 'N/A',
                    'Bottom Quartile Loss (25th percentile)': format_indian_number(losses.quantile(0.25)) if not losses.empty else 'N/A',
                    'Max Win': format_indian_number(pnl.max()),
                    'Max Loss': format_indian_number(pnl.min()),
                    'CAGR Without Top 5 Winning Trades (estimated)': (
                        f"{adjusted_cagr:.2%}" if pd.notna(adjusted_cagr) else "N/A"
                    ),
                }
                top_trades = winning_trades.to_html(
                    classes="table table-striped table-sm",
                    index=False,
                    float_format=format_indian_number,
                )
                top_losses = losing_trades.to_html(
                    classes="table table-striped table-sm",
                    index=False,
                    float_format=format_indian_number,
                )
        
        sections.append({
            'name': name,
            'metrics': formatted_metrics,
            'metrics_raw': metrics,  # Keep raw for reference
            'monthly': monthly.to_html(float_format=lambda x: f"{x:.2%}" if pd.notna(x) else "", classes="table table-striped") if not monthly.empty else "No data",
            'quarterly': quarterly.to_html(float_format=lambda x: f"{x:.2%}" if pd.notna(x) else "", classes="table table-striped") if not quarterly.empty else "No data",
            'trade_stats': trade_stats,
            'top_trades': top_trades,
            'top_losses': top_losses,
        })
    
    # Template - Fixed version without abs issues
    template = Template("""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Backtest Report</title>
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 30px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }
        h2 { color: #34495e; margin-top: 30px; border-bottom: 2px solid #ecf0f1; padding-bottom: 8px; }
        h3 { color: #555; margin-top: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #3498db; color: white; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .table-sm { font-size: 0.9em; }
        .table-striped tbody tr:nth-child(odd) { background-color: #f2f2f2; }
        .metric-card { background: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; border-left: 4px solid #3498db; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
        .metric-item { background: white; padding: 10px; border-radius: 5px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .metric-value { font-size: 1.2em; font-weight: bold; color: #2c3e50; }
        .metric-label { font-size: 0.85em; color: #7f8c8d; }
        img { max-width: 100%; height: auto; margin: 15px 0; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
        .chart-container { background: white; padding: 15px; margin: 20px 0; border-radius: 5px; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .summary-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 15px; margin: 20px 0; }
        .summary-card { background: white; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }
        .summary-card .value { font-size: 1.8em; font-weight: bold; margin: 5px 0; }
        .summary-card .label { color: #7f8c8d; font-size: 0.9em; }
        .badge { display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 0.8em; font-weight: bold; }
        .badge-success { background: #27ae60; color: white; }
        .badge-danger { background: #e74c3c; color: white; }
        .badge-warning { background: #f39c12; color: white; }
        hr { margin: 30px 0; border: 0; border-top: 2px solid #ecf0f1; }
        .text-center { text-align: center; }
        .mt-20 { margin-top: 20px; }
    </style>
</head>
<body>
<div class="container">
    <h1>📊 Backtest Report</h1>
    <p><strong>Generated:</strong> {{ timestamp }}</p>
    <p><strong>Universe:</strong> {{ universe }}</p>
    
    {% if config %}
    <h2>Configuration</h2>
    <div class="metric-card">
        <strong>Period:</strong> {{ config.backtest.start }} to {{ config.backtest.end }}<br>
        <strong>Initial Capital:</strong> ₹{{ "{:,.0f}".format(config.backtest.initial_cash) }}<br>
        <strong>Max Positions:</strong> {{ config.backtest.max_positions }}<br>
        <strong>Fees:</strong> {{ "{:.2f}".format(config.backtest.fees * 100) }}%<br>
    </div>
    {% endif %}
    
    <h2>Strategy Comparison</h2>
    {{ comparison|safe }}
    
    <h2>Charts</h2>
    <div class="chart-container">
        <h3>Equity Curves</h3>
        <img src="data:image/png;base64,{{ charts.equity }}">
    </div>
    
    <div class="chart-container">
        <h3>Drawdown</h3>
        <img src="data:image/png;base64,{{ charts.drawdown }}">
    </div>
    
    <div class="chart-container">
        <h3>Strategy Correlation</h3>
        <img src="data:image/png;base64,{{ charts.correlation }}">
    </div>
    
    <div class="chart-container">
        <h3>Rolling Sharpe Ratio</h3>
        <img src="data:image/png;base64,{{ charts.rolling_sharpe }}">
    </div>
    
    <div class="chart-container">
        <h3>Trade Distribution</h3>
        <img src="data:image/png;base64,{{ charts.trade_distribution }}">
    </div>
    
    <div class="chart-container">
        <h3>Monthly Returns</h3>
        <img src="data:image/png;base64,{{ charts.monthly_returns }}">
    </div>
    
    {% for section in sections %}
    <hr>
    <h2>{{ section.name }}</h2>
    
    <h3>Performance Metrics</h3>
    <div class="metric-grid">
        {% for key, value in section.metrics.items() %}
        <div class="metric-item">
            <div class="metric-label">{{ key }}</div>
            <div class="metric-value">{{ value }}</div>
        </div>
        {% endfor %}
    </div>
    
    <h3>Trade Statistics</h3>
    <table class="table-striped">
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>
        {% for key, value in section.trade_stats.items() %}
        <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
        {% endfor %}
        </tbody>
    </table>
    
    <h3>Monthly Returns</h3>
    {{ section.monthly|safe }}
    
    <h3>Quarterly Returns</h3>
    {{ section.quarterly|safe }}
    
    <h3>Top 5 Winning Trades</h3>
    {{ section.top_trades|safe }}

    <h3>Top 5 Losing Trades</h3>
    {{ section.top_losses|safe }}
    {% endfor %}
</div>
</body>
</html>
    """)
    
    # Render
    html = template.render(
        timestamp=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        universe=config.get("backtest", {}).get("universe", "Unknown") if config else "Unknown",
        config=config,
        comparison=summary.to_html(
            float_format=format_indian_number,
            classes="table table-striped",
        ),
        charts=charts,
        sections=sections,
    )
    
    output_path.write_text(html, encoding='utf-8')
    return html

def export_excel(
    results: Dict[str, BacktestResult],
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Export results to Excel."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Summary
        summary.to_excel(writer, sheet_name='Summary')
        
        # Each strategy
        for name, result in results.items():
            safe_name = name[:25].replace('/', '_')
            
            # Metrics
            metrics = result.metrics or calculate_metrics(result.portfolio, value=result.value)
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_excel(writer, sheet_name=f'{safe_name}_Metrics', index=False)
            
            # Trade log
            if not result.trade_log.empty:
                result.trade_log.to_excel(writer, sheet_name=f'{safe_name}_Trades', index=False)
            
            # Diagnostics
            if not result.diagnostics.empty:
                result.diagnostics.to_excel(writer, sheet_name=f'{safe_name}_Signals', index=False)
            
            # Monthly returns
            monthly = monthly_returns(result.portfolio, value=result.value)
            if not monthly.empty:
                monthly.to_excel(writer, sheet_name=f'{safe_name}_Monthly')
            
            # Quarterly returns
            quarterly = quarterly_returns(result.portfolio, value=result.value)
            if not quarterly.empty:
                quarterly.to_excel(writer, sheet_name=f'{safe_name}_Quarterly')

def generate_pdf_report(html_path: Path, pdf_path: Path) -> None:
    """Generate PDF from HTML report."""
    try:
        from weasyprint import HTML
        HTML(filename=str(html_path)).write_pdf(str(pdf_path))
        print(f"PDF report saved to {pdf_path}")
    except ImportError:
        print("WeasyPrint not installed. Install with: pip install weasyprint")
        print("PDF generation skipped.")
    except Exception as e:
        print(f"PDF generation failed: {e}")
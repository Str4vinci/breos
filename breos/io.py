"""
I/O module for data export and import.

This module provides functions for:
- Exporting simulation results to CSV/TXT
- Saving cost analysis reports
- Generating formatted summary reports
"""

from typing import Optional, Dict, Any, List, Union
from pathlib import Path
import os

import pandas as pd
import numpy as np


def export_results(
    results_df: pd.DataFrame,
    results_directory: str,
    prefix: str = "",
    suffix: str = "",
    format: str = 'csv',
    index: bool = False
) -> str:
    """
    Export simulation results to CSV or TXT.
    
    Args:
        results_df: DataFrame with simulation results
        results_directory: Directory to save the file
        prefix: Optional prefix for filename
        suffix: Optional suffix for filename
        format: Output format ('csv' or 'txt')
        index: Whether to include DataFrame index
        
    Returns:
        Path to the saved file
    """
    os.makedirs(results_directory, exist_ok=True)
    
    # Build filename
    parts = [p for p in [prefix, "results", suffix] if p]
    filename = "_".join(parts) + f".{format}"
    filepath = os.path.join(results_directory, filename)
    
    if format == 'csv':
        results_df.to_csv(filepath, index=index)
    elif format == 'txt':
        results_df.to_csv(filepath, index=index, sep='\t')
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'txt'.")
    
    return filepath


def export_cost_analysis(
    cost_df: pd.DataFrame,
    results_directory: str,
    prefix: str = "",
    suffix: str = "",
    format: str = 'csv',
    index: bool = False
) -> str:
    """
    Export cost projection analysis to CSV or TXT.
    
    Args:
        cost_df: DataFrame from cost_analysis_projection()
        results_directory: Directory to save the file
        prefix: Optional prefix for filename
        suffix: Optional suffix for filename
        format: Output format ('csv' or 'txt')
        index: Whether to include DataFrame index
        
    Returns:
        Path to the saved file
    """
    os.makedirs(results_directory, exist_ok=True)
    
    parts = [p for p in [prefix, "cost_analysis", suffix] if p]
    filename = "_".join(parts) + f".{format}"
    filepath = os.path.join(results_directory, filename)
    
    if format == 'csv':
        cost_df.to_csv(filepath, index=index)
    elif format == 'txt':
        cost_df.to_csv(filepath, index=index, sep='\t')
    else:
        raise ValueError(f"Unsupported format: {format}. Use 'csv' or 'txt'.")
    
    return filepath


def export_summary(
    summary_df: pd.DataFrame,
    results_directory: str,
    prefix: str = "",
    suffix: str = "",
    format: str = 'txt'
) -> str:
    """
    Export summary statistics as formatted text or CSV.
    
    Args:
        summary_df: Summary DataFrame (typically single row with key metrics)
        results_directory: Directory to save the file
        prefix: Optional prefix for filename
        suffix: Optional suffix for filename
        format: Output format ('txt' for formatted text, 'csv' for raw)
        
    Returns:
        Path to the saved file
    """
    os.makedirs(results_directory, exist_ok=True)
    
    parts = [p for p in [prefix, "summary", suffix] if p]
    filename = "_".join(parts) + f".{format}"
    filepath = os.path.join(results_directory, filename)
    
    if format == 'txt':
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SIMULATION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            for col in summary_df.columns:
                value = summary_df[col].iloc[0]
                if isinstance(value, float):
                    f.write(f"{col}: {value:.2f}\n")
                else:
                    f.write(f"{col}: {value}\n")
            
            f.write("\n" + "=" * 60 + "\n")
    else:
        summary_df.to_csv(filepath, index=False)
    
    return filepath


def save_simulation_report(
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    costs_dict: Optional[Dict[str, Any]] = None,
    cost_projection_df: Optional[pd.DataFrame] = None,
    degradation_df: Optional[pd.DataFrame] = None,
    results_directory: str = "results",
    scenario_name: str = ""
) -> Dict[str, str]:
    """
    Save complete simulation report with all outputs.
    
    Generates:
    - results_{scenario}.csv: Full simulation time series
    - summary_{scenario}.txt: Key metrics summary
    - costs_{scenario}.csv: Cost analysis (if provided)
    - degradation_{scenario}.csv: Battery degradation data (if provided)
    
    Args:
        results_df: Full simulation results DataFrame
        summary_df: Summary statistics DataFrame
        costs_dict: Optional cost parameters dictionary
        cost_projection_df: Optional cost projection DataFrame
        degradation_df: Optional degradation tracking DataFrame
        results_directory: Directory to save all files
        scenario_name: Scenario identifier for filenames
        
    Returns:
        Dictionary mapping file types to saved file paths
    """
    os.makedirs(results_directory, exist_ok=True)
    
    saved_files = {}
    suffix = scenario_name if scenario_name else ""
    
    # Save results
    saved_files['results'] = export_results(
        results_df, results_directory, suffix=suffix, format='csv'
    )
    
    # Save summary
    saved_files['summary'] = export_summary(
        summary_df, results_directory, suffix=suffix, format='txt'
    )
    
    # Save cost projection if provided
    if cost_projection_df is not None:
        saved_files['cost_projection'] = export_cost_analysis(
            cost_projection_df, results_directory, suffix=suffix, format='csv'
        )
    
    # Save degradation data if provided
    if degradation_df is not None:
        filepath = os.path.join(results_directory, f"degradation_{suffix}.csv" if suffix else "degradation.csv")
        degradation_df.to_csv(filepath, index=False)
        saved_files['degradation'] = filepath
    
    # Save costs dict as txt
    if costs_dict is not None:
        filepath = os.path.join(results_directory, f"costs_{suffix}.txt" if suffix else "costs.txt")
        with open(filepath, 'w') as f:
            f.write("COST PARAMETERS\n")
            f.write("=" * 40 + "\n")
            for key, value in costs_dict.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
        saved_files['costs'] = filepath
    
    return saved_files


def load_results(
    filepath: str,
    parse_dates: Union[bool, List[str]] = True
) -> pd.DataFrame:
    """
    Load simulation results from CSV or TXT file.
    
    Args:
        filepath: Path to the results file
        parse_dates: Whether to parse datetime columns (True, False, or list of column names)
        
    Returns:
        DataFrame with loaded results
    """
    if filepath.endswith('.txt'):
        df = pd.read_csv(filepath, sep='\t', parse_dates=parse_dates)
    else:
        df = pd.read_csv(filepath, parse_dates=parse_dates)
    
    # Try to set Datetime as index if present
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    
    return df


def export_monthly_summary(
    results_df: pd.DataFrame,
    results_directory: str,
    prefix: str = "",
    suffix: str = ""
) -> str:
    """
    Export monthly aggregated summary to CSV.
    
    Args:
        results_df: Full simulation results with Datetime index
        results_directory: Directory to save the file
        prefix: Optional prefix for filename
        suffix: Optional suffix for filename
        
    Returns:
        Path to the saved file
    """
    os.makedirs(results_directory, exist_ok=True)
    
    # Ensure datetime index
    if 'Datetime' in results_df.columns:
        df = results_df.set_index('Datetime')
    else:
        df = results_df.copy()
    
    df.index = pd.to_datetime(df.index)
    
    # Define aggregation rules
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    monthly = df[numeric_cols].resample('ME').sum()
    
    # Add month name
    monthly['Month'] = monthly.index.strftime('%B')
    
    parts = [p for p in [prefix, "monthly_summary", suffix] if p]
    filename = "_".join(parts) + ".csv"
    filepath = os.path.join(results_directory, filename)
    
    monthly.to_csv(filepath)
    return filepath


def export_yearly_summary(
    results_df: pd.DataFrame,
    results_directory: str,
    prefix: str = "",
    suffix: str = ""
) -> str:
    """
    Export yearly aggregated summary to CSV.
    
    Args:
        results_df: Full simulation results with Datetime index
        results_directory: Directory to save the file
        prefix: Optional prefix for filename
        suffix: Optional suffix for filename
        
    Returns:
        Path to the saved file
    """
    os.makedirs(results_directory, exist_ok=True)
    
    # Ensure datetime index
    if 'Datetime' in results_df.columns:
        df = results_df.set_index('Datetime')
    else:
        df = results_df.copy()
    
    df.index = pd.to_datetime(df.index)
    
    # Define aggregation rules
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    yearly = df[numeric_cols].resample('YE').sum()
    
    parts = [p for p in [prefix, "yearly_summary", suffix] if p]
    filename = "_".join(parts) + ".csv"
    filepath = os.path.join(results_directory, filename)
    
    yearly.to_csv(filepath)
    return filepath

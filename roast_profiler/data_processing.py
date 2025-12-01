#!/usr/bin/env python3
"""Advanced data processing for coffee roast profiles.

Focus: data cleaning (missing seconds, smoothing, noise reduction, outlier tagging),
       preparation for downstream descriptive / diagnostic analytics.

Usage (CLI):
    python data_processing.py path/to/roast.csv --outdir ./processed_out --savgol 11 3

Programmatic:
    from data_processing import process_roast
    df_clean, report = process_roast(path, smooth_method='savgol')

"""
from __future__ import annotations
import argparse
from pathlib import Path
import os
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import seaborn as sns

from input_data import load_csv, get_ror_column

sns.set(style="whitegrid")

@dataclass
class ProcessingReport:
    rows_original: int
    rows_filled: int
    seconds_missing_filled: int
    outlier_count_beans: int
    outlier_count_air: int
    smoothing_method: str
    ror_recomputed: bool

# ----------------------------- Core Processing Functions ----------------------------- #

def fill_missing_seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a continuous second-by-second index; interpolate missing rows.
    Assumes 'seconds' exists and is numeric.
    """
    df = df.sort_values('seconds').reset_index(drop=True)
    all_seconds = pd.RangeIndex(int(df['seconds'].min()), int(df['seconds'].max()) + 1)
    missing = set(all_seconds) - set(df['seconds'].astype(int))
    reindexed = df.set_index(df['seconds'].astype(int)).reindex(all_seconds)
    # Interpolate numeric columns
    numeric_cols = reindexed.select_dtypes(include=[np.number]).columns
    reindexed[numeric_cols] = reindexed[numeric_cols].interpolate(method='linear').ffill().bfill()
    # Forward fill non-numeric
    non_numeric = [c for c in reindexed.columns if c not in numeric_cols]
    for c in non_numeric:
        reindexed[c] = reindexed[c].ffill().bfill()
    reindexed['seconds'] = reindexed.index.astype(float)
    reindexed['__filled'] = (~reindexed.index.isin(df['seconds'].astype(int))).astype(int)
    return reindexed.reset_index(drop=True), len(missing)


def detect_outliers(series: pd.Series, z_thresh: float = 3.5, window: int = 21) -> pd.Series:
    """Return boolean mask of outliers using rolling median absolute deviation (MAD)."""
    if window < 5:
        window = 5
    roll_med = series.rolling(window, center=True, min_periods=3).median()
    diff = (series - roll_med).abs()
    mad = diff.rolling(window, center=True, min_periods=3).median()
    # Avoid division by zero
    mad_replaced = mad.replace(0, np.nan)
    z_score = diff / mad_replaced
    mask = (z_score > z_thresh) & z_score.notna()
    return mask.fillna(False)


def apply_outlier_treatment(df: pd.DataFrame, column: str, mask: pd.Series, method: str = 'interpolate') -> pd.Series:
    """Replace outliers according to method (interpolate | clip)."""
    treated = df[column].copy()
    if method == 'interpolate':
        treated[mask] = np.nan
        treated = treated.interpolate('linear').ffill().bfill()
    elif method == 'clip':
        # Clip to rolling median
        roll_med = df[column].rolling(11, center=True, min_periods=3).median()
        treated[mask] = roll_med[mask]
    else:
        raise ValueError("Unsupported outlier treatment method")
    return treated


def smooth_series(series: pd.Series, method: str = 'rolling', window: int = 11, polyorder: int = 3) -> pd.Series:
    """Smooth a signal using rolling mean or Savitzky-Golay."""
    if method == 'rolling':
        return series.rolling(window, center=True, min_periods=1).mean()
    elif method == 'savgol':
        # Ensure odd window and >= polyorder+2
        if window % 2 == 0:
            window += 1
        window = max(window, polyorder + 2 if (polyorder + 2) % 2 == 1 else polyorder + 3)
        filled = series.ffill().bfill()
        return pd.Series(savgol_filter(filled, window_length=window, polyorder=polyorder), index=series.index)
    else:
        raise ValueError("Unsupported smoothing method")


def compute_ror_if_missing(df: pd.DataFrame, temp_col: str = 'beans', span: int = 30) -> pd.DataFrame:
    """Compute a simple rate of rise (delta per minute) if no existing RoR column.
    span: smoothing span (seconds) for derivative.
    """
    ror_col = get_ror_column(df)
    if ror_col:
        return df, False
    # derivative: dT/dt * 60 (to per minute)
    temp = df[temp_col].rolling(span, center=True, min_periods=5).mean()
    dt = df['seconds'].diff().replace(0, np.nan)
    dtemp = temp.diff()
    ror = (dtemp / dt) * 60
    df['ror'] = ror.rolling(11, center=True, min_periods=3).mean()
    return df, True

# ----------------------------- Pipeline ----------------------------- #

def process_roast(path: str | Path,
                  smooth_method: str = 'savgol',
                  smooth_window: int = 11,
                  savgol_poly: int = 3,
                  outlier_z: float = 3.5,
                  outlier_method: str = 'interpolate'):
    """Full processing pipeline returning cleaned DataFrame and report."""
    df = load_csv(path)
    rows_original = len(df)

    # Fill missing seconds
    df_filled, missing_count = fill_missing_seconds(df)

    # Outlier detection on beans & air
    beans_mask = detect_outliers(df_filled['beans'], z_thresh=outlier_z)
    air_mask = detect_outliers(df_filled['air'], z_thresh=outlier_z) if 'air' in df_filled.columns else pd.Series([False]*len(df_filled))

    df_filled['beans_clean'] = apply_outlier_treatment(df_filled, 'beans', beans_mask, method=outlier_method)
    if 'air' in df_filled.columns:
        df_filled['air_clean'] = apply_outlier_treatment(df_filled, 'air', air_mask, method=outlier_method)
    else:
        df_filled['air_clean'] = np.nan

    # Smoothing
    df_filled['beans_smooth'] = smooth_series(df_filled['beans_clean'], method=smooth_method, window=smooth_window, polyorder=savgol_poly)
    if 'air_clean' in df_filled.columns:
        df_filled['air_smooth'] = smooth_series(df_filled['air_clean'], method=smooth_method, window=smooth_window, polyorder=savgol_poly)

    # Recompute RoR if missing
    df_final, ror_recomputed = compute_ror_if_missing(df_filled)
    report = ProcessingReport(
        rows_original=rows_original,
        rows_filled=len(df_final),
        seconds_missing_filled=missing_count,
        outlier_count_beans=int(beans_mask.sum()),
        outlier_count_air=int(air_mask.sum()),
        smoothing_method=smooth_method,
        ror_recomputed=ror_recomputed,
    )
    return df_final, report, beans_mask, air_mask

# ----------------------------- Visualization ----------------------------- #

def plot_processing(df: pd.DataFrame, beans_mask: pd.Series, air_mask: pd.Series, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    t = df['seconds']

    # Temperature cleaning
    fig, ax = plt.subplots(figsize=(11,5))
    ax.plot(t, df['beans'], color='red', alpha=0.25, label='Beans raw')
    ax.plot(t, df['beans_clean'], color='red', linewidth=1, label='Beans clean')
    ax.plot(t, df['beans_smooth'], color='darkred', linewidth=1.5, label='Beans smooth')
    if 'air' in df.columns:
        ax.plot(t, df['air'], color='blue', alpha=0.15, label='Air raw')
        ax.plot(t, df['air_clean'], color='blue', linewidth=1, label='Air clean')
        ax.plot(t, df['air_smooth'], color='navy', linewidth=1.5, label='Air smooth')
    # Mark outliers
    # Scatter outliers (filter both x and y so lengths match)
    ax.scatter(t[beans_mask], df['beans'][beans_mask], color='orange', s=15, label='Beans outliers')
    if 'air' in df.columns:
        ax.scatter(t[air_mask], df['air'][air_mask], color='cyan', s=15, label='Air outliers')
    ax.set_xlabel('Seconds')
    ax.set_ylabel('Temp (°C)')
    ax.set_title('Temperature Cleaning & Smoothing')
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / 'cleaning_temperature.png', dpi=150)
    plt.close(fig)

    # RoR
    if 'ror' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(11,4))
        ax2.plot(t, df['ror'], color='green', linewidth=1, label='RoR')
        ax2.set_xlabel('Seconds')
        ax2.set_ylabel('RoR (°C/min)')
        ax2.set_title('Rate of Rise')
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(outdir / 'ror.png', dpi=150)
        plt.close(fig2)

# ----------------------------- CLI ----------------------------- #

def main():
    ap = argparse.ArgumentParser(description='Advanced roast data processing')
    ap.add_argument('csv', help='Path to roast CSV')
    ap.add_argument('--outdir', default=None, help='Output directory (default: <csv_dir>/processed_out)')
    ap.add_argument('--method', choices=['rolling','savgol'], default='savgol', help='Smoothing method')
    ap.add_argument('--window', type=int, default=11, help='Smoothing window')
    ap.add_argument('--poly', type=int, default=3, help='Savgol polyorder (if method=savgol)')
    ap.add_argument('--zthresh', type=float, default=3.5, help='Outlier Z/MAD threshold')
    ap.add_argument('--outlier-treatment', choices=['interpolate','clip'], default='interpolate', help='Outlier treatment strategy')
    args = ap.parse_args()

    csv_path = Path(args.csv)
    outdir = Path(args.outdir) if args.outdir else csv_path.parent / 'processed_out'
    outdir.mkdir(parents=True, exist_ok=True)

    df_clean, report, beans_mask, air_mask = process_roast(
        csv_path,
        smooth_method=args.method,
        smooth_window=args.window,
        savgol_poly=args.poly,
        outlier_z=args.zthresh,
        outlier_method=args.outlier_treatment,
    )

    # Save cleaned CSV
    cleaned_path = outdir / 'roast_cleaned.csv'
    df_clean.to_csv(cleaned_path, index=False)

    # Visualization
    plot_processing(df_clean, beans_mask, air_mask, outdir)

    # Report
    report_path = outdir / 'processing_report.txt'
    with open(report_path, 'w') as f:
        for k, v in report.__dict__.items():
            f.write(f'{k}: {v}\n')

    print('Processing complete:')
    for k, v in report.__dict__.items():
        print(f' - {k}: {v}')
    print('Saved cleaned CSV:', cleaned_path)
    print('Saved report:', report_path)
    print('Saved plots in:', outdir)

if __name__ == '__main__':
    main()

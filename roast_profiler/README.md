roast_profiler

Simple roast profile analysis tools for coffee roasting data.

Installation

```bash
pip install -r roast_profiler/requirements.txt
```

Modules

### input_data.py

Utilities for loading and validating roast CSV data. Key functions:

- `load_csv(path)` — Loads CSV, normalizes columns, converts `time` → `seconds`
- `get_ror_column(df)` — Detects RoR column (`ror` > `ror_f` > `computed_air_ror`)
- `validate_roast_data(df)` — Validates structure, data ranges, and reports issues

**Usage:**
```bash
python roast_profiler/input_data.py path/to/roast.csv
```

**In code:**
```python
from roast_profiler.input_data import load_csv, validate_roast_data

df = load_csv('roast.csv')
validation = validate_roast_data(df)
if validation['valid']:
    print(f"Loaded {validation['row_count']} rows")
```

### analyze_roast.py

Full roast analyzer: loads data, computes stats, smooths traces, detects peaks, saves plots.

**Usage:**
```bash
python roast_profiler/analyze_roast.py path/to/roast.csv
```

**Options:**
```bash
python roast_profiler/analyze_roast.py roast.csv \
  --outdir ./my_outputs \
  --smooth 7 \
  --ror-threshold 5.0
```

**Outputs:**
- `temperature_profile.png` — Bean & air temps (raw + smoothed)
- `ror_profile.png` — RoR trace with peak detection
- `summary.txt` — Roast stats (times, temps, peaks)

Features

- Handles common roaster CSV formats (multiple RoR column names)
- Auto-normalizes column names and converts time → seconds
- Validates data quality and reports issues
- Smooths temperature and RoR traces (configurable window)
- Detects RoR peaks and marks thresholds
- Generates publication-ready plots

Example Roast CSV Columns (required)

- `seconds` or `time` (HH:MM:SS format)
- `beans` (bean temperature in °C)
- `air` (air temperature in °C)
- `ror` or `ror_f` or `computed_air_ror` (rate of rise)

### data_processing.py

Advanced cleaning and preparation pipeline.

Features:
- Fills missing seconds (interpolation) and flags filled rows (`__filled`).
- Detects temperature outliers via rolling MAD (configurable threshold) and treats them (interpolate or clip).
- Multiple smoothing methods: rolling mean or Savitzky–Golay.
- Recomputes RoR automatically if absent.
- Generates side-by-side plots (raw vs cleaned vs smoothed, outlier markers, RoR).

**Usage:**
```bash
python roast_profiler/data_processing.py path/to/roast.csv \
  --method savgol --window 15 --poly 3 --zthresh 3.5 --outlier-treatment interpolate
```

**In code:**
```python
from roast_profiler.data_processing import process_roast
df_clean, report, beans_mask, air_mask = process_roast('roast.csv')
print(report)
```

Outputs saved to `processed_out/` (unless overridden):
- `roast_cleaned.csv` — cleaned + smoothed columns (`beans_clean`, `beans_smooth`, etc.)
- `processing_report.txt` — summary of operations
- `cleaning_temperature.png`, `ror.png` — visualization artifacts




LEARNING_TASKS.md - 13 progressive tasks across 4 skill levels:

🟢 Beginner: data loading, missing data, type conversions
🟡 Intermediate: statistics, groupby, visualization
🟠 Advanced: time series, customer behavior, anomaly detection
🔴 Expert: forecasting, recommendations, dashboards


tasks/ - Starter templates with function stubs and docstrings:

task_1_1_data_exploration.py
task_1_2_missing_data.py
task_2_1_statistics.py
task_3_1_time_series.py
task_4_1_forecasting.py
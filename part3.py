import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Load Data
df = pd.read_csv('part3.csv')
# Buat kolom Date untuk plotting time series
df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str) + '-01')

# ==========================================
# TASK 1: STATISTIK DESKRIPTIF & PLOTTING
# ==========================================
def compute_stats(data):
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    return {
        'Count': n,
        'Min': np.min(data),
        'Max': np.max(data),
        'Range': np.max(data) - np.min(data),
        'Mean': mean,
        'Median': np.median(data),
        'Std Dev': std,
        'Variance': np.var(data, ddof=1),
        'CV (%)': (std / mean) * 100 if mean != 0 else np.nan,
        'Q1': q1,
        'Q3': q3,
        'IQR': iqr,
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data)
    }

stats_raw = compute_stats(df['Sea Level'])
print("--- TASK 1: Raw Data Statistics ---")
for k, v in stats_raw.items():
    print(f"{k}: {v:.4f}")

# Visualisasi Task 1
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Time Series Plot
axes[0].plot(df['Date'], df['Sea Level'], marker='o', color='tab:blue', linestyle='-')
axes[0].set_title('Time Series Sea Level (Raw)')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Sea Level')
axes[0].grid(True)

# Histogram
axes[1].hist(df['Sea Level'], bins=20, color='tab:blue', edgecolor='black', alpha=0.7)
axes[1].axvline(stats_raw['Mean'], color='red', linestyle='--', label=f"Mean ({stats_raw['Mean']:.1f})")
axes[1].axvline(stats_raw['Median'], color='orange', linestyle='-', label=f"Median ({stats_raw['Median']:.1f})")
axes[1].set_title('Histogram Distribution (Raw)')
axes[1].legend()

# Boxplot
axes[2].boxplot(df['Sea Level'], vert=True)
axes[2].set_title('Boxplot (Raw)')
plt.tight_layout()
plt.show()

# ===

# ===

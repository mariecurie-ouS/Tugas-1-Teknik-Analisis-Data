import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Data
df = pd.read_csv('part3.csv')
sea_level = df['Sea Level'].values
x = np.arange(len(sea_level))

# ==========================================
# METODE 1: MOVING MEDIAN (Window = 5)
# ==========================================
window_size = 5
moving_med = pd.Series(sea_level).rolling(window=window_size, center=True, min_periods=1).median().values
diff_mm = np.abs(sea_level - moving_med)
# Outlier jika selisih dari median bergerak > 500 mm
outliers_mm = diff_mm > 500

# ==========================================
# METODE 2: LEAST SQUARES (Garis Tren Linear)
# ==========================================
p = np.polyfit(x, sea_level, 1)
y_pred = np.polyval(p, x)
residuals_ls = sea_level - y_pred
z_scores_ls = np.abs(residuals_ls / np.std(residuals_ls))
# Outlier jika nilai mutlak Z-score sisaan > 3
outliers_ls = z_scores_ls > 3

# ==========================================
# PLOTTING
# ==========================================
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Plot 1: Moving Median
axes[0].plot(x, sea_level, label='Data Asli (Raw Data)', color='gray', alpha=0.5, linestyle='--')
axes[0].plot(x, moving_med, label='Moving Median (Window=5)', color='blue', linewidth=2)
axes[0].scatter(x[outliers_mm], sea_level[outliers_mm], color='red', label=f'Outlier Terdeteksi ({outliers_mm.sum()} titik)', zorder=5)
axes[0].set_title('Deteksi Outlier Menggunakan Moving Median', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Sea Level')
axes[0].legend(loc='upper right')
axes[0].grid(True, linestyle=':', alpha=0.6)

# Plot 2: Least Squares
axes[1].plot(x, sea_level, label='Data Asli (Raw Data)', color='gray', alpha=0.5, linestyle='--')
axes[1].plot(x, y_pred, label=f'Garis Tren Least Squares (y = {p[0]:.2f}x + {p[1]:.2f})', color='green', linewidth=2)
axes[1].scatter(x[outliers_ls], sea_level[outliers_ls], color='red', label=f'Outlier Terdeteksi |Z| > 3 ({outliers_ls.sum()} titik)', zorder=5)
axes[1].set_title('Deteksi Outlier Menggunakan Least Squares Residuals (|Z| > 3)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Indeks Data (Bulan ke-)')
axes[1].set_ylabel('Sea Level')
axes[1].legend(loc='upper right')
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.show()

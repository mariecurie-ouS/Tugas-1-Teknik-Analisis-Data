import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Config tampilan grafik
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = 'Arial'

# 1. Load Data
df = pd.read_csv('part1.csv')
data = df['Ellipsoidal_height_m']

# 2. Kalkulasi Descriptive Statistics
n_obs = len(data)
min_val = data.min()
max_val = data.max()
range_val = max_val - min_val
mean_val = data.mean()
median_val = data.median()
std_val = data.std()               # Sample standard deviation
var_val = data.var()               # Sample variance
cv_val = (std_val / mean_val) * 100 # Coefficient of Variation (%)
q1 = data.quantile(0.25)
q3 = data.quantile(0.75)
iqr_val = q3 - q1

# Tampilkan Hasil Statistik
print("=== STATISTICAL ANALYSIS (PART 1 - TASK 1) ===")
print(f"Number of observations : {n_obs}")
print(f"Minimum                : {min_val:.4f} m")
print(f"Maximum                : {max_val:.4f} m")
print(f"Range                  : {range_val:.4f} m")
print(f"Arithmetic Mean        : {mean_val:.4f} m")
print(f"Median                 : {median_val:.4f} m")
print(f"Standard Deviation     : {std_val:.4f} m")
print(f"Variance               : {var_val:.6f} m²")
print(f"Coeff. of Variation    : {cv_val:.4f} %")
print(f"First Quartile (Q1)    : {q1:.4f} m")
print(f"Third Quartile (Q3)    : {q3:.4f} m")
print(f"Interquartile Range    : {iqr_val:.4f} m")
print("==============================================")

# 3. Visualisasi (Histogram & Boxplot)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Histogram dengan Fit Kurva Normal
sns.histplot(data, stat="density", bins=20, color='skyblue', edgecolor='black', ax=axes[0])
mu, std = stats.norm.fit(data)
xmin, xmax = axes[0].get_xlim()
x = np.linspace(xmin, xmax, 100)
p = stats.norm.pdf(x, mu, std)
axes[0].plot(x, p, 'r-', linewidth=2, label='Fit Kurva Normal')
axes[0].axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean ({mean_val:.4f})')
axes[0].axvline(median_val, color='green', linestyle='-', linewidth=1.5, label=f'Median ({median_val:.4f})')
axes[0].set_title('Histogram of Ellipsoidal Height', fontweight='bold')
axes[0].set_xlabel('Ellipsoidal Height (m)')
axes[0].set_ylabel('Density')
axes[0].legend()

# Plot 2: Boxplot
sns.boxplot(y=data, ax=axes[1], color='lightgreen', flierprops={"markerfacecolor": "red", "marker": "o"})
axes[1].set_title('Boxplot of Ellipsoidal Height', fontweight='bold')
axes[1].set_ylabel('Ellipsoidal Height (m)')

plt.tight_layout()
plt.show()

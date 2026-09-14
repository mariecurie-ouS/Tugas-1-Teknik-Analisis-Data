import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = pd.read_csv("part1.csv")
y_orig = data['Ellipsoidal_height_m'].dropna().values

# 2. Identify and remove outliers using the 1.5 * IQR rule (or |Z| > 3 rule)
q1_orig = np.percentile(y_orig, 25)
q3_orig = np.percentile(y_orig, 75)
iqr_orig = q3_orig - q1_orig

lower_bound = q1_orig - 1.5 * iqr_orig
upper_bound = q3_orig + 1.5 * iqr_orig

# Filtered data (without outliers)
y_clean = y_orig[(y_orig >= lower_bound) & (y_orig <= upper_bound)]

# 3. Calculate statistics BEFORE and AFTER
def get_stats(data_array):
    return {
        'n': len(data_array),
        'mean': np.mean(data_array),
        'median': np.median(data_array),
        'std_dev': np.std(data_array, ddof=1),
        'variance': np.var(data_array, ddof=1),
        'range': np.max(data_array) - np.min(data_array)
    }

stats_before = get_stats(y_orig)
stats_after = get_stats(y_clean)

# Print Comparison Table
df_comp = pd.DataFrame([stats_before, stats_after], index=['Before Outlier Removal', 'After Outlier Removal'])
df_comp['mean_shift'] = df_comp['mean'] - stats_before['mean']
df_comp['median_shift'] = df_comp['median'] - stats_before['median']

print("=== TASK 3: COMPARISON TABLE ===")
print(df_comp[['n', 'mean', 'median', 'std_dev', 'range']])
print("\n--- Absolute Changes ---")
print(f"Mean Change   : {stats_after['mean'] - stats_before['mean']:.6f} m")
print(f"Median Change : {stats_after['median'] - stats_before['median']:.6f} m")
print(f"Std Dev Change: {stats_after['std_dev'] - stats_before['std_dev']:.6f} m")

# 4. Side-by-Side Visualization
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before Removal
sns.boxplot(y=y_orig, ax=axes[0], color='lightgreen', flierprops=dict(marker='o', markerfacecolor='red'))
axes[0].set_title(f"Before Removal (n={stats_before['n']})\nMean: {stats_before['mean']:.4f} | Median: {stats_before['median']:.4f}", fontweight='bold')
axes[0].set_ylabel("Ellipsoidal Height (m)")

# After Removal
sns.boxplot(y=y_clean, ax=axes[1], color='skyblue', flierprops=dict(marker='o', markerfacecolor='red'))
axes[1].set_title(f"After Removal (n={stats_after['n']})\nMean: {stats_after['mean']:.4f} | Median: {stats_after['median']:.4f}", fontweight='bold')
axes[1].set_ylabel("Ellipsoidal Height (m)")

plt.tight_layout()
plt.show()

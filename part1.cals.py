import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# 1. Load the dataset
data = pd.read_csv("part1.csv")
# Replace 'height' with the exact column name in your CSV
y = data['Ellipsoidal_height_m'].dropna().values 

# 2. Perform Calculations
n = len(y)
y_min = np.min(y)
y_max = np.max(y)
y_range = y_max - y_min
mean = np.mean(y)
median = np.median(y)
std_dev = np.std(y, ddof=1)  # Sample standard deviation
variance = np.var(y, ddof=1) # Sample variance
cv = (std_dev / mean) * 100  # Coefficient of variation (%)

# Quartiles and IQR
q1 = np.percentile(y, 25)
q3 = np.percentile(y, 75)
iqr = q3 - q1

# Print Summary Table
print("=== STATISTICAL SUMMARY ===")
print(f"Number of observations (n) : {n}")
print(f"Minimum                    : {y_min:.4f}")
print(f"Maximum                    : {y_max:.4f}")
print(f"Range                      : {y_range:.4f}")
print(f"Arithmetic Mean            : {mean:.4f}")
print(f"Median                     : {median:.4f}")
print(f"Standard Deviation         : {std_dev:.4f}")
print(f"Variance                   : {variance:.4f}")
print(f"Coefficient of Variation (%): {cv:.4f}%")
print(f"First Quartile (Q1)        : {q1:.4f}")
print(f"Third Quartile (Q3)        : {q3:.4f}")
print(f"Interquartile Range (IQR)  : {iqr:.4f}")

# 3. Visualization (Styled to match Figure 2)
sns.set_theme(style="whitegrid")

plt.figure(figsize=(14, 5))

# Subplot 1: Histogram with Normal Fit, Mean, and Median
plt.subplot(1, 2, 1)

count, bins, _ = plt.hist(
    y, 
    bins=20, 
    density=True, 
    color='skyblue', 
    edgecolor='black', 
    alpha=0.8
)

xmin, xmax = plt.xlim()
x_pdf = np.linspace(xmin, xmax, 200)
p_pdf = stats.norm.pdf(x_pdf, mean, std_dev)
plt.plot(x_pdf, p_pdf, color='red', linewidth=2, label='Fit Kurva Normal')

# Vertical dashed line for Mean
plt.axvline(
    mean, 
    color='red', 
    linestyle='--', 
    linewidth=1.5, 
    label=f'Mean ({mean:.4f})'
)

# Vertical solid line for Median
plt.axvline(
    median, 
    color='green', 
    linestyle='-', 
    linewidth=1.5, 
    label=f'Median ({median:.4f})'
)

plt.title('Histogram of Ellipsoidal Height', fontweight='bold')
plt.xlabel('Ellipsoidal Height (m)')
plt.ylabel('Density')
plt.legend(frameon=True, facecolor='white', framealpha=0.8)

# Subplot 2: Boxplot with Light Green Fill & Highlighted Outliers
plt.subplot(1, 2, 2)

flierprops = dict(
    marker='o', 
    markerfacecolor='red', 
    markeredgecolor='dimgray', 
    markersize=6
)

plt.boxplot(
    y, 
    vert=True, 
    patch_artist=True, 
    widths=0.8,
    flierprops=flierprops,
    boxprops=dict(facecolor='lightgreen', color='dimgray'),
    whiskerprops=dict(color='dimgray'),
    capprops=dict(color='dimgray'),
    medianprops=dict(color='dimgray')
)

plt.title('Boxplot of Ellipsoidal Height', fontweight='bold')
plt.ylabel('Ellipsoidal Height (m)')
plt.xticks([])  # Hide x-axis tick label "1"

plt.tight_layout()
plt.show()

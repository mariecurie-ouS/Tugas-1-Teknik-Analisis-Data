import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
data = pd.read_csv("part1.csv")
y = data['Ellipsoidal_height_m'].dropna().values

n = len(y)
mean = np.mean(y)
s = np.std(y, ddof=1) # Sample standard deviation

# 2. Calculate 95% Confidence Interval for the Mean
t_score = stats.t.ppf(0.975, df=n-1)
margin_of_error = t_score * (s / np.sqrt(n))

ci_lower = mean - margin_of_error
ci_upper = mean + margin_of_error

# 3. Calculate 95% Individual Prediction Limits (Expected Range for Individual Points)
# For individual points: Mean ± 1.96 * s (or t * s * sqrt(1 + 1/n))
pred_lower = mean - 1.96 * s
pred_upper = mean + 1.96 * s

# Identify observations outside expected limits
outliers_ci = y[(y < ci_lower) | (y > ci_upper)]
outliers_pred = y[(y < pred_lower) | (y > pred_upper)]

# Print Results
print("=== TASK 2 CALCULATIONS ===")
print(f"Sample Mean (x̄)          : {mean:.4f}")
print(f"Sample Std Dev (s)        : {s:.4f}")
print(f"Critical t-value (df={n-1}): {t_score:.4f}")
print(f"95% CI for Mean           : [{ci_lower:.4f}, {ci_upper:.4f}]")
print(f"Observations outside CI   : {len(outliers_ci)} out of {n}")
print(f"95% Individual Limit      : [{pred_lower:.4f}, {pred_upper:.4f}]")
print(f"Observations outside 1.96s: {len(outliers_pred)} out of {n}")

# 4. Visualization
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))

# Plot all observations against observation index
indices = np.arange(1, n + 1)
plt.scatter(indices, y, color='dodgerblue', alpha=0.7, edgecolors='b', label='Observations')

# Draw Mean & 95% CI Band
plt.axhline(mean, color='red', linestyle='-', label=f'Mean ({mean:.4f})')
plt.axhline(ci_lower, color='orange', linestyle='--', label=f'95% CI Lower ({ci_lower:.4f})')
plt.axhline(ci_upper, color='orange', linestyle='--', label=f'95% CI Upper ({ci_upper:.4f})')

# Draw Individual Prediction Band (Mean ± 1.96s)
plt.axhline(pred_lower, color='purple', linestyle=':', label=f'Lower Limit (Mean - 1.96s: {pred_lower:.4f})')
plt.axhline(pred_upper, color='purple', linestyle=':', label=f'Upper Limit (Mean + 1.96s: {pred_upper:.4f})')

# Highlight points outside 1.96s
outlier_mask = (y < pred_lower) | (y > pred_upper)
plt.scatter(indices[outlier_mask], y[outlier_mask], color='red', s=60, label='Potential Outliers (>1.96s)')

plt.title('Task 2: Observation Consistency vs. 95% CI & Prediction Limits', fontweight='bold')
plt.xlabel('Observation Index')
plt.ylabel('Ellipsoidal Height (m)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

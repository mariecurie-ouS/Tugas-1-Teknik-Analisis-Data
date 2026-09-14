import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. Load data
df = pd.read_csv("Part2.csv")

# Ensure date parsing and sort by date
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)
series = df["Vertical_displacement_mm"]

# Compute decimal time in years for least-squares model
df["Time_Years"] = (df["Date"] - df["Date"].min()).dt.days / 365.25

# ===========================================================
# 2. OUTLIER DETECTION METHOD 1: MOVING MEDIAN
# ===========================================================
window_size = 7  # 7-month sliding window
df["Moving_Median"] = series.rolling(window=window_size, center=True, min_periods=3).median()

# Calculate Median Absolute Deviation (MAD) for robust thresholding
median_dev = np.abs(series - df["Moving_Median"])
mad = median_dev.median()
mad_threshold = 3 * 1.4826 * mad

# Flag points where deviation > 3 * MAD (scaled by 1.4826 for normal consistency)
df["Outlier_Moving_Median"] = median_dev > mad_threshold

# ===========================================================
# 3. OUTLIER DETECTION METHOD 2: LEAST-SQUARES METHOD
# ===========================================================
# Parametric Model: Linear Trend + Annual (1-yr) + Semi-Annual (0.5-yr) Harmonics
def gnss_model(t, a, b, c, d, e, f):
    return (
        a * t
        + b
        + c * np.sin(2 * np.pi * t)
        + d * np.cos(2 * np.pi * t)
        + e * np.sin(4 * np.pi * t)
        + f * np.cos(4 * np.pi * t)
    )

# Fit model parameters using Least Squares Estimation
popt, _ = curve_fit(gnss_model, df["Time_Years"], series)
df["Least_Squares_Fit"] = gnss_model(df["Time_Years"], *popt)

# Residual Analysis & Z-Score Calculations
df["Residuals"] = series - df["Least_Squares_Fit"]
res_mean = df["Residuals"].mean()
res_std = df["Residuals"].std()

# Explicit numerical z-score calculation
df["Residual_Z_Score"] = (df["Residuals"] - res_mean) / res_std

# Flag residuals exceeding 3 standard deviations (|z| > 3)
df["Outlier_Residual"] = np.abs(df["Residual_Z_Score"]) > 3

# Combined flag (detected by either method)
df["Final_Outlier"] = df["Outlier_Moving_Median"] | df["Outlier_Residual"]

# ===========================================================
# 4. NUMERICAL PRINT OUTS TO TERMINAL
# ===========================================================
print("=" * 70)
print("STATISTICAL THRESHOLDS FOR OUTLIER DETECTION")
print("=" * 70)
print(f"Moving Median Window Size  : {window_size} points")
print(f"Median Absolute Dev (MAD)  : {mad:.4f} mm")
print(f"Moving Median Cutoff (3*MAD): {mad_threshold:.4f} mm")
print(f"Residual Mean              : {res_mean:.4f} mm")
print(f"Residual Standard Dev (σ)  : {res_std:.4f} mm")
print(f"Upper 3σ Threshold         : +{3 * res_std:.4f} mm")
print(f"Lower 3σ Threshold         : -{3 * res_std:.4f} mm")
print("=" * 70)

print("\n" + "=" * 70)
print("LEAST-SQUARES RESIDUAL OUTLIERS (|z| > 3)")
print("=" * 70)
ls_table = df[df["Outlier_Residual"]][
    ["Date", "Vertical_displacement_mm", "Least_Squares_Fit", "Residuals", "Residual_Z_Score"]
].copy()
ls_table["Date"] = ls_table["Date"].dt.strftime("%Y-%m-%d")
print(ls_table.to_string(index=False))

print("\n" + "=" * 70)
print("MOVING MEDIAN OUTLIERS")
print("=" * 70)
mm_table = df[df["Outlier_Moving_Median"]][
    ["Date", "Vertical_displacement_mm", "Moving_Median"]
].copy()
mm_table["Deviation"] = np.abs(mm_table["Vertical_displacement_mm"] - mm_table["Moving_Median"])
mm_table["Date"] = mm_table["Date"].dt.strftime("%Y-%m-%d")
print(mm_table.to_string(index=False))

# ===========================================================
# 5. PRODUCE PLOTS WITH ANNOTATED NUMERICAL VALUES
# ===========================================================
fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True)

# Panel 1: Moving Median Detection
axes[0].plot(df["Date"], series, color="navy", alpha=0.5, label="Observed Series")
axes[0].plot(df["Date"], df["Moving_Median"], color="darkorange", linewidth=2, label="Moving Median (7-pt)")
mm_outliers = df[df["Outlier_Moving_Median"]]
axes[0].scatter(
    mm_outliers["Date"],
    mm_outliers["Vertical_displacement_mm"],
    color="red",
    facecolors="none",
    edgecolors="red",
    s=80,
    linewidth=2,
    zorder=5,
    label=f"Moving Median Outliers (n={len(mm_outliers)})",
)
axes[0].set_title(f"1. Moving Median Local Anomaly Detection (Threshold > {mad_threshold:.2f} mm)")
axes[0].set_ylabel("Displacement (mm)")
axes[0].grid(True, linestyle="--", alpha=0.6)
axes[0].legend(loc="upper right")

# Panel 2: Least-Squares Fit
axes[1].plot(df["Date"], series, color="navy", alpha=0.5, label="Observed Series")
axes[1].plot(df["Date"], df["Least_Squares_Fit"], color="green", linewidth=2, label="Least-Squares Model Fit")
res_outliers = df[df["Outlier_Residual"]]
axes[1].scatter(
    res_outliers["Date"],
    res_outliers["Vertical_displacement_mm"],
    color="red",
    marker="x",
    s=100,
    linewidth=2,
    zorder=5,
    label=f"LS Residual Outliers (|z| > 3σ, n={len(res_outliers)})",
)

# Annotate numerical value next to each LS outlier point
for idx, row in res_outliers.iterrows():
    axes[1].annotate(
        f"{row['Vertical_displacement_mm']:.2f} mm",
        (row["Date"], row["Vertical_displacement_mm"]),
        textcoords="offset points",
        xytext=(0, 10 if row["Vertical_displacement_mm"] > -20 else -15),
        ha="center",
        fontsize=9,
        fontweight="bold",
        color="darkred",
    )

axes[1].set_title(f"2. Least-Squares Parametric Fitting (Residual σ = {res_std:.2f} mm)")
axes[1].set_ylabel("Displacement (mm)")
axes[1].grid(True, linestyle="--", alpha=0.6)
axes[1].legend(loc="upper right")

# Panel 3: Detrended Residuals
axes[2].plot(df["Date"], df["Residuals"], color="purple", alpha=0.7, marker=".", label="Model Residuals")
axes[2].axhline(3 * res_std, color="red", linestyle="--", label=f"+3σ (+{3*res_std:.2f} mm)")
axes[2].axhline(-3 * res_std, color="red", linestyle="--", label=f"-3σ ({-3*res_std:.2f} mm)")
axes[2].axhline(0, color="black", linewidth=0.8)

final_outliers = df[df["Final_Outlier"]]
axes[2].scatter(
    final_outliers["Date"],
    final_outliers["Residuals"],
    color="red",
    marker="D",
    s=60,
    zorder=5,
    label=f"Flagged Residuals (n={len(final_outliers)})",
)

# Annotate z-score numbers on the residual panel
for idx, row in res_outliers.iterrows():
    axes[2].annotate(
        f"z={row['Residual_Z_Score']:.2f}\n({row['Residuals']:.1f}mm)",
        (row["Date"], row["Residuals"]),
        textcoords="offset points",
        xytext=(0, 10 if row["Residuals"] > 0 else -20),
        ha="center",
        fontsize=8,
        fontweight="bold",
        color="crimson",
    )

axes[2].set_title("3. Detrended Residual Thresholding (|z| > 3)")
axes[2].set_xlabel("Date")
axes[2].set_ylabel("Residual (mm)")
axes[2].grid(True, linestyle="--", alpha=0.6)
axes[2].legend(loc="upper right")

plt.tight_layout()
plt.show()

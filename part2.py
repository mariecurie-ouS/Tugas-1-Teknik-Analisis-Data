import matplotlib.pyplot as plt
import pandas as pd

# 1. Load data
df = pd.read_csv("Part2.csv")

# Ensure date parsing
df["Date"] = pd.to_datetime(df["Date"])
series = df["Vertical_displacement_mm"]

# 2. Calculate Statistical Metrics
q1 = series.quantile(0.25)
q3 = series.quantile(0.75)

stats = {
    "Number of Observations": series.count(),
    "Minimum": series.min(),
    "Maximum": series.max(),
    "Mean": series.mean(),
    "Median": series.median(),
    "Standard Deviation": series.std(),
    "Variance": series.var(),
    "Range": series.max() - series.min(),
    "IQR": q3 - q1,
    "Skewness": series.skew(),
    "Kurtosis": series.kurtosis(),  # Sample excess kurtosis
}

# Display results
stats_df = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
print(stats_df)

# 3. Produce Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Time-Series Plot
axes[0].plot(
    df["Date"],
    series,
    color="navy",
    linestyle="-",
    marker="o",
    markersize=3,
    linewidth=1,
)
axes[0].set_title("GNSS Vertical Displacement Time Series")
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Vertical Displacement (mm)")
axes[0].grid(True, linestyle="--", alpha=0.6)

# Histogram Plot
axes[1].hist(
    series, bins=15, color="skyblue", edgecolor="black", density=True, alpha=0.7
)
series.plot(kind="kde", ax=axes[1], color="red", linewidth=2)  # KDE curve overlay
axes[1].set_title("Histogram & Density Distribution")
axes[1].set_xlabel("Vertical Displacement (mm)")
axes[1].set_ylabel("Density")
axes[1].grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()

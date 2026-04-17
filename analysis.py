import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.15)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
PALETTE = sns.color_palette("tab20")

# ── 1. Load & clean ────────────────────────────────────────────────────────────

df = pd.read_csv("Quality of Service 5G.csv")

df["Signal_Strength"] = df["Signal_Strength"].str.replace(" dBm", "", regex=False).astype(float)
df["Latency"] = df["Latency"].str.replace(" ms", "", regex=False).astype(float)
df["Resource_Allocation"] = df["Resource_Allocation"].str.replace("%", "", regex=False).astype(float)

def _to_mbps(val: str) -> float:
    if "Kbps" in val:
        return float(val.replace(" Kbps", "")) / 1000
    return float(val.replace(" Mbps", ""))

df["Required_Bandwidth_Mbps"] = df["Required_Bandwidth"].apply(_to_mbps)
df["Allocated_Bandwidth_Mbps"] = df["Allocated_Bandwidth"].apply(_to_mbps)

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df["Hour"] = df["Timestamp"].dt.hour
df["Minute"] = df["Timestamp"].dt.minute

# Resource share per row (percentage of total)
df["Resource_Allocation_%"] = df["Resource_Allocation"] / df["Resource_Allocation"].sum() * 100

# Bandwidth efficiency — guard against zero denominator
df["Bandwidth_Efficiency"] = np.where(
    df["Required_Bandwidth_Mbps"] > 0,
    df["Allocated_Bandwidth_Mbps"] / df["Required_Bandwidth_Mbps"],
    np.nan,
)

# ── 2. Encode & build feature matrix ───────────────────────────────────────────

le = LabelEncoder()
df["Application_Type_Encoded"] = le.fit_transform(df["Application_Type"])

FEATURES = [
    "Application_Type_Encoded",
    "Signal_Strength",
    "Latency",
    "Required_Bandwidth_Mbps",
    "Hour",
    "Minute",
]
TARGET = "Resource_Allocation"

clean = df.dropna(subset=FEATURES + [TARGET]).copy()
X = clean[FEATURES]
y = clean[TARGET]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 3. Train model ─────────────────────────────────────────────────────────────

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae  = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
r2   = r2_score(y_test, y_pred)

print("=" * 50)
print("  Model Evaluation")
print("=" * 50)
print(f"  MAE  : {mae:.4f}")
print(f"  RMSE : {rmse:.4f}")
print(f"  R²   : {r2:.4f}")

# ── 4. Summaries ───────────────────────────────────────────────────────────────

app_stats = (
    df.groupby("Application_Type")["Resource_Allocation"]
    .agg(["mean", "min", "max"])
    .sort_values("mean", ascending=False)
)
bw_eff = (
    df.groupby("Application_Type")["Bandwidth_Efficiency"]
    .mean()
    .sort_values(ascending=False)
)
feat_imp = pd.DataFrame(
    {"Feature": FEATURES, "Importance": model.feature_importances_}
).sort_values("Importance", ascending=False)

print("\nResource Allocation by Application Type:")
print(app_stats.to_string())
print("\nBandwidth Allocation Efficiency by Application Type:")
print(bw_eff.to_string())
print("\nFeature Importance:")
print(feat_imp.to_string(index=False))

# ── 5. Visualisations ──────────────────────────────────────────────────────────

app_order = app_stats.index.tolist()
APP_COLORS = {app: PALETTE[i % len(PALETTE)] for i, app in enumerate(app_order)}
rng = np.random.default_rng(0)

# ── Fig 1: Predicted allocation trends by signal strength ──────────────────────
signal_range = np.linspace(-110, -50, 200)

fig, ax = plt.subplots(figsize=(13, 7))
for i, app in enumerate(app_order):
    app_idx = le.transform([app])[0]
    app_rows = df[df["Application_Type"] == app]
    sample = pd.DataFrame({
        "Application_Type_Encoded": [app_idx] * len(signal_range),
        "Signal_Strength": signal_range,
        "Latency": [app_rows["Latency"].mean()] * len(signal_range),
        "Required_Bandwidth_Mbps": [app_rows["Required_Bandwidth_Mbps"].mean()] * len(signal_range),
        "Hour": [10] * len(signal_range),
        "Minute": [0] * len(signal_range),
    })
    ax.plot(signal_range, model.predict(sample),
            label=app, linewidth=2.2, color=PALETTE[i % len(PALETTE)])

ax.set_xlabel("Signal Strength (dBm)", fontsize=12)
ax.set_ylabel("Predicted Resource Allocation (%)", fontsize=12)
ax.set_title("Predicted Resource Allocation Trends by Signal Strength",
             fontsize=14, fontweight="bold")
ax.legend(title="Application Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
ax.set_xlim(signal_range[0], signal_range[-1])
plt.tight_layout()
plt.savefig("fig1.png", dpi=150)
plt.savefig("predicted_trends.png", dpi=150)
plt.close()

# ── Fig 2: Resource allocation by app type — strip plot (data is discrete) ─────
fig, ax = plt.subplots(figsize=(13, 6))
means = app_stats["mean"]
for i, app in enumerate(app_order):
    subset = df[df["Application_Type"] == app]["Resource_Allocation"].values
    y_jitter = rng.uniform(-0.3, 0.3, len(subset))
    ax.scatter(subset, np.full(len(subset), i) + y_jitter,
               color=APP_COLORS[app], alpha=0.55, s=28, zorder=2)
    # mean shown as a white-outlined diamond so it's always visible
    ax.scatter([means[app]], [i], marker="D", s=90,
               color="black", edgecolors="white", linewidths=1.2, zorder=4)

ax.set_yticks(range(len(app_order)))
ax.set_yticklabels(app_order, fontsize=10)
ax.set_xlabel("Resource Allocation (%)", fontsize=12)
ax.set_title("Resource Allocation Distribution by Application Type\n"
             "(vertical bar = mean)", fontsize=14, fontweight="bold")
ax.invert_yaxis()
ax.set_xlim(45, 95)
plt.tight_layout()
plt.savefig("fig2.png", dpi=150)
plt.savefig("resource_allocation_by_app.png", dpi=150)
plt.close()

# ── Fig 3: Feature importance ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
feat_colors = [PALETTE[i] for i in range(len(feat_imp))]
bars = ax.barh(feat_imp["Feature"], feat_imp["Importance"],
               color=feat_colors, edgecolor="white", linewidth=0.5)
ax.bar_label(bars, fmt="%.4f", padding=5, fontsize=9)
ax.set_title("Feature Importance for Resource Allocation Prediction",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Importance Score")
ax.invert_yaxis()
ax.set_xlim(0, feat_imp["Importance"].max() * 1.15)
plt.tight_layout()
plt.savefig("fig3.png", dpi=150)
plt.savefig("feature_importance.png", dpi=150)
plt.close()

# ── Fig 4: Predicted vs actual, dots coloured by absolute error ────────────────
errors = np.abs(y_pred - y_test.values)
lims = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]

fig, ax = plt.subplots(figsize=(8, 8))
sc = ax.scatter(y_test, y_pred, c=errors, cmap="RdYlGn_r",
                vmin=0, vmax=errors.max(), alpha=0.75, s=55, edgecolors="none", zorder=3)
cbar = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
cbar.set_label("Absolute Error (%)", fontsize=10)
ax.plot(lims, lims, color="black", linestyle="--", linewidth=1.5,
        label="Perfect prediction", zorder=2)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("Actual Resource Allocation (%)", fontsize=12)
ax.set_ylabel("Predicted Resource Allocation (%)", fontsize=12)
ax.set_title(
    f"Predicted vs Actual Values\nMAE = {mae:.2f}%   RMSE = {rmse:.2f}%   R² = {r2:.4f}",
    fontsize=13, fontweight="bold",
)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("fig4.png", dpi=150)
plt.close()

# ── Fig 5: Signal strength vs resource allocation (Y-jittered) ────────────────
fig, ax = plt.subplots(figsize=(12, 6))
for app in app_order:
    subset = df[df["Application_Type"] == app]
    y_jitter = rng.uniform(-1.2, 1.2, len(subset))
    ax.scatter(subset["Signal_Strength"],
               subset["Resource_Allocation"] + y_jitter,
               label=app, color=APP_COLORS[app], alpha=0.55, s=18, edgecolors="none")
ax.set_xlabel("Signal Strength (dBm)", fontsize=12)
ax.set_ylabel("Resource Allocation (%)", fontsize=12)
ax.set_title("Signal Strength vs Resource Allocation", fontsize=14, fontweight="bold")
ax.legend(title="App Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("signal_vs_resource.png", dpi=150)
plt.close()

# ── Fig 6: Latency vs resource allocation (Y-jittered) ────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
for app in app_order:
    subset = df[df["Application_Type"] == app]
    y_jitter = rng.uniform(-1.2, 1.2, len(subset))
    ax.scatter(subset["Latency"],
               subset["Resource_Allocation"] + y_jitter,
               label=app, color=APP_COLORS[app], alpha=0.55, s=18, edgecolors="none")
ax.set_xlabel("Latency (ms)", fontsize=12)
ax.set_ylabel("Resource Allocation (%)", fontsize=12)
ax.set_title("Latency vs Resource Allocation", fontsize=14, fontweight="bold")
ax.legend(title="App Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("latency_vs_resource.png", dpi=150)
plt.close()

# ── Fig 7: Bandwidth efficiency ────────────────────────────────────────────────
bw_sorted = bw_eff.sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.barh(bw_sorted.index, bw_sorted.values,
               color=[APP_COLORS[a] for a in bw_sorted.index],
               edgecolor="white", linewidth=0.5)
ax.axvline(1.0, color="crimson", linestyle="--", linewidth=1.5, label="Efficiency = 1.0")
ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=9)
ax.set_title("Bandwidth Allocation Efficiency by Application Type",
             fontsize=14, fontweight="bold")
ax.set_xlabel("Allocated / Required Bandwidth")
ax.legend(fontsize=10)
ax.set_xlim(0, bw_sorted.max() * 1.12)
plt.tight_layout()
plt.savefig("bandwidth_efficiency.png", dpi=150)
plt.close()

# ── Fig 8: Dashboard (2×2) ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38)

# top-left: mean resource allocation (sorted barh)
ax0 = fig.add_subplot(gs[0, 0])
mean_alloc = app_stats["mean"].sort_values(ascending=True)
ax0.barh(mean_alloc.index, mean_alloc.values,
         color=[APP_COLORS[a] for a in mean_alloc.index], edgecolor="white")
ax0.set_xlim(0, 100)
ax0.bar_label(ax0.containers[0], fmt="%.1f%%", padding=3, fontsize=8)
ax0.set_title("Mean Resource Allocation", fontweight="bold")
ax0.set_xlabel("Mean Allocation (%)")

# top-right: feature importance
ax1 = fig.add_subplot(gs[0, 1])
ax1.barh(feat_imp["Feature"], feat_imp["Importance"],
         color=[PALETTE[i] for i in range(len(feat_imp))], edgecolor="white")
ax1.bar_label(ax1.containers[0], fmt="%.3f", padding=3, fontsize=8)
ax1.invert_yaxis()
ax1.set_title("Feature Importance", fontweight="bold")
ax1.set_xlabel("Importance Score")
ax1.set_xlim(0, feat_imp["Importance"].max() * 1.18)

# bottom-left: predicted vs actual
ax2 = fig.add_subplot(gs[1, 0])
sc2 = ax2.scatter(y_test, y_pred, c=errors, cmap="RdYlGn_r",
                  vmin=0, vmax=errors.max(), alpha=0.7, s=30, edgecolors="none")
ax2.plot(lims, lims, "k--", linewidth=1.2)
ax2.set_xlim(lims); ax2.set_ylim(lims)
ax2.set_xlabel("Actual (%)")
ax2.set_ylabel("Predicted (%)")
ax2.set_title(f"Predicted vs Actual  (R² = {r2:.3f})", fontweight="bold")
fig.colorbar(sc2, ax=ax2, fraction=0.04, pad=0.02).set_label("Error", fontsize=8)

# bottom-right: bandwidth efficiency
ax3 = fig.add_subplot(gs[1, 1])
bw_dash = bw_eff.sort_values(ascending=True)
ax3.barh(bw_dash.index, bw_dash.values,
         color=[APP_COLORS[a] for a in bw_dash.index], edgecolor="white")
ax3.axvline(1.0, color="crimson", linestyle="--", linewidth=1.4, label="1.0")
ax3.bar_label(ax3.containers[0], fmt="%.2f", padding=3, fontsize=8)
ax3.set_title("Bandwidth Efficiency (Alloc / Required)", fontweight="bold")
ax3.set_xlabel("Efficiency Ratio")
ax3.legend(fontsize=9)
ax3.set_xlim(0, bw_dash.max() * 1.14)

fig.suptitle("5G Resource Allocation — Analysis Dashboard",
             fontsize=17, fontweight="bold")
plt.savefig("dashboard.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nAll figures saved. Done.")

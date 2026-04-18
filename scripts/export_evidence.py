"""Export benchmark charts to evidence/ as static PNGs."""

import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

RESULTS_DIR = Path("results")
EVIDENCE_DIR = Path("evidence")
EVIDENCE_DIR.mkdir(exist_ok=True)

# --- 1. Throughput bar chart ---
df = pd.read_csv(RESULTS_DIR / "benchmark_results.csv")
fig = go.Figure(go.Bar(
    x=df["Configuration"],
    y=df["Throughput (req/s)"],
    marker_color=["#636EFA", "#636EFA", "#EF553B", "#00CC96", "#00CC96"],
    text=df["Throughput (req/s)"].apply(lambda x: f"{x:.0f}"),
    textposition="outside",
))
fig.update_layout(
    title="Throughput by Configuration (req/s)",
    yaxis_title="Requests per Second",
    xaxis_title="",
    template="plotly_white",
    width=800, height=500,
    margin=dict(t=60, b=100),
)
fig.write_image(str(EVIDENCE_DIR / "throughput_comparison.png"), scale=2)
print("Saved throughput_comparison.png")

# --- 2. Latency comparison (grouped bar) ---
fig2 = go.Figure()
for col, name in [("p50 (ms)", "p50"), ("p95 (ms)", "p95"), ("p99 (ms)", "p99")]:
    fig2.add_trace(go.Bar(name=name, x=df["Configuration"], y=df[col]))
fig2.update_layout(
    title="Latency Percentiles by Configuration (ms)",
    yaxis_title="Latency (ms)",
    xaxis_title="",
    barmode="group",
    template="plotly_white",
    width=800, height=500,
    margin=dict(t=60, b=100),
)
fig2.write_image(str(EVIDENCE_DIR / "latency_comparison.png"), scale=2)
print("Saved latency_comparison.png")

# --- 3. Cost analysis ---
cost_df = pd.read_csv(RESULTS_DIR / "cost_analysis.csv")
fig3 = go.Figure(go.Bar(
    x=cost_df["Configuration"],
    y=cost_df["Cost per 1K requests ($)"],
    marker_color=["#636EFA", "#636EFA", "#EF553B", "#00CC96", "#00CC96"],
    text=cost_df["Cost per 1K requests ($)"].apply(lambda x: f"${x:.4f}"),
    textposition="outside",
))
fig3.update_layout(
    title="Cost per 1,000 Embedding Requests ($)",
    yaxis_title="Cost ($)",
    xaxis_title="",
    template="plotly_white",
    width=800, height=500,
    margin=dict(t=60, b=100),
)
fig3.write_image(str(EVIDENCE_DIR / "cost_analysis.png"), scale=2)
print("Saved cost_analysis.png")

# --- 4. Batch size sweep ---
batch_df = pd.read_csv(RESULTS_DIR / "batch_throughput.csv")
fig4 = go.Figure(go.Scatter(
    x=batch_df["batch_size"],
    y=batch_df["throughput_rps"],
    mode="lines+markers",
    marker=dict(size=10),
    line=dict(width=3),
))
fig4.update_layout(
    title="Throughput vs. Batch Size (ONNX + INT8)",
    xaxis_title="Batch Size",
    yaxis_title="Throughput (req/s)",
    template="plotly_white",
    width=800, height=500,
)
fig4.write_image(str(EVIDENCE_DIR / "batch_throughput_sweep.png"), scale=2)
print("Saved batch_throughput_sweep.png")

print(f"\nAll charts exported to {EVIDENCE_DIR.resolve()}")

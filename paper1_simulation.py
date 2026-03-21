"""
PAPER 1: "When the AI Predicts Progression and Treatment Brings Remission"

Complete simulation code: three simulations demonstrating three levels of failure.

Simulation 1: h(x;theta) vs h(x,u;theta) -- omission failure (Move 1)
Simulation 2: h(x,u;theta) directionality test -- trajectory indeterminacy (Move 2)
Simulation 3: Observational degeneracy -- deployment collapse (Move 3)

All outputs saved to RUN_DIR.
"""

import os
import json
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------
# 0. Setup
# --------------------------------------------------

RUN_ID = time.strftime("%Y%m%d_%H%M%S")
BASE_DIR = Path("/kaggle/working/glioma_paper1")
RUN_DIR = BASE_DIR / f"run_{RUN_ID}"
RUN_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
print(f"Run dir: {RUN_DIR}")


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def safe_to_csv(df, path):
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# --------------------------------------------------
# 1. Dynamics
# --------------------------------------------------

K = 1.0
dt = 0.25
n_steps = 50
n_patients = 300
x0_low, x0_high = 0.15, 0.65
r_low, r_high = 0.08, 0.14
alpha_prog = 0.00
alpha_rem = 0.25
shared_lo, shared_hi = 0.30, 0.60

epochs = 40
eval_every = 5
batch_size = 256
hidden = 32
lr = 1e-3


def step_gompertz(x, r, K, alpha, dt):
    x = max(float(x), 1e-8)
    dxdt = r * x * np.log(K / x) - alpha * x
    x_next = x + dt * dxdt
    return max(float(x_next), 1e-8), float(dxdt)


def simulate_trajectory(x0, r, K, alpha, dt, n_steps, noise_std=0.0):
    xs = [float(x0)]
    deltas = []
    for _ in range(n_steps - 1):
        x_next, _ = step_gompertz(xs[-1], r, K, alpha, dt)
        if noise_std > 0:
            x_next = max(float(x_next + np.random.normal(0, noise_std)), 1e-8)
        delta_x = x_next - xs[-1]
        xs.append(x_next)
        deltas.append(delta_x)
    return np.array(xs, dtype=np.float32), np.array(deltas, dtype=np.float32)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------
# 2. Generate trajectories (shared across all simulations)
# --------------------------------------------------

print("\n" + "=" * 60)
print("GENERATING TRAJECTORIES")
print("=" * 60)

rows = []
for regime_name, regime_id, alpha in [
    ("progression", 0, alpha_prog),
    ("remission", 1, alpha_rem),
]:
    for patient_id in range(n_patients):
        x0 = np.random.uniform(x0_low, x0_high)
        r = np.random.uniform(r_low, r_high)
        xs, deltas = simulate_trajectory(x0, r, K, alpha, dt, n_steps)
        for t in range(n_steps - 1):
            rows.append({
                "patient_id": patient_id,
                "regime": regime_name,
                "regime_id": regime_id,
                "t": t,
                "x_t": float(xs[t]),
                "x_next": float(xs[t + 1]),
                "delta_x": float(deltas[t]),
                "alpha": float(alpha),
                "r": float(r),
                "K": float(K),
            })

df_all = pd.DataFrame(rows)
safe_to_csv(df_all, RUN_DIR / "trajectories_all.csv")
print(f"Total rows: {len(df_all)}")

# Shared-state subset
shared_df = df_all[(df_all["x_t"] >= shared_lo) & (df_all["x_t"] <= shared_hi)].copy()
prog_shared = shared_df[shared_df["regime_id"] == 0]
rem_shared = shared_df[shared_df["regime_id"] == 1]

mean_prog_dx = float(prog_shared["delta_x"].mean())
mean_rem_dx = float(rem_shared["delta_x"].mean())
shared_discrepancy = abs(mean_prog_dx - mean_rem_dx)
print(f"Shared-state discrepancy: {shared_discrepancy:.6f}")

# Global arrays
X_state = df_all[["x_t"]].to_numpy(dtype=np.float32)
X_state_treat = df_all[["x_t", "alpha"]].to_numpy(dtype=np.float32)
Y = df_all[["delta_x"]].to_numpy(dtype=np.float32)
regime = df_all["regime_id"].to_numpy(dtype=np.int64)

N = len(Y)
idx = np.random.permutation(N)
split = int(0.8 * N)
tr_idx, te_idx = idx[:split], idx[split:]

# TBR computation
n_bins_tbr = 40
bins_tbr = np.linspace(df_all["x_t"].min(), df_all["x_t"].max(), n_bins_tbr + 1)
df_all["x_bin"] = pd.cut(df_all["x_t"], bins=bins_tbr, include_lowest=True)

tbr_rows = []
for xbin, grp in df_all.groupby("x_bin", observed=False):
    g0 = grp[grp["regime_id"] == 0]["delta_x"]
    g1 = grp[grp["regime_id"] == 1]["delta_x"]
    if len(g0) == 0 or len(g1) == 0:
        continue
    x_mid = float((xbin.left + xbin.right) / 2.0)
    tbr_rows.append({
        "x_mid": x_mid,
        "tbr_local": float(abs(g0.mean() - g1.mean())),
        "n_prog": int(len(g0)),
        "n_rem": int(len(g1)),
    })
df_tbr = pd.DataFrame(tbr_rows)
safe_to_csv(df_tbr, RUN_DIR / "tbr_curve.csv")

# ============================================================
# SIMULATION 1: OMISSION FAILURE (Move 1)
# h(x;theta) vs h(x,u;theta)
# ============================================================

print("\n" + "=" * 60)
print("SIMULATION 1: OMISSION FAILURE")
print("h(x;theta) cannot distinguish treatments at shared x")
print("=" * 60)


def train_model(name, model, X_tr, Y_tr, X_te, Y_te, epochs=40, lr=1e-3):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    loader = DataLoader(TensorDataset(X_tr, Y_tr), batch_size=batch_size, shuffle=True)

    best_mse = float("inf")
    best_state = None
    hist = []

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep == 1 or ep % eval_every == 0 or ep == epochs:
            model.eval()
            with torch.no_grad():
                test_mse = loss_fn(model(X_te.to(device)), Y_te.to(device)).item()
            if test_mse < best_mse:
                best_mse = test_mse
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            hist.append({"epoch": ep, "test_mse": test_mse})
            if ep % 10 == 0 or ep == epochs:
                print(f"  [{name}] ep={ep} test_mse={test_mse:.8f}")

    if best_state:
        model.load_state_dict(best_state)
    return model, best_mse, pd.DataFrame(hist)


# Train state-only
X_s_tr = torch.tensor(X_state[tr_idx], dtype=torch.float32)
X_s_te = torch.tensor(X_state[te_idx], dtype=torch.float32)
X_st_tr = torch.tensor(X_state_treat[tr_idx], dtype=torch.float32)
X_st_te = torch.tensor(X_state_treat[te_idx], dtype=torch.float32)
Y_tr_t = torch.tensor(Y[tr_idx], dtype=torch.float32)
Y_te_t = torch.tensor(Y[te_idx], dtype=torch.float32)
reg_te = regime[te_idx]

print("\nTraining h(x;theta)...")
model_x, best_x, hist_x = train_model(
    "state_only", MLP(1, hidden), X_s_tr, Y_tr_t, X_s_te, Y_te_t, epochs, lr
)

print("\nTraining h(x,u;theta)...")
model_xu, best_xu, hist_xu = train_model(
    "treat_aware", MLP(2, hidden), X_st_tr, Y_tr_t, X_st_te, Y_te_t, epochs, lr
)

improvement = best_x / best_xu if best_xu > 0 else float("inf")
print(f"\nState-only best MSE:      {best_x:.8f}")
print(f"Treatment-aware best MSE: {best_xu:.8f}")
print(f"Improvement factor:       {improvement:.1f}x")


def pred_np(model, X):
    model.eval()
    with torch.no_grad():
        return model(X.to(device)).cpu().numpy().reshape(-1)


pred_state = pred_np(model_x, X_s_te)
pred_treat = pred_np(model_xu, X_st_te)
y_true = Y_te_t.numpy().reshape(-1)

# Minimal Simulation 1 figure
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].scatter(y_true, pred_state, s=8, alpha=0.25, label="h(x;theta)")
axes[0].plot(
    [y_true.min(), y_true.max()],
    [y_true.min(), y_true.max()],
    linestyle="--",
    linewidth=1,
    color="black",
)
axes[0].set_title("State-only predictions")
axes[0].set_xlabel("True delta_x")
axes[0].set_ylabel("Predicted delta_x")
axes[0].grid(alpha=0.2)

axes[1].scatter(y_true, pred_treat, s=8, alpha=0.25, label="h(x,u;theta)")
axes[1].plot(
    [y_true.min(), y_true.max()],
    [y_true.min(), y_true.max()],
    linestyle="--",
    linewidth=1,
    color="black",
)
axes[1].set_title("Treatment-aware predictions")
axes[1].set_xlabel("True delta_x")
axes[1].set_ylabel("Predicted delta_x")
axes[1].grid(alpha=0.2)

fig.suptitle("Move 1: Omission failure", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(RUN_DIR / "figure1_omission_failure.png", dpi=220, bbox_inches="tight")
plt.savefig(RUN_DIR / "figure1_omission_failure.pdf", bbox_inches="tight")
plt.close()

safe_to_csv(hist_x, RUN_DIR / "sim1_state_only_history.csv")
safe_to_csv(hist_xu, RUN_DIR / "sim1_treat_aware_history.csv")

sim1_summary = {
    "state_only_best_mse": float(best_x),
    "treat_aware_best_mse": float(best_xu),
    "improvement_factor": float(improvement),
    "shared_discrepancy": float(shared_discrepancy),
}
save_json(sim1_summary, RUN_DIR / "sim1_summary.json")

# ============================================================
# SIMULATION 2: TRAJECTORY INDETERMINACY (Move 2)
# h(x,u;theta) knows treatment but cannot determine trajectory
# from a single observation
# ============================================================

print("\n" + "=" * 60)
print("SIMULATION 2: TRAJECTORY INDETERMINACY")
print("h(x,u;theta) sees treatment but cannot determine trajectory")
print("=" * 60)

# -- 2A: Directionality accuracy --
print("\n-- 2A: Directionality accuracy at shared states --")

te_shared_mask = (
    (X_st_te[:, 0].numpy() >= shared_lo)
    & (X_st_te[:, 0].numpy() <= shared_hi)
)

x_te_shared = X_st_te[te_shared_mask]
y_te_shared = Y_te_t[te_shared_mask].numpy().reshape(-1)
reg_te_shared = reg_te[te_shared_mask]

pred_shared = pred_np(model_xu, x_te_shared)

sign_true = np.sign(y_te_shared)
sign_pred = np.sign(pred_shared)

prog_mask_s = reg_te_shared == 0
rem_mask_s = reg_te_shared == 1

sign_acc_prog = float(np.mean(sign_true[prog_mask_s] == sign_pred[prog_mask_s]))
sign_acc_rem = float(np.mean(sign_true[rem_mask_s] == sign_pred[rem_mask_s]))
sign_acc_all = float(np.mean(sign_true == sign_pred))

print(f"Directionality accuracy (progression): {sign_acc_prog:.4f}")
print(f"Directionality accuracy (remission):   {sign_acc_rem:.4f}")
print(f"Directionality accuracy (all shared):  {sign_acc_all:.4f}")

# -- 2B: Counterfactual coverage test --
print("\n-- 2B: Counterfactual coverage analysis --")

n_bins_cf = 20
cf_bins = np.linspace(shared_lo, shared_hi, n_bins_cf + 1)

x_train_vals = X_st_tr[:, 0].numpy()
alpha_train_vals = X_st_tr[:, 1].numpy()

cf_rows = []
for i in range(n_bins_cf):
    lo, hi = cf_bins[i], cf_bins[i + 1]
    x_mid = (lo + hi) / 2.0

    in_bin = (x_train_vals >= lo) & (x_train_vals < hi)
    n_prog_train = int(np.sum(in_bin & np.isclose(alpha_train_vals, alpha_prog)))
    n_rem_train = int(np.sum(in_bin & np.isclose(alpha_train_vals, alpha_rem)))

    total = n_prog_train + n_rem_train
    if total > 0 and max(n_prog_train, n_rem_train) > 0:
        coverage_ratio = min(n_prog_train, n_rem_train) / max(n_prog_train, n_rem_train)
    else:
        coverage_ratio = 0.0

    te_in_bin = (
        (x_te_shared[:, 0].numpy() >= lo)
        & (x_te_shared[:, 0].numpy() < hi)
    )
    if np.sum(te_in_bin) > 0:
        bin_sign_acc = float(np.mean(sign_true[te_in_bin] == sign_pred[te_in_bin]))
    else:
        bin_sign_acc = float("nan")

    cf_rows.append({
        "x_mid": x_mid,
        "n_prog_train": n_prog_train,
        "n_rem_train": n_rem_train,
        "coverage_ratio": coverage_ratio,
        "sign_accuracy": bin_sign_acc,
    })

df_cf = pd.DataFrame(cf_rows)
safe_to_csv(df_cf, RUN_DIR / "sim2_counterfactual_coverage.csv")

print(f"Mean coverage ratio: {df_cf['coverage_ratio'].mean():.3f}")
print(
    f"Bins with ratio < 0.3 (poor counterfactual): "
    f"{(df_cf['coverage_ratio'] < 0.3).sum()}/{len(df_cf)}"
)

# -- 2C: Single-observation indeterminacy --
print("\n-- 2C: Single-observation indeterminacy --")

n_test_points = 200
x_test_novel = np.random.uniform(shared_lo, shared_hi, n_test_points).astype(np.float32)

with torch.no_grad():
    input_prog = torch.tensor(
        np.column_stack([x_test_novel, np.full(n_test_points, alpha_prog)]).astype(np.float32)
    )
    input_rem = torch.tensor(
        np.column_stack([x_test_novel, np.full(n_test_points, alpha_rem)]).astype(np.float32)
    )

    pred_as_prog = model_xu(input_prog.to(device)).cpu().numpy().reshape(-1)
    pred_as_rem = model_xu(input_rem.to(device)).cpu().numpy().reshape(-1)

r_ref = (r_low + r_high) / 2.0
true_delta_prog = np.array([
    dt * (r_ref * x * np.log(K / x) - alpha_prog * x) for x in x_test_novel
])
true_delta_rem = np.array([
    dt * (r_ref * x * np.log(K / x) - alpha_rem * x) for x in x_test_novel
])

sign_correct_prog = float(np.mean(np.sign(pred_as_prog) == np.sign(true_delta_prog)))
sign_correct_rem = float(np.mean(np.sign(pred_as_rem) == np.sign(true_delta_rem)))

direction_separation = float(np.mean(np.sign(pred_as_prog) != np.sign(pred_as_rem)))

pred_gap = pred_as_prog - pred_as_rem
true_gap = true_delta_prog - true_delta_rem

gap_ratio = (
    float(np.mean(np.abs(pred_gap)) / np.mean(np.abs(true_gap)))
    if np.mean(np.abs(true_gap)) > 0
    else float("nan")
)

print(f"Sign accuracy (progression query): {sign_correct_prog:.4f}")
print(f"Sign accuracy (remission query):   {sign_correct_rem:.4f}")
print(f"Direction separation rate:         {direction_separation:.4f}")
print(f"Mean |predicted gap|:              {np.mean(np.abs(pred_gap)):.6f}")
print(f"Mean |true gap|:                   {np.mean(np.abs(true_gap)):.6f}")
print(f"Gap ratio (pred/true):             {gap_ratio:.4f}")

# -- 2D: The trajectory indeterminacy measure (TIM) --
tim_values = np.abs(true_delta_prog - true_delta_rem) / np.maximum(
    np.abs(true_delta_prog), np.abs(true_delta_rem)
)
tim_values = np.where(np.isfinite(tim_values), tim_values, 0.0)

mean_tim = float(np.mean(tim_values))
print("\nTrajectory Indeterminacy Measure (TIM):")
print(f"  Mean TIM in shared window: {mean_tim:.4f}")
print("  TIM = 1 means the two regimes have equal-magnitude")
print("  but opposite dynamics. TIM > 0 means a single")
print("  observation cannot determine trajectory.")

sim2_indeterminacy = pd.DataFrame({
    "x": x_test_novel,
    "true_delta_prog": true_delta_prog,
    "true_delta_rem": true_delta_rem,
    "pred_as_prog": pred_as_prog,
    "pred_as_rem": pred_as_rem,
    "pred_gap": pred_gap,
    "true_gap": true_gap,
    "TIM": tim_values,
})
safe_to_csv(sim2_indeterminacy, RUN_DIR / "sim2_indeterminacy.csv")

sim2_summary = {
    "sign_accuracy_progression": float(sign_correct_prog),
    "sign_accuracy_remission": float(sign_correct_rem),
    "sign_accuracy_all_shared": float(sign_acc_all),
    "direction_separation_rate": float(direction_separation),
    "gap_ratio_pred_vs_true": float(gap_ratio),
    "mean_TIM": float(mean_tim),
    "mean_coverage_ratio": float(df_cf["coverage_ratio"].mean()),
}
save_json(sim2_summary, RUN_DIR / "sim2_summary.json")

# -- Simulation 2 Figures --
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

ax = axes[0]
order = np.argsort(x_test_novel)
ax.plot(
    x_test_novel[order],
    true_delta_prog[order],
    linewidth=2,
    label="True f(x, u0) [progression]",
    color="tab:blue",
)
ax.plot(
    x_test_novel[order],
    true_delta_rem[order],
    linewidth=2,
    label="True f(x, u1) [remission]",
    color="tab:orange",
)
ax.scatter(
    x_test_novel,
    pred_as_prog,
    s=8,
    alpha=0.4,
    label="h(x, u0; theta) prediction",
    color="tab:blue",
    marker="x",
)
ax.scatter(
    x_test_novel,
    pred_as_rem,
    s=8,
    alpha=0.4,
    label="h(x, u1; theta) prediction",
    color="tab:orange",
    marker="x",
)
ax.axhline(0, color="black", linewidth=1)
ax.set_title("Treatment-aware predictions vs truth")
ax.set_xlabel("Tumour volume x")
ax.set_ylabel("delta_x")
ax.legend(fontsize=7)
ax.grid(alpha=0.2)

ax = axes[1]
ax.scatter(
    x_test_novel,
    true_gap,
    s=12,
    alpha=0.4,
    label="True gap: f(x,u0)-f(x,u1)",
    color="tab:purple",
)
ax.scatter(
    x_test_novel,
    pred_gap,
    s=12,
    alpha=0.4,
    label="Predicted gap: h(x,u0)-h(x,u1)",
    color="tab:green",
)
ax.axhline(0, color="black", linewidth=1, linestyle=":")
ax.set_title("Treatment effect: predicted vs true")
ax.set_xlabel("Tumour volume x")
ax.set_ylabel("Gap (progression - remission)")
ax.legend(fontsize=8)
ax.grid(alpha=0.2)

ax = axes[2]
ax.scatter(x_test_novel[order], tim_values[order], s=12, alpha=0.5, color="tab:red")
ax.axhline(
    mean_tim,
    color="tab:red",
    linewidth=1.5,
    linestyle="--",
    label=f"Mean TIM = {mean_tim:.3f}",
)
ax.set_title("Trajectory Indeterminacy Measure (TIM)")
ax.set_xlabel("Tumour volume x")
ax.set_ylabel("TIM(x)")
ax.legend(fontsize=9)
ax.grid(alpha=0.2)

fig.suptitle(
    "Move 2: Treatment awareness != trajectory awareness",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(RUN_DIR / "figure6_trajectory_indeterminacy.png", dpi=220, bbox_inches="tight")
plt.savefig(RUN_DIR / "figure6_trajectory_indeterminacy.pdf", bbox_inches="tight")
plt.close()
print("\nSaved figure 6: trajectory indeterminacy")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
ax.bar(
    df_cf["x_mid"],
    df_cf["n_prog_train"],
    width=0.012,
    alpha=0.6,
    label="Progression training points",
    color="tab:blue",
)
ax.bar(
    df_cf["x_mid"],
    -df_cf["n_rem_train"],
    width=0.012,
    alpha=0.6,
    label="Remission training points",
    color="tab:orange",
)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Counterfactual coverage by x-bin")
ax.set_xlabel("Tumour volume x")
ax.set_ylabel("Training samples (prog up / rem down)")
ax.legend(fontsize=8)

ax = axes[1]
valid = df_cf.dropna(subset=["sign_accuracy"])
ax.scatter(
    valid["coverage_ratio"],
    valid["sign_accuracy"],
    s=40,
    alpha=0.7,
    color="tab:red",
)
ax.set_title("Coverage ratio vs directionality accuracy")
ax.set_xlabel("Coverage ratio (0 = one regime only, 1 = balanced)")
ax.set_ylabel("Sign(delta_x) accuracy")
ax.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(RUN_DIR / "figure7_coverage_vs_accuracy.png", dpi=220, bbox_inches="tight")
plt.savefig(RUN_DIR / "figure7_coverage_vs_accuracy.pdf", bbox_inches="tight")
plt.close()
print("Saved figure 7: coverage vs accuracy")

# ============================================================
# SIMULATION 3: OBSERVATIONAL DEGENERACY (Move 3)
# At deployment, the model receives an image -- just x.
# It does not receive u. The treatment-aware architecture
# h(x,u;theta) cannot be populated from imaging data alone.
# ============================================================

print("\n" + "=" * 60)
print("SIMULATION 3: OBSERVATIONAL DEGENERACY")
print("h(x,u;theta) degrades when u is unavailable at inference")
print("=" * 60)

y_te_np = Y_te_t.numpy().reshape(-1)
x_te_np = X_st_te[:, 0].numpy()
alpha_te_np = X_st_te[:, 1].numpy()

pred_ideal = pred_np(model_xu, X_st_te)
mse_ideal = float(np.mean((pred_ideal - y_te_np) ** 2))

# Scenario A: assume untreated (u = 0)
X_te_assume_untreated = torch.tensor(
    np.column_stack([x_te_np, np.zeros_like(alpha_te_np)]).astype(np.float32)
)
pred_A = pred_np(model_xu, X_te_assume_untreated)
mse_A = float(np.mean((pred_A - y_te_np) ** 2))

# Scenario B: assume average treatment
alpha_mean = float(np.mean(alpha_te_np))
X_te_assume_avg = torch.tensor(
    np.column_stack([x_te_np, np.full_like(alpha_te_np, alpha_mean)]).astype(np.float32)
)
pred_B = pred_np(model_xu, X_te_assume_avg)
mse_B = float(np.mean((pred_B - y_te_np) ** 2))

# Scenario C: random u per observation
np.random.seed(99)
alpha_random = np.random.choice([alpha_prog, alpha_rem], size=len(alpha_te_np))
X_te_random = torch.tensor(
    np.column_stack([x_te_np, alpha_random]).astype(np.float32)
)
pred_C = pred_np(model_xu, X_te_random)
mse_C = float(np.mean((pred_C - y_te_np) ** 2))

# Scenario D: swapped u
alpha_swapped = np.where(np.isclose(alpha_te_np, alpha_prog), alpha_rem, alpha_prog)
X_te_swapped = torch.tensor(
    np.column_stack([x_te_np, alpha_swapped]).astype(np.float32)
)
pred_D = pred_np(model_xu, X_te_swapped)
mse_D = float(np.mean((pred_D - y_te_np) ** 2))

mse_state_only = float(np.mean((pred_state - y_te_np) ** 2))

print("\nDeployment scenario MSEs:")
print(f"  Ideal (correct u):           {mse_ideal:.8f}")
print(f"  Scenario A (assume u=0):     {mse_A:.8f}  ({mse_A/mse_ideal:.1f}x ideal)")
print(f"  Scenario B (assume u=mean):  {mse_B:.8f}  ({mse_B/mse_ideal:.1f}x ideal)")
print(f"  Scenario C (random u):       {mse_C:.8f}  ({mse_C/mse_ideal:.1f}x ideal)")
print(f"  Scenario D (swapped u):      {mse_D:.8f}  ({mse_D/mse_ideal:.1f}x ideal)")
print(f"  State-only h(x;theta):       {mse_state_only:.8f}  ({mse_state_only/mse_ideal:.1f}x ideal)")

# -- Directionality under each scenario --
te_sw = (x_te_np >= shared_lo) & (x_te_np <= shared_hi)


def sign_acc(pred, true, mask):
    return float(np.mean(np.sign(pred[mask]) == np.sign(true[mask])))


sa_ideal = sign_acc(pred_ideal, y_te_np, te_sw)
sa_A = sign_acc(pred_A, y_te_np, te_sw)
sa_B = sign_acc(pred_B, y_te_np, te_sw)
sa_C = sign_acc(pred_C, y_te_np, te_sw)
sa_D = sign_acc(pred_D, y_te_np, te_sw)
sa_state = sign_acc(pred_state, y_te_np, te_sw)

print("\nDirectionality accuracy (shared window):")
print(f"  Ideal:       {sa_ideal:.4f}")
print(f"  Assume u=0:  {sa_A:.4f}")
print(f"  Assume mean: {sa_B:.4f}")
print(f"  Random u:    {sa_C:.4f}")
print(f"  Swapped u:   {sa_D:.4f}")
print(f"  State-only:  {sa_state:.4f}")

# -- Regime-specific collapse --
rem_test_mask = (reg_te == 1) & te_sw

if np.sum(rem_test_mask) > 0:
    rem_correct_sign = float(np.mean(np.sign(pred_ideal[rem_test_mask]) < 0))
    rem_wrong_sign = float(np.mean(np.sign(pred_A[rem_test_mask]) > 0))
    rem_swapped_sign = float(np.mean(np.sign(pred_D[rem_test_mask]) > 0))

    print("\nRemission patients in shared window:")
    print(f"  Correct u -> predicts shrinkage: {rem_correct_sign:.4f}")
    print(f"  Assume u=0 -> predicts GROWTH:   {rem_wrong_sign:.4f}")
    print(f"  Swapped u -> predicts GROWTH:    {rem_swapped_sign:.4f}")
    print("  THIS IS THE CLINICAL DANGER: the AI says growth,")
    print("  the patient has response.")
else:
    rem_correct_sign = float("nan")
    rem_wrong_sign = float("nan")
    rem_swapped_sign = float("nan")

sim3_summary = {
    "mse_ideal": float(mse_ideal),
    "mse_assume_untreated": float(mse_A),
    "mse_assume_mean": float(mse_B),
    "mse_random_u": float(mse_C),
    "mse_swapped_u": float(mse_D),
    "mse_state_only": float(mse_state_only),
    "degradation_assume_untreated": float(mse_A / mse_ideal),
    "degradation_assume_mean": float(mse_B / mse_ideal),
    "degradation_random": float(mse_C / mse_ideal),
    "degradation_swapped": float(mse_D / mse_ideal),
    "sign_acc_ideal": float(sa_ideal),
    "sign_acc_assume_untreated": float(sa_A),
    "sign_acc_assume_mean": float(sa_B),
    "sign_acc_random": float(sa_C),
    "sign_acc_swapped": float(sa_D),
    "sign_acc_state_only": float(sa_state),
    "remission_correct_u_predicts_shrinkage": float(rem_correct_sign),
    "remission_assume_untreated_predicts_growth": float(rem_wrong_sign),
    "remission_swapped_u_predicts_growth": float(rem_swapped_sign),
}
save_json(sim3_summary, RUN_DIR / "sim3_summary.json")

# -- Simulation 3 Figures --
fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

ax = axes[0]
scenarios = [
    "Ideal\n(correct u)",
    "Assume\nuntreated",
    "Assume\nmean u",
    "Random u",
    "Swapped u",
    "State-only\nh(x;theta)",
]
mses = [mse_ideal, mse_A, mse_B, mse_C, mse_D, mse_state_only]
colors = ["tab:green", "tab:orange", "tab:orange", "tab:orange", "tab:red", "tab:gray"]

bars = ax.bar(scenarios, mses, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Mean Squared Error")
ax.set_title("Model performance under deployment scenarios")
ax.grid(axis="y", alpha=0.2)

for bar, mse_val in zip(bars, mses):
    if mse_val != mse_ideal:
        ratio = mse_val / mse_ideal
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{ratio:.1f}x",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

ax = axes[1]
sign_accs = [sa_ideal, sa_A, sa_B, sa_C, sa_D, sa_state]
bars = ax.bar(scenarios, sign_accs, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Directionality accuracy (shared window)")
ax.set_title("Can the model predict growth vs shrinkage?")
ax.set_ylim(0, 1.05)
ax.axhline(0.5, color="black", linewidth=1, linestyle=":", label="Chance")
ax.legend()
ax.grid(axis="y", alpha=0.2)

fig.suptitle(
    "Move 3: Without treatment data, the fix collapses",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(RUN_DIR / "figure8_deployment_collapse.png", dpi=220, bbox_inches="tight")
plt.savefig(RUN_DIR / "figure8_deployment_collapse.pdf", bbox_inches="tight")
plt.close()
print("\nSaved figure 8: deployment collapse")

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

rem_test = reg_te == 1

ax = axes[0]
ax.scatter(
    x_te_np[rem_test],
    pred_ideal[rem_test],
    s=6,
    alpha=0.2,
    color="tab:green",
    label="Correct u -> predicts remission",
)
ax.scatter(
    x_te_np[rem_test],
    pred_A[rem_test],
    s=6,
    alpha=0.2,
    color="tab:red",
    label="Assume untreated -> predicts...",
)
ax.axhline(0, color="black", linewidth=1.5)
ax.axvspan(shared_lo, shared_hi, alpha=0.06, color="gray")
ax.set_title("Remission patients: correct u vs assume untreated")
ax.set_xlabel("Tumour volume x")
ax.set_ylabel("Predicted delta_x")
ax.legend(fontsize=8, markerscale=4)
ax.annotate(
    "delta_x > 0: model says GROWTH\ndelta_x < 0: model says SHRINKAGE",
    xy=(0.02, 0.98),
    xycoords="axes fraction",
    fontsize=8,
    verticalalignment="top",
    bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
)

ax = axes[1]
ax.scatter(
    x_te_np[te_sw],
    y_te_np[te_sw],
    s=6,
    alpha=0.15,
    color="gray",
    label="True delta_x",
)
ax.scatter(
    x_te_np[te_sw],
    pred_D[te_sw],
    s=6,
    alpha=0.15,
    color="tab:red",
    label="Prediction (swapped u)",
)
ax.axhline(0, color="black", linewidth=1.5)
ax.set_title("Shared window: predictions under swapped treatment labels")
ax.set_xlabel("Tumour volume x")
ax.set_ylabel("delta_x")
ax.legend(fontsize=8, markerscale=4)

fig.suptitle(
    "When the AI predicts progression and treatment brings remission",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)
plt.tight_layout()
plt.savefig(RUN_DIR / "figure9_clinical_danger.png", dpi=220, bbox_inches="tight")
plt.savefig(RUN_DIR / "figure9_clinical_danger.pdf", bbox_inches="tight")
plt.close()
print("Saved figure 9: clinical danger")

fig, ax = plt.subplots(figsize=(7, 5))
if len(df_tbr) > 0:
    ax.plot(df_tbr["x_mid"], df_tbr["tbr_local"], linewidth=2.5, color="tab:blue")
    ax.axvspan(shared_lo, shared_hi, alpha=0.12, color="gray", label="shared-state window")
    ax.set_title("Local Treatment Blindness Risk (TBR)")
    ax.set_xlabel("Tumour volume x")
    ax.set_ylabel("TBR(x)")
    ax.legend()
    ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(RUN_DIR / "figure10_tbr.png", dpi=220, bbox_inches="tight")
plt.savefig(RUN_DIR / "figure10_tbr.pdf", bbox_inches="tight")
plt.close()
print("Saved figure 10: TBR curve")

# ============================================================
# SIDE ANALYSIS 1: PROGNOSTIC RANKING VS TRAJECTORY AWARENESS
# Paste this BEFORE the "FINAL SUMMARY" section in your
# original Kaggle code. All variables from above are reused.
# ============================================================

from scipy.stats import spearmanr

print("\n" + "=" * 60)
print("SIDE ANALYSIS 1: Prognostic ranking vs trajectory awareness")
print("=" * 60)

rho_so_raw, p_so_raw = spearmanr(y_true, pred_state)
rho_ta_raw, p_ta_raw = spearmanr(y_true, pred_treat)
rho_so_abs, p_so_abs = spearmanr(np.abs(y_true), np.abs(pred_state))
rho_ta_abs, p_ta_abs = spearmanr(np.abs(y_true), np.abs(pred_treat))

prog_test_sw = (reg_te == 0) & te_sw
rem_test_sw = (reg_te == 1) & te_sw

sign_so_shared = float(np.mean(np.sign(pred_state[te_sw]) == np.sign(y_true[te_sw])))
sign_ta_shared = float(np.mean(np.sign(pred_treat[te_sw]) == np.sign(y_true[te_sw])))
sign_so_prog = float(np.mean(np.sign(pred_state[prog_test_sw]) == np.sign(y_true[prog_test_sw])))
sign_so_rem = float(np.mean(np.sign(pred_state[rem_test_sw]) == np.sign(y_true[rem_test_sw])))
sign_ta_rem = float(np.mean(np.sign(pred_treat[rem_test_sw]) == np.sign(y_true[rem_test_sw])))

pct_rem_growth_so = float(np.mean(pred_state[rem_test_sw] > 0))

print(f"Overall Spearman (raw dx):  SO={rho_so_raw:.4f} (p={p_so_raw:.2e})  TA={rho_ta_raw:.4f}")
print(f"Overall Spearman (|dx|):    SO={rho_so_abs:.4f}  TA={rho_ta_abs:.4f}")
print(f"Sign acc shared window:     SO={sign_so_shared:.4f}  TA={sign_ta_shared:.4f}")
print(f"Sign acc remission:         SO={sign_so_rem:.4f}  TA={sign_ta_rem:.4f}")
print(f"Remission predicted growth: SO={pct_rem_growth_so:.1%}")

# Figure 12
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
ax.scatter(np.abs(y_true), np.abs(pred_state), alpha=0.1, s=6, color="steelblue")
ax.set_xlabel("True |delta_x|", fontsize=12)
ax.set_ylabel("Predicted |delta_x| (state-only)", fontsize=12)
ax.set_title(f"Magnitude ranking preserved\nSpearman rho = {rho_so_abs:.3f}", fontsize=13)
maxv = max(np.abs(y_true).max(), np.abs(pred_state).max())
ax.plot([0, maxv], [0, maxv], "k--", alpha=0.3)

ax = axes[1]
colors_sw = np.where(reg_te[te_sw] == 0, "steelblue", "coral")
ax.scatter(y_true[te_sw], pred_state[te_sw], alpha=0.25, s=10, c=colors_sw)
ax.axhline(0, color="black", linewidth=0.8)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("True delta_x", fontsize=12)
ax.set_ylabel("Predicted delta_x (state-only)", fontsize=12)
ax.set_title(f"Sign accuracy shared: {sign_so_shared:.1%}\nRemission as growth: {pct_rem_growth_so:.0%}", fontsize=13)
from matplotlib.lines import Line2D
ax.legend(handles=[
   Line2D([0],[0],marker="o",color="w",markerfacecolor="steelblue",label="Progression",markersize=8),
   Line2D([0],[0],marker="o",color="w",markerfacecolor="coral",label="Remission",markersize=8)
], fontsize=10)

ax = axes[2]
metrics = ["Overall\nSpearman rho", "Sign accuracy\n(shared window)", "Remission\ncorrect sign"]
so_v = [rho_so_raw, sign_so_shared, sign_so_rem]
ta_v = [rho_ta_raw, sign_ta_shared, sign_ta_rem]
xp = np.arange(len(metrics))
w = 0.35
b1 = ax.bar(xp - w/2, so_v, w, label="State-only", color="steelblue", alpha=0.8)
b2 = ax.bar(xp + w/2, ta_v, w, label="Treatment-aware", color="forestgreen", alpha=0.8)
ax.set_xticks(xp)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Prognostic ranking != trajectory awareness", fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.15)
for bar in b1:
   ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{bar.get_height():.2f}", ha="center", fontsize=9)
for bar in b2:
   ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f"{bar.get_height():.2f}", ha="center", fontsize=9)

plt.tight_layout()
plt.savefig(RUN_DIR / "figure12_prognostic_vs_trajectory.png", dpi=220, bbox_inches="tight")
plt.close()
print("Saved figure 12: prognostic vs trajectory")

save_json({
   "spearman_raw_so": float(rho_so_raw),
   "spearman_raw_ta": float(rho_ta_raw),
   "spearman_abs_so": float(rho_so_abs),
   "sign_acc_so_shared": sign_so_shared,
   "sign_acc_so_rem": sign_so_rem,
   "pct_rem_growth_so": pct_rem_growth_so,
}, RUN_DIR / "side1_summary.json")


# ============================================================
# SIDE ANALYSIS 2: TBR AND TIM AS PRE-DEPLOYMENT PREDICTORS
# ============================================================

print("\n" + "=" * 60)
print("SIDE ANALYSIS 2: TBR and TIM as pre-deployment diagnostics")
print("=" * 60)

r_mean = (r_low + r_high) / 2.0
x_range_plot = np.linspace(0.1, 0.7, 200)

def gompertz_dx_analytical(x, r, alpha):
   return dt * (r * x * np.log(K / np.clip(x, 1e-6, K - 1e-6)) - alpha * x)

dx_prog_analytical = gompertz_dx_analytical(x_range_plot, r_mean, alpha_prog)
dx_rem_analytical = gompertz_dx_analytical(x_range_plot, r_mean, alpha_rem)
TBR_analytical = np.abs(dx_prog_analytical - dx_rem_analytical)
denom_a = np.maximum(np.abs(dx_prog_analytical), np.abs(dx_rem_analytical))
TIM_analytical = np.where(denom_a > 1e-10, np.abs(dx_prog_analytical - dx_rem_analytical) / denom_a, 0)

# Binned sign accuracy for state-only model
n_bins_sa = 20
sa_bin_edges = np.linspace(shared_lo, shared_hi, n_bins_sa + 1)
sa_bin_centers = 0.5 * (sa_bin_edges[:-1] + sa_bin_edges[1:])

sign_acc_bins = []
tbr_bins = []
tim_bins = []

for i in range(n_bins_sa):
   lo_b, hi_b = sa_bin_edges[i], sa_bin_edges[i+1]
   mask_b = te_sw & (x_te_np >= lo_b) & (x_te_np < hi_b)
   if mask_b.sum() > 0:
       acc = float(np.mean(np.sign(pred_state[mask_b]) == np.sign(y_te_np[mask_b])))
   else:
       acc = float("nan")
   sign_acc_bins.append(acc)

   xc = sa_bin_centers[i]
   dp = gompertz_dx_analytical(xc, r_mean, alpha_prog)
   dr = gompertz_dx_analytical(xc, r_mean, alpha_rem)
   tbr_bins.append(abs(dp - dr))
   d = max(abs(dp), abs(dr))
   tim_bins.append(abs(dp - dr) / d if d > 1e-10 else 0)

sign_acc_bins = np.array(sign_acc_bins)
tbr_bins = np.array(tbr_bins)
tim_bins = np.array(tim_bins)
sign_err_bins = 1 - sign_acc_bins

valid_bins = ~np.isnan(sign_acc_bins)
rho_tbr, p_tbr = spearmanr(tbr_bins[valid_bins], sign_err_bins[valid_bins])
rho_tim, p_tim = spearmanr(tim_bins[valid_bins], sign_err_bins[valid_bins])

print(f"TBR range: [{tbr_bins.min():.4f}, {tbr_bins.max():.4f}]")
print(f"TIM range: [{tim_bins.min():.4f}, {tim_bins.max():.4f}]")
print(f"TBR > 0 everywhere: {np.all(tbr_bins > 0)}")
print(f"TIM > 1.0 everywhere: {np.all(tim_bins > 1.0)}")
print(f"TBR-error correlation: rho={rho_tbr:.4f} (p={p_tbr:.6f})")
print(f"TIM-error correlation: rho={rho_tim:.4f} (p={p_tim:.6f})")

# Figure 13
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

ax = axes[0]
ax.plot(x_range_plot, TBR_analytical, color="darkred", linewidth=2.5)
ax.axvspan(shared_lo, shared_hi, alpha=0.12, color="gray", label="Shared-state window")
ax.set_xlabel("Tumour volume x", fontsize=12)
ax.set_ylabel("TBR(x)", fontsize=12)
ax.set_title("Treatment Blindness Risk\n(pre-computable)", fontsize=13)
ax.legend(fontsize=10)

ax = axes[1]
vr = x_range_plot >= 0.12
ax.plot(x_range_plot[vr], TIM_analytical[vr], color="darkorange", linewidth=2.5)
ax.axvspan(shared_lo, shared_hi, alpha=0.12, color="gray", label="Shared-state window")
ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xlabel("Tumour volume x", fontsize=12)
ax.set_ylabel("TIM(x)", fontsize=12)
ax.set_title("Trajectory Indeterminacy Measure\nTIM > 1: sign-reversal risk", fontsize=13)
ax.legend(fontsize=10)

ax = axes[2]
ax2 = ax.twinx()
ax.bar(sa_bin_centers, sign_err_bins, width=0.012, color="firebrick", alpha=0.6, label="Sign error rate")
ax2.plot(sa_bin_centers, tbr_bins, color="darkblue", linewidth=2.5, marker="o", markersize=5, label="TBR(x)")
ax.set_xlabel("Tumour volume x", fontsize=12)
ax.set_ylabel("Sign error rate", fontsize=12, color="firebrick")
ax2.set_ylabel("TBR(x)", fontsize=12, color="darkblue")
ax.set_title(f"TBR predicts sign-error regions\nSpearman rho = {rho_tbr:.3f}", fontsize=13)
ax.tick_params(axis="y", labelcolor="firebrick")
ax2.tick_params(axis="y", labelcolor="darkblue")
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, lb1+lb2, fontsize=10, loc="upper left")

plt.tight_layout()
plt.savefig(RUN_DIR / "figure13_tbr_tim_predeployment.png", dpi=220, bbox_inches="tight")
plt.close()
print("Saved figure 13: TBR/TIM pre-deployment")

save_json({
   "tbr_min": float(tbr_bins.min()),
   "tbr_max": float(tbr_bins.max()),
   "tim_min": float(tim_bins.min()),
   "tim_max": float(tim_bins.max()),
   "tbr_error_rho": float(rho_tbr),
   "tbr_error_p": float(p_tbr),
   "tim_error_rho": float(rho_tim),
   "tim_error_p": float(p_tim),
}, RUN_DIR / "side2_summary.json")

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PAPER 1 SIMULATION COMPLETE")
print("=" * 60)

print("\nMove 1 (Omission failure):")
print(f"  h(x;theta) MSE:       {best_x:.8f}")
print(f"  h(x,u;theta) MSE:     {best_xu:.8f}")
print(f"  Improvement:          {improvement:.1f}x")

print("\nMove 2 (Trajectory indeterminacy):")
print(f"  Sign accuracy (prog): {sign_correct_prog:.4f}")
print(f"  Sign accuracy (rem):  {sign_correct_rem:.4f}")
print(f"  Gap ratio:            {gap_ratio:.4f}")
print(f"  Mean TIM:             {mean_tim:.4f}")

print("\nMove 3 (Observational degeneracy):")
print(f"  Ideal MSE:            {mse_ideal:.8f}")
print(f"  Assume untreated MSE: {mse_A:.8f} ({mse_A/mse_ideal:.1f}x)")
print(f"  Swapped u MSE:        {mse_D:.8f} ({mse_D/mse_ideal:.1f}x)")
print(f"  State-only MSE:       {mse_state_only:.8f}")
print(f"  Remission -> predicts growth when u unknown: {rem_wrong_sign:.1%}")

print(f"\nAll outputs: {RUN_DIR}")

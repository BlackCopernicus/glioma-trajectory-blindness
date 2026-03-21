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
ax.scatter(np.abs(y_true), np.abs(pred_state), alpha=0.1, s=6, color='steelblue')
ax.set_xlabel('True |delta_x|', fontsize=12)
ax.set_ylabel('Predicted |delta_x| (state-only)', fontsize=12)
ax.set_title(f'Magnitude ranking preserved\nSpearman rho = {rho_so_abs:.3f}', fontsize=13)
maxv = max(np.abs(y_true).max(), np.abs(pred_state).max())
ax.plot([0, maxv], [0, maxv], 'k--', alpha=0.3)

ax = axes[1]
colors_sw = np.where(reg_te[te_sw] == 0, 'steelblue', 'coral')
ax.scatter(y_true[te_sw], pred_state[te_sw], alpha=0.25, s=10, c=colors_sw)
ax.axhline(0, color='black', linewidth=0.8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('True delta_x', fontsize=12)
ax.set_ylabel('Predicted delta_x (state-only)', fontsize=12)
ax.set_title(f'Sign accuracy shared: {sign_so_shared:.1%}\nRemission as growth: {pct_rem_growth_so:.0%}', fontsize=13)
from matplotlib.lines import Line2D
ax.legend(handles=[
    Line2D([0],[0],marker='o',color='w',markerfacecolor='steelblue',label='Progression',markersize=8),
    Line2D([0],[0],marker='o',color='w',markerfacecolor='coral',label='Remission',markersize=8)
], fontsize=10)

ax = axes[2]
metrics = ['Overall\nSpearman rho', 'Sign accuracy\n(shared window)', 'Remission\ncorrect sign']
so_v = [rho_so_raw, sign_so_shared, sign_so_rem]
ta_v = [rho_ta_raw, sign_ta_shared, sign_ta_rem]
xp = np.arange(len(metrics))
w = 0.35
b1 = ax.bar(xp - w/2, so_v, w, label='State-only', color='steelblue', alpha=0.8)
b2 = ax.bar(xp + w/2, ta_v, w, label='Treatment-aware', color='forestgreen', alpha=0.8)
ax.set_xticks(xp)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Prognostic ranking != trajectory awareness', fontsize=13)
ax.legend(fontsize=10)
ax.set_ylim(0, 1.15)
for bar in b1:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f'{bar.get_height():.2f}', ha='center', fontsize=9)
for bar in b2:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.02, f'{bar.get_height():.2f}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(RUN_DIR / 'figure12_prognostic_vs_trajectory.png', dpi=220, bbox_inches='tight')
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
        acc = float('nan')
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
ax.plot(x_range_plot, TBR_analytical, color='darkred', linewidth=2.5)
ax.axvspan(shared_lo, shared_hi, alpha=0.12, color='gray', label='Shared-state window')
ax.set_xlabel('Tumour volume x', fontsize=12)
ax.set_ylabel('TBR(x)', fontsize=12)
ax.set_title('Treatment Blindness Risk\n(pre-computable)', fontsize=13)
ax.legend(fontsize=10)

ax = axes[1]
vr = x_range_plot >= 0.12
ax.plot(x_range_plot[vr], TIM_analytical[vr], color='darkorange', linewidth=2.5)
ax.axvspan(shared_lo, shared_hi, alpha=0.12, color='gray', label='Shared-state window')
ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
ax.set_xlabel('Tumour volume x', fontsize=12)
ax.set_ylabel('TIM(x)', fontsize=12)
ax.set_title('Trajectory Indeterminacy Measure\nTIM > 1: sign-reversal risk', fontsize=13)
ax.legend(fontsize=10)

ax = axes[2]
ax2 = ax.twinx()
ax.bar(sa_bin_centers, sign_err_bins, width=0.012, color='firebrick', alpha=0.6, label='Sign error rate')
ax2.plot(sa_bin_centers, tbr_bins, color='darkblue', linewidth=2.5, marker='o', markersize=5, label='TBR(x)')
ax.set_xlabel('Tumour volume x', fontsize=12)
ax.set_ylabel('Sign error rate', fontsize=12, color='firebrick')
ax2.set_ylabel('TBR(x)', fontsize=12, color='darkblue')
ax.set_title(f'TBR predicts sign-error regions\nSpearman rho = {rho_tbr:.3f}', fontsize=13)
ax.tick_params(axis='y', labelcolor='firebrick')
ax2.tick_params(axis='y', labelcolor='darkblue')
l1, lb1 = ax.get_legend_handles_labels()
l2, lb2 = ax2.get_legend_handles_labels()
ax.legend(l1+l2, lb1+lb2, fontsize=10, loc='upper left')

plt.tight_layout()
plt.savefig(RUN_DIR / 'figure13_tbr_tim_predeployment.png', dpi=220, bbox_inches='tight')
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

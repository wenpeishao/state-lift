"""
Camera-ready fig1: regret-bound validation WITHOUT per-domain markers.

Identical simulation to gen_fig1_fig5_updated.py (population variance of the
action means, so sigma^2 = SL * Var(R) exactly). The reviewed version placed nine
domains on the curve using probe values, two of which (CaSiNo 0.220, DealOrNoDeal
0.180) were withdrawn, and the probe estimates pooled rather than within-state
state-lift, so no domain is placed on the curve.
"""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from nature_style import apply_style, PALETTE, C_UPPER, C_LOWER, C_SIM, panel_label, save_fig

apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PAPER = os.path.join(HERE, 'out')
os.makedirs(OUT_PAPER, exist_ok=True)

K = 5
np.random.seed(42)
mu = np.sort(np.random.randn(K))[::-1]
var_mu = np.var(mu)
gap_01 = mu[0] - mu[1]

sl_range = np.linspace(0.001, 0.99, 300)
regs_sim, regs_upper, regs_lower = [], [], []
for sl in sl_range:
    sigma = np.sqrt(sl * var_mu / (1 - sl))
    actual_var = var_mu + sigma ** 2
    R_sim = mu + np.random.randn(20000, K) * sigma
    regs_sim.append(np.mean(R_sim.max(axis=1) - R_sim[:, 0]))
    regs_upper.append(np.sqrt(sl * actual_var) * np.sqrt(2 * np.log(K)))
    z = gap_01 / (sigma * np.sqrt(2))
    regs_lower.append(sigma * np.sqrt(2) * norm.pdf(z) - gap_01 * norm.cdf(-z))
sl_range, regs_sim = np.array(sl_range), np.array(regs_sim)
regs_upper, regs_lower = np.array(regs_upper), np.array(regs_lower)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 2.9),
                               gridspec_kw={'width_ratios': [1.2, 1], 'wspace': 0.38})
panel_label(ax1, 'a', x=-0.30, y=1.0)
mask = sl_range <= 0.75
ax1.fill_between(sl_range[mask], regs_lower[mask], regs_upper[mask],
                 alpha=0.08, color=PALETTE['neutral_light'])
ax1.plot(sl_range[mask], regs_upper[mask], '--', color=C_UPPER, lw=1.5, label='Upper bound (Thm 1ii)')
ax1.plot(sl_range[mask], regs_lower[mask], ':', color=C_LOWER, lw=1.5, label='Lower bound (Thm 1iii)')
ax1.plot(sl_range[mask], regs_sim[mask], '-', color=C_SIM, lw=2.0, label=f'Simulated ($K$={K})')
ax1.axvline(0.10, color=PALETTE['neutral_mid'], ls='--', lw=0.8)
ax1.text(0.115, 0.02, r'$\mathrm{SL}_{\mathrm{within}}=0.10$', fontsize=7, color=PALETTE['neutral_dark'], rotation=90, va='bottom')
ax1.set_xlabel(r'$\mathrm{SL}_{\mathrm{within}}$')
ax1.set_ylabel('E[regret per step]')
ax1.legend(loc='upper left', fontsize=6.5, handletextpad=0.4, bbox_to_anchor=(0.16, 1.0))
ax1.set_xlim(-0.02, 0.75)
ax1.set_ylim(-0.02, 0.50)

panel_label(ax2, 'b', x=-0.30, y=1.0)
predictor = np.sqrt(sl_range * (var_mu + sl_range * var_mu / (1 - sl_range)))
ax2.scatter(predictor[mask][::2], regs_sim[mask][::2], s=10, c=C_SIM, alpha=0.4, edgecolors='none', zorder=3)
coeffs = np.polyfit(predictor[mask], regs_sim[mask], 1)
x_fit = np.linspace(0, max(predictor[mask]) * 1.05, 100)
ax2.plot(x_fit, np.polyval(coeffs, x_fit), '--', color=PALETTE['neutral_dark'], lw=0.8, alpha=0.6)
r_corr = np.corrcoef(predictor[mask], regs_sim[mask])[0, 1]
r_full = np.corrcoef(predictor, regs_sim)[0, 1]
ax2.text(0.05, 0.90, f'$r = {r_corr:.2f}$ (plotted range)', transform=ax2.transAxes, fontsize=9, fontweight='bold')
ax2.set_xlabel(r'$\sqrt{\mathrm{SL}_{\mathrm{within}} \cdot \mathrm{Var}(R)}$')
ax2.set_ylabel('Simulated regret')

save_fig(fig, os.path.join(OUT_PAPER, 'fig1_regret_bound'))
print(f'r (SL<=0.75) = {r_corr:.4f}; r (full 0-0.99) = {r_full:.4f}; saved fig1_regret_bound.pdf')

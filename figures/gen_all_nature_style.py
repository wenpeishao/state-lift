"""
Regenerate ALL State-Lift figures following Nature Figure Making skill specifications.
Dense journal-width multi-panel preset: font_size=9, axes_linewidth=1.5.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
import json
import shutil

from nature_style import (apply_style, COLORS, BAR_KW, panel_label, save_fig,
                          C_MATH, C_ESCONV, C_HH, C_TOOL, C_CODE, C_NEG, C_WEB,
                          C_UPPER, C_LOWER, C_SIM, PALETTE)

apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT_PAPER = os.path.join(HERE, 'out')
os.makedirs(OUT_PAPER, exist_ok=True)

# Load robustness results
with open(os.path.join(ROOT, 'results', 'r1_robustness.json')) as f:
    R = json.load(f)


def save_both(fig, name):
    """Save to figures/out/ (svg + pdf + png)."""
    save_fig(fig, os.path.join(OUT_PAPER, name))
    print(f"  -> {os.path.join(OUT_PAPER, name)}.pdf")


# ===================================================================
# FIGURE 1: Regret Bound Validation (2-panel)
# ===================================================================
print('\n[1] Regret bound...')

K = 5
np.random.seed(42)
mu = np.sort(np.random.randn(K))[::-1]
var_mu = np.var(mu)
gap_01 = mu[0] - mu[1]

sl_range = np.linspace(0.001, 0.99, 300)
regs_sim, regs_upper, regs_lower = [], [], []

for sl in sl_range:
    sigma = np.sqrt(sl * var_mu / (1 - sl))
    actual_var = var_mu + sigma**2
    regs = []
    for _ in range(20000):
        R_sim = mu + np.random.randn(K) * sigma
        regs.append(np.max(R_sim) - R_sim[0])
    regs_sim.append(np.mean(regs))
    regs_upper.append(np.sqrt(sl * actual_var) * np.sqrt(2 * np.log(K)))
    if sigma > 0:
        z = gap_01 / (sigma * np.sqrt(2))
        regs_lower.append(sigma * np.sqrt(2) * norm.pdf(z) - gap_01 * norm.cdf(-z))
    else:
        regs_lower.append(0)

sl_range = np.array(sl_range)
regs_sim = np.array(regs_sim)
regs_upper = np.array(regs_upper)
regs_lower = np.array(regs_lower)

C_DOND      = '#FFD700'
C_PROSOCIAL = '#EA84DD'
C_USS       = '#9A4D8E'
C_AIRDIAL   = '#767676'

domains_d1 = [
    ('Math',        0.584, C_MATH),
    ('Tool-Use',    0.398, C_TOOL),
    ('CaSiNo',      0.220, C_NEG),
    ('DealOrNo',    0.180, C_DOND),
    ('ESConv',      0.028, C_ESCONV),
    ('ProsocialD',  0.023, C_PROSOCIAL),
    ('USS',         0.012, C_USS),
    ('Code',        0.003, C_CODE),
    ('AirDialogue', 0.004, C_AIRDIAL),
]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0),
                                gridspec_kw={'width_ratios': [1.2, 1], 'wspace': 0.38})

domain_markers = {
    'Math': 'D', 'Tool-Use': 's', 'CaSiNo': '^', 'DealOrNo': 'h',
    'ESConv': 'v', 'Code': 'o',
    'ProsocialD': 'P', 'USS': 'X', 'AirDialogue': 'd',
}

# Panel a: Bound validation
panel_label(ax1, 'a')
mask = sl_range <= 0.75
ax1.fill_between(sl_range[mask], regs_lower[mask], regs_upper[mask],
                 alpha=0.08, color=PALETTE['neutral_light'])
ax1.plot(sl_range[mask], regs_upper[mask], '--', color=C_UPPER, lw=1.5,
         label='Upper bound (Thm 1ii)')
ax1.plot(sl_range[mask], regs_lower[mask], ':', color=C_LOWER, lw=1.5,
         label='Lower bound (Thm 1iii)')
ax1.plot(sl_range[mask], regs_sim[mask], '-', color=C_SIM, lw=2.0,
         label=f'Simulated ($K$={K})')

# Plot domain markers on curve — use legend instead of inline labels
for name, sl, color in domains_d1:
    idx = np.argmin(np.abs(sl_range - sl))
    reg = regs_sim[idx]
    ax1.plot(sl, reg, domain_markers[name], color=color, ms=7, zorder=5,
             markeredgecolor='black', markeredgewidth=0.6, label=name)

ax1.set_xlabel('State-lift (SL)')
ax1.set_ylabel('E[regret per step]')
# Two-column legend: theory lines + domain markers
ax1.legend(loc='upper left', fontsize=6, ncol=2, columnspacing=0.8,
           handletextpad=0.4)
ax1.set_xlim(-0.02, 0.75)
ax1.set_ylim(-0.02, 0.50)

# Panel b: Prediction accuracy
panel_label(ax2, 'b')
predictor = np.sqrt(sl_range * (var_mu + sl_range * var_mu / (1 - sl_range)))
ax2.scatter(predictor[mask][::2], regs_sim[mask][::2], s=10, c=C_SIM, alpha=0.4,
            edgecolors='none', zorder=3)

coeffs = np.polyfit(predictor[mask], regs_sim[mask], 1)
x_fit = np.linspace(0, max(predictor[mask]) * 1.05, 100)
ax2.plot(x_fit, np.polyval(coeffs, x_fit), '--', color=PALETTE['neutral_dark'],
         lw=0.8, alpha=0.6)

r_corr = np.corrcoef(predictor[mask], regs_sim[mask])[0, 1]
ax2.text(0.05, 0.90, f'$r = {r_corr:.3f}$', transform=ax2.transAxes,
         fontsize=9, fontweight='bold')

for name, sl, color in domains_d1:
    idx = np.argmin(np.abs(sl_range - sl))
    ax2.plot(predictor[idx], regs_sim[idx], domain_markers[name], color=color,
             ms=6, zorder=5, markeredgecolor='black', markeredgewidth=0.5)

ax2.set_xlabel(r'$\sqrt{\mathrm{SL} \cdot \mathrm{Var}(R)}$')
ax2.set_ylabel('Simulated regret')

save_both(fig, 'fig1_regret_bound')


# ===================================================================
# FIGURE 2: Cost Efficiency Scatter
# ===================================================================
print('[2] Cost-efficiency scatter...')

fig, ax = plt.subplots(figsize=(5.2, 3.8))

from matplotlib.lines import Line2D

# Context-correctness = squares, Genuine = circles (original distinction)
# Each domain gets its own color; marker shape encodes label type
all_data = [
    ('Math',     0.584,  0.474, C_MATH,   's'),
    ('Tool-Use', 0.398,  0.503, C_TOOL,   's'),
    ('Code',     0.003,  0.020, C_CODE,   's'),
    ('CaSiNo',   0.220,  0.055, C_NEG,    'o'),
    ('DealOrNo', 0.180,  0.025, C_DOND,   'o'),
    ('ESConv',   0.028,  0.017, C_ESCONV, 'o'),
    ('HH-RLHF', -0.003, 0.034, C_HH,     'o'),
]

for name, sl, lift, color, marker in all_data:
    ax.scatter(sl, lift, c=color, s=80, zorder=5, marker=marker,
               edgecolors='black', linewidths=0.7)

# Build legend with two sections: label type + domains
legend_handles = [
    # Label type
    Line2D([0], [0], marker='s', color='w', markerfacecolor=PALETTE['neutral_light'],
           markeredgecolor='black', markersize=7, linewidth=0, label='Context-correct.'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE['neutral_light'],
           markeredgecolor='black', markersize=7, linewidth=0, label='Genuine quality'),
]
# Separator
legend_handles.append(Line2D([0], [0], color='w', linewidth=0, label=''))
# Domains
for name, _, _, color, marker in all_data:
    legend_handles.append(
        Line2D([0], [0], marker=marker, color='w', markerfacecolor=color,
               markeredgecolor='black', markersize=6, linewidth=0, label=name)
    )

ax.legend(handles=legend_handles, loc='upper left', fontsize=6.5, ncol=3,
          columnspacing=0.5, handletextpad=0.2, labelspacing=0.4)

ax.set_xlabel('State-lift (Ridge regression, minutes)')
ax.set_ylabel('PRM$-$ORM AUC lift (Llama-8B training)')
ax.set_xlim(-0.10, 0.70)
ax.set_ylim(-0.06, 0.60)
ax.axhline(y=0, color=PALETTE['neutral_light'], lw=0.8, zorder=1)

save_both(fig, 'fig2_cost_efficiency')


# ===================================================================
# FIGURE 3: Proxy Pitfall (2-panel)
# ===================================================================
print('[3] Proxy pitfall...')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.2, 2.8),
                                gridspec_kw={'wspace': 0.45})

# Panel a
panel_label(ax1, 'a')
domains_p = ['Math', 'ESConv', 'HH-RLHF', 'Code']
proxy = [79.5, 3.67, 17.2, 10.6]
real = [0.584, 0.028, -0.003, 0.003]

x = np.arange(len(domains_p))
w = 0.32
ax1.bar(x - w/2, proxy, w, label='Proxy SDI',
        color=PALETTE['red_1'], edgecolor='black', linewidth=1.0)
ax1.bar(x + w/2, [r * 100 for r in real], w,
        label='Real SL ($\\times$100)',
        color=PALETTE['blue_secondary'], alpha=0.5,
        edgecolor='black', linewidth=1.0)

ax1.set_xticks(x)
ax1.set_xticklabels(domains_p, fontsize=8)
ax1.set_ylabel('Value')
ax1.set_title('Proxy vs. real state-lift', fontsize=9, pad=4)
ax1.legend(fontsize=7, loc='upper right')

# Subtle bracket: gray text near the false-positive bar, no arrow
ax1.text(2, 20, 'False\npositive', fontsize=6.5, ha='center', va='bottom',
         color=PALETTE['neutral_mid'], style='italic')

# Panel b
panel_label(ax2, 'b')
cats = ['State\npersistence', 'Quality\nstate-dep.']
hh_vals = [0.505, -0.003]
math_vals = [0.002, 0.584]

x2 = np.arange(2)
ax2.bar(x2 - w/2, hh_vals, w, label='HH-RLHF', color=C_HH, **BAR_KW)
ax2.bar(x2 + w/2, math_vals, w, label='Math', color=C_MATH, **BAR_KW)

ax2.set_xticks(x2)
ax2.set_xticklabels(cats, fontsize=8)
ax2.set_ylabel('Value')
ax2.set_title('Persistence vs. quality', fontsize=9, pad=4)
ax2.legend(fontsize=7)
ax2.axhline(y=0, color=PALETTE['neutral_light'], lw=0.8)

save_both(fig, 'fig3_proxy_pitfall')


# ===================================================================
# FIGURE 4: Seven Domains Bar Chart
# ===================================================================
print('[4] Seven domains...')

fig, ax = plt.subplots(figsize=(7.0, 2.8))

domains_7 = [
    ('Math\nreasoning',    0.584, C_MATH),
    ('Tool-use\nagents',   0.398, C_TOOL),
    ('CaSiNo\nneg.',       0.220, C_NEG),
    ('DealOrNo\nneg.',     0.180, C_DOND),
    ('Emot.\nsupport',     0.028, C_ESCONV),
    ('Prosocial\nDialog',  0.023, C_PROSOCIAL),
    ('USS\nSatisf.',       0.012, C_USS),
    ('Code\nreasoning',    0.003, C_CODE),
    ('Air\nDialogue',      0.004, C_AIRDIAL),
    ('Chat QA',           -0.003, C_HH),
    ('Web\nnav.',         -0.003, C_WEB),
]

names = [d[0] for d in domains_7]
vals = [d[1] for d in domains_7]
colors = [d[2] for d in domains_7]

bars = ax.bar(range(len(domains_7)), vals, color=colors, width=0.65, **BAR_KW)

for bar, v in zip(bars, vals):
    y_pos = max(v, 0) + 0.012
    ax.text(bar.get_x() + bar.get_width()/2, y_pos, f'{v:.3f}',
            ha='center', va='bottom', fontsize=7, fontweight='bold')

ax.set_xticks(range(len(domains_7)))
ax.set_xticklabels(names, fontsize=7)
ax.set_ylabel('State-lift')
ax.axhline(y=0, color='black', lw=0.5)
ax.axhline(y=0.10, color=PALETTE['neutral_mid'], ls='--', lw=0.8)
ax.text(len(domains_7)-0.5, 0.11, 'SL = 0.10', fontsize=7, ha='right',
        color=PALETTE['neutral_mid'], style='italic')

save_both(fig, 'fig4_ten_domains')


# ===================================================================
# FIGURE 5: Effective Gap Regimes
# ===================================================================
print('[5] Effective gap...')

import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(7.0, 4.2))
gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 12], hspace=0.0)

ax_strip = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1])

eff_gaps = np.linspace(0.1, 18, 300)
sigma = 1.0
regrets = []
for eg in eff_gaps:
    gap_val = eg * sigma
    z = gap_val / (sigma * np.sqrt(2))
    reg = sigma * np.sqrt(2) * norm.pdf(z) - gap_val * norm.cdf(-z)
    regrets.append(reg)

ax.plot(eff_gaps, regrets, '-', color=PALETTE['neutral_black'], lw=2.0)

# Subtle background shading in main plot (very light, no text)
ax.axvspan(0, 1, alpha=0.04, color=PALETTE['red_strong'])
ax.axvspan(1, 3, alpha=0.03, color=PALETTE['gold'])
ax.axvspan(3, 15, alpha=0.02, color=PALETTE['green_3'])
ax.axvline(x=1, color=PALETTE['neutral_light'], ls='--', lw=0.4)
ax.axvline(x=3, color=PALETTE['neutral_light'], ls='--', lw=0.4)

# Formal colored strip at top — regime bar with labels fully inside
ax_strip.barh(0, 1, left=0, height=1, color=PALETTE['red_strong'], alpha=0.30,
              edgecolor=PALETTE['neutral_mid'], linewidth=0.5)
ax_strip.barh(0, 2, left=1, height=1, color=PALETTE['gold'], alpha=0.25,
              edgecolor=PALETTE['neutral_mid'], linewidth=0.5)
ax_strip.barh(0, 15, left=3, height=1, color=PALETTE['green_3'], alpha=0.22,
              edgecolor=PALETTE['neutral_mid'], linewidth=0.5)

# Labels inside each segment — uniform small font, vertically centered
ax_strip.text(0.5, 0.15, 'PRM ess.', fontsize=5, ha='center', va='center',
              fontweight='bold', color=PALETTE['neutral_dark'])
ax_strip.text(2.0, 0.15, 'Hybrid', fontsize=5, ha='center', va='center',
              fontweight='bold', color=PALETTE['neutral_dark'])
ax_strip.text(10.0, 0.15, 'ORM suff.', fontsize=5, ha='center', va='center',
              fontweight='bold', color=PALETTE['neutral_dark'])

ax_strip.set_xlim(0, 18)
ax_strip.set_ylim(0, 1)
ax_strip.set_xticks([])
ax_strip.set_yticks([])
for spine in ax_strip.spines.values():
    spine.set_visible(False)

domain_egs = [
    ('Math',        1.11,  C_MATH,      'D'),
    ('Tool',        1.23,  C_TOOL,      's'),
    ('CaSiNo',      1.88,  C_NEG,       '^'),
    ('DealOrNo',    2.14,  C_DOND,      'h'),
    ('ESConv',      5.79,  C_ESCONV,    'v'),
    ('ProsocialD',  6.50,  C_PROSOCIAL, 'P'),
    ('USS',         9.27,  C_USS,       'X'),
    ('Code',       11.91,  C_CODE,      'o'),
    ('AirDialogue',15.82,  C_AIRDIAL,   'd'),
]

for name, eg, color, marker in domain_egs:
    idx = np.argmin(np.abs(np.array(eff_gaps) - eg))
    reg = regrets[idx]
    ax.plot(eg, reg, marker, color=color, ms=7, zorder=5,
            markeredgecolor='black', markeredgewidth=0.6, label=name)

ax.legend(fontsize=8, loc='center right',
          bbox_to_anchor=(0.98, 0.62), handletextpad=0.3)

ax.set_xlabel('Effective gap = $\\sqrt{(1 - \\mathrm{SL}) / \\mathrm{SL}}$')
ax.set_ylabel('E[regret] (2-action bound)')
ax.set_xlim(0, 18)
ymax = max(regrets)
ax.set_ylim(-0.02, ymax * 1.08)

save_both(fig, 'fig5_effective_gap')


# ===================================================================
# FIGURE 6: Artifact Checks (shuffle control)
# ===================================================================
print('[6] Artifact checks...')

fig, ax = plt.subplots(figsize=(4.8, 2.8))

doms_a = ['Math', 'MathDial', 'Code', 'ESConv']
real_v = [0.584, 0.114, 0.010, 0.028]
shuf_v = [-0.013, 0.003, 0.002, 0.006]
drops = [100, 97, 79, 79]

x = np.arange(len(doms_a))
w = 0.30

bars_r = ax.bar(x - w/2, real_v, w, label='Real',
                color=PALETTE['teal'], **BAR_KW)
bars_s = ax.bar(x + w/2, shuf_v, w, label='Shuffled',
                color=PALETTE['neutral_light'], **BAR_KW)

for i, drop in enumerate(drops):
    y_pos = real_v[i] + 0.015
    ax.text(x[i], y_pos, f'$-${drop}%',
            fontsize=7, fontweight='bold', color=PALETTE['neutral_dark'],
            ha='center', va='bottom')

ax.set_xticks(x)
ax.set_xticklabels(doms_a, fontsize=8)
ax.set_ylabel('State-lift')
ax.legend(fontsize=8, loc='upper right')
ax.axhline(y=0, color='black', lw=0.5)
ax.set_ylim(-0.05, 0.72)

save_both(fig, 'fig6_artifact_checks')


# ===================================================================
# FIGURE 7: Sample Size Sensitivity (3-panel)
# ===================================================================
print('[7] Sample size...')

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.5), sharey=False)
plt.subplots_adjust(wspace=0.38)

for ax, name, color in zip(axes, ['Math', 'ESConv', 'HH-RLHF'],
                            [C_MATH, C_ESCONV, C_HH]):
    ns, means, ci_los, ci_his = [], [], [], []
    for key, val in R.items():
        if key.startswith(f'sample_size_{name}_n'):
            n = int(key.split('_n')[-1])
            ns.append(n)
            means.append(val['mean'])
            ci_los.append(val['ci_lo'])
            ci_his.append(val['ci_hi'])
    if ns:
        ns, means, ci_los, ci_his = zip(*sorted(zip(ns, means, ci_los, ci_his)))
        ns = np.array(ns); means = np.array(means)
        ci_los = np.array(ci_los); ci_his = np.array(ci_his)
        ax.plot(ns, means, 'o-', color=color, lw=1.8, ms=4,
                markeredgecolor='black', markeredgewidth=0.5)
        ax.fill_between(ns, ci_los, ci_his, alpha=0.15, color=color)
        ax.axhline(y=0, color=PALETTE['neutral_light'], lw=0.6, ls='--')
    ax.set_xlabel('$n$', fontsize=9)
    ax.set_title(name, fontsize=9, fontweight='bold')
    ax.set_xscale('log')

axes[0].set_ylabel('State-lift')

save_both(fig, 'fig7_sample_size')


# ===================================================================
# FIGURE 8: Encoder Sensitivity
# ===================================================================
print('[8] Encoder sensitivity...')

fig, ax = plt.subplots(figsize=(4.5, 2.8))

enc_data = R.get('encoder_sensitivity', {})
encoders = [e for e in enc_data if 'error' not in enc_data[e]]
domains_e = ['Math', 'ESConv', 'HH-RLHF']
colors_e = [C_MATH, C_ESCONV, C_HH]

x = np.arange(len(encoders))
w = 0.22

for i, (dom, col) in enumerate(zip(domains_e, colors_e)):
    vals = [enc_data[e].get(dom, 0) for e in encoders]
    ax.bar(x + i * w, vals, w, label=dom, color=col, **BAR_KW)

ax.set_xticks(x + w)
ax.set_xticklabels(encoders, fontsize=8)
ax.set_ylabel('State-lift')
ax.legend(fontsize=7)
ax.axhline(y=0, color='black', lw=0.5)

save_both(fig, 'fig8_encoder_sensitivity')


# ===================================================================
# FIGURE 9: Persistence Controls
# ===================================================================
print('[9] Persistence controls...')

fig, ax = plt.subplots(figsize=(5.2, 3.0))

doms_p = ['Math', 'ESConv', 'HH-RLHF']
methods = ['standard_sl', 'lagged_sl', 'block_shuffle_sl', 'residualized_sl']
labels_m = ['Standard', 'Lagged $z_{t-1}$', 'Block shuffle', 'PC0 removed']
colors_m = [PALETTE['neutral_dark'], C_MATH, C_TOOL, C_CODE]

x = np.arange(len(doms_p))
w = 0.18

for i, (method, label, col) in enumerate(zip(methods, labels_m, colors_m)):
    vals = [R.get(f'persistence_{d}', {}).get(method, 0) for d in doms_p]
    ax.bar(x + i * w, vals, w, label=label, color=col, **BAR_KW)

ax.set_xticks(x + w * 1.5)
ax.set_xticklabels(doms_p, fontsize=8)
ax.set_ylabel('State-lift')
ax.legend(fontsize=7, ncol=2)
ax.axhline(y=0, color='black', lw=0.5)

save_both(fig, 'fig9_persistence')


# ===================================================================
# FIGURE 10: Nonlinear Estimators
# ===================================================================
print('[10] Nonlinear estimators...')

fig, ax = plt.subplots(figsize=(4.5, 2.8))

doms_n = ['Math', 'ESConv', 'HH-RLHF']
meths = ['sl_linear', 'sl_kernel', 'sl_mlp']
labs = ['Linear Ridge', 'Kernel Ridge', 'MLP']
cols = [C_CODE, C_TOOL, C_ESCONV]

x = np.arange(len(doms_n))
w = 0.22

for i, (m, l, co) in enumerate(zip(meths, labs, cols)):
    vals = [R.get(f'nonlinear_{d}', {}).get(m, 0) for d in doms_n]
    ax.bar(x + i * w, vals, w, label=l, color=co, **BAR_KW)

ax.set_xticks(x + w)
ax.set_xticklabels(doms_n, fontsize=8)
ax.set_ylabel('State-lift')
ax.legend(fontsize=7)
ax.axhline(y=0, color='black', lw=0.5)

save_both(fig, 'fig10_nonlinear')


# ===================================================================
# FIGURE R11: Resolution Experiment (PRM800K)
# ===================================================================
print('[R11] Resolution experiment...')

fig, ax = plt.subplots(figsize=(5.5, 3.2))

methods_r11 = [
    'Linear Ridge\n(MiniLM)',
    'Kernel Ridge\n(MiniLM)',
    'MLP\n(MiniLM)',
    'Linear Ridge\n(Llama-8B)',
    'LLM Training\nLift',
]
sl_vals = [0.071, 0.071, 0.071, 0.046, 0.238]

# Encode-separately = blue family, end-to-end = red/hero
bar_colors = [PALETTE['blue_main']]*3 + [PALETTE['blue_secondary']] + [C_MATH]

x = np.arange(len(methods_r11))
bars = ax.bar(x, sl_vals, color=bar_colors, width=0.58, **BAR_KW)

# Hatching on the LLM bar for emphasis
bars[4].set_hatch('///')

for bar, v in zip(bars, sl_vals):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.006, f'{v:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(methods_r11, fontsize=7)
ax.set_ylabel('Quality signal (AUC lift)')
ax.axhline(y=0, color='black', lw=0.5)

# Dashed line at SL = 0.05 — label placed just left of the red bar
ax.axhline(y=0.05, color=PALETTE['neutral_mid'], ls='--', lw=0.6)
ax.text(4.55, 0.038, 'SL = 0.05', fontsize=6.5,
        color=PALETTE['neutral_mid'], ha='left', va='top')

# Legend
ax.bar([], [], color=PALETTE['blue_main'], label='Encode-separately', **BAR_KW)
ax.bar([], [], color=C_MATH, hatch='///', label='End-to-end (joint attention)', **BAR_KW)
ax.legend(fontsize=7, loc='upper left')

# Interaction-level signal annotation — subtle gray bracket, right of bar
ax.annotate('', xy=(4.35, 0.058), xytext=(4.35, 0.232),
            arrowprops=dict(arrowstyle='<->', color=PALETTE['neutral_mid'], lw=0.8))
ax.text(4.55, 0.14, 'Interaction-\nlevel signal', fontsize=6.5,
        color=PALETTE['neutral_dark'], ha='left', style='italic')

ax.set_ylim(-0.02, 0.28)
ax.set_xlim(-0.5, 4.8)

save_both(fig, 'fig_r11_resolution')


print('\n=== All figures regenerated (Nature style) ===')

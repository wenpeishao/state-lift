"""
Camera-ready fig5: effective-gap regimes WITHOUT per-domain markers.

The reviewed version placed CaSiNo (SL 0.22) and DealOrNoDeal (SL 0.18) on this curve;
both values were withdrawn (context-correctness construction / structure proxy), and the
x-axis is now defined on SL_within (Theorem 1 as restated with the level term m(s)).
Same style and curve as gen_fig1_fig5_updated.py; only the markers/legend are removed and
the 0.10 threshold (EffGap = 3) is annotated.
"""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from nature_style import apply_style, PALETTE, save_fig

apply_style()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PAPER = os.path.join(HERE, 'out')
os.makedirs(OUT_PAPER, exist_ok=True)

fig = plt.figure(figsize=(7.0, 3.6))
gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 12], hspace=0.0)
ax_strip = fig.add_subplot(gs[0])
ax = fig.add_subplot(gs[1])

eff_gaps = np.linspace(0.02, 6, 600)
sigma = 1.0
# Two actions: std(r_blind) = Delta/2, so EffGap = Delta/(2 sigma)  =>  Delta = 2 * EffGap * sigma.
# (The reviewed figure used Delta = EffGap * sigma, which overstates regret at a given EffGap.)
delta = 2 * eff_gaps * sigma
z = delta / (sigma * np.sqrt(2))
regrets = sigma * np.sqrt(2) * norm.pdf(z) - delta * norm.cdf(-z)
ax.plot(eff_gaps, regrets, '-', color=PALETTE['neutral_black'], lw=2.0)

ax.axvspan(0, 1, alpha=0.04, color=PALETTE['red_strong'])
ax.axvspan(1, 3, alpha=0.03, color=PALETTE['gold'])
ax.axvspan(3, 6, alpha=0.02, color=PALETTE['green_3'])
ax.axvline(x=1, color=PALETTE['neutral_light'], ls='--', lw=0.4)
ax.axvline(x=3, color=PALETTE['neutral_mid'], ls='--', lw=0.8)

ceiling = sigma * np.sqrt(2) * norm.pdf(0)
r3 = float(np.interp(3.0, eff_gaps, regrets))
ax.annotate(f'EffGap $=3$  $\\Leftrightarrow$  $\\mathrm{{SL}}_{{\\mathrm{{within}}}}=0.10$\n'
            f'($\\Delta = 6\\sigma$; regret $< 10^{{-5}}$ of noise-dominant ceiling)',
            xy=(3, r3), xytext=(3.3, 0.35 * ceiling), fontsize=8,
            color=PALETTE['neutral_dark'],
            arrowprops=dict(arrowstyle='->', lw=0.6, color=PALETTE['neutral_mid']))

ax_strip.barh(0, 1, left=0, height=1, color=PALETTE['red_strong'], alpha=0.30,
              edgecolor=PALETTE['neutral_mid'], linewidth=0.5)
ax_strip.barh(0, 2, left=1, height=1, color=PALETTE['gold'], alpha=0.25,
              edgecolor=PALETTE['neutral_mid'], linewidth=0.5)
ax_strip.barh(0, 3, left=3, height=1, color=PALETTE['green_3'], alpha=0.22,
              edgecolor=PALETTE['neutral_mid'], linewidth=0.5)
for x, s in [(0.5, 'Cond. ess.'), (2.0, 'Hybrid'), (4.5, 'Blind suff.')]:
    ax_strip.text(x, 0.15, s, fontsize=7, ha='center', va='center',
                  fontweight='bold', color=PALETTE['neutral_dark'])
ax_strip.set_xlim(0, 6); ax_strip.set_ylim(0, 1)
ax_strip.set_xticks([]); ax_strip.set_yticks([])
for sp in ax_strip.spines.values():
    sp.set_visible(False)

ax.set_xlabel('Effective gap $= \\sqrt{(1 - \\mathrm{SL}_{\\mathrm{within}}) / \\mathrm{SL}_{\\mathrm{within}}}$')
ax.set_ylabel('E[regret] (2 actions, $\\sigma = 1$)')
ax.set_xlim(0, 6)
ax.set_ylim(-0.02, regrets.max() * 1.08)

save_fig(fig, os.path.join(OUT_PAPER, 'fig5_effective_gap'))
print(f'regret at EffGap=3: {r3:.4f} = {100 * r3 / ceiling:.2f}% of ceiling {ceiling:.4f}')
print('saved fig5_effective_gap.pdf -> figures/out')

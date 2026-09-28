"""
Figure 3: Proxy pitfall (2-panel).
(a) Proxy SDI vs real state-lift bars.
(b) Persistence vs quality scatter (HH-RLHF vs Math).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from paper_plot_style import apply_style, COLORS, BAR_KW, panel_label, save_fig

apply_style()

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out')
os.makedirs(OUT, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.5, 2.5),
                                gridspec_kw={'wspace': 0.4})

# ---- Panel (a): Proxy vs Real ----
ax1.set_title('Proxy vs. real state-lift', fontsize=9)
ax1.text(-0.30, 1.0, 'a', transform=ax1.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

domains_p = ['Math', 'ESConv', 'HH-RLHF', 'Code']
proxy = [79.5, 3.67, 17.2, 10.6]
real = [0.584, 0.028, -0.003, 0.003]

x = np.arange(len(domains_p))
w = 0.32
ax1.bar(x - w/2, proxy, w, label='Proxy SDI',
        color='#ffcccc', edgecolor='#cc0000', lw=0.5)
ax1.bar(x + w/2, [r * 100 for r in real], w,
        label='Real SL ($\\times$100)',
        color='#cce5ff', edgecolor='#0066cc', lw=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels(domains_p, fontsize=7)
ax1.set_ylabel('Value')
ax1.legend(fontsize=7, loc='upper right', frameon=False)

# Mark the false positive with a small label directly above the bar pair (no arrow)
ax1.text(2, 17.2 + 3.0, 'false\npositive', ha='center', va='bottom', fontsize=6.2,
         color='#333333', style='italic', linespacing=1.1)

# ---- Panel (b): Persistence vs Quality ----
ax2.set_title('Persistence vs. quality', fontsize=9)
ax2.text(-0.30, 1.0, 'b', transform=ax2.transAxes, fontsize=12, fontweight='bold', va='top', ha='left')

cats = ['State\npersistence', 'Quality\nstate-dep.']
hh_vals = [0.505, -0.003]
math_vals = [0.002, 0.584]

x2 = np.arange(2)
ax2.bar(x2 - w/2, hh_vals, w, label='HH-RLHF',
        color=COLORS['hh'], **BAR_KW)
ax2.bar(x2 + w/2, math_vals, w, label='Math',
        color=COLORS['math'], **BAR_KW)

ax2.set_xticks(x2)
ax2.set_xticklabels(cats, fontsize=7)
ax2.set_ylabel('Value')
ax2.legend(fontsize=7, frameon=False)
ax2.axhline(y=0, color='gray', lw=0.4)

save_fig(fig, os.path.join(OUT, 'fig3_proxy_pitfall'))
print("[3] Done: fig3_proxy_pitfall")

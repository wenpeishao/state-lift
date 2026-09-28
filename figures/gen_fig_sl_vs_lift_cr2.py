"""
Camera-ready replacement for the reviewed Figure 4 (cost-efficiency scatter), v2.

Same encoding as the reviewed figure: marker SHAPE = label type (square = context-
correctness, circle = genuine label, diamond = constructed verifier), marker COLOR =
domain, one legend entry per domain, no in-plot text (nothing to overlap). Hollow
markers = inadmissible (labels never vary within a state). Panel (a) is the full
range including the context-correctness domains; panel (b) zooms on the genuine-label
cluster and adds seed SDs. Values are those of Table 4.
"""
import sys, os, shutil
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from nature_style import (apply_style, PALETTE, C_MATH, C_TOOL, C_CODE, C_NEG, C_ESCONV, C_HH,
                          panel_label, save_fig)

apply_style()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PAPER = os.path.join(HERE, 'out')
os.makedirs(OUT_PAPER, exist_ok=True)

C_DOND = PALETTE['gold']; C_P4G = PALETTE['magenta']; C_CRAIG = PALETTE['neutral_mid']
C_ARITH = '#2E7D32'; C_CONS = PALETTE['green_3']

# name, SL, lift, sd, shape, color, admissible
rows = [
    ('Math',               0.584, 0.474, None,  's', C_MATH,   True),
    ('Tool-Use',           0.398, 0.503, None,  's', C_TOOL,   True),
    ('Code',               0.003, 0.020, None,  's', C_CODE,   True),
    ('HH-RLHF',           -0.003, 0.030, 0.012, 'o', C_HH,     True),
    ('ESConv',             0.028, 0.008, 0.009, 'o', C_ESCONV, True),
    ('DealOrNoDeal',       0.057, 0.039, 0.013, 'o', C_DOND,   False),
    ('CaSiNo',            -0.003, 0.027, 0.022, 'o', C_NEG,    False),
    ('PersuasionForGood', -0.001, 0.088, 0.025, 'o', C_P4G,    False),
    ('CraigslistBargain',  0.003, 0.089, 0.032, 'o', C_CRAIG,  False),
    ('Arithmetic',         0.010, 0.343, 0.080, 'D', C_ARITH,  True),
    ('Constraints',       -0.005, 0.345, 0.012, 'D', C_CONS,   True),
]

def draw(ax, zoom):
    for name, sl, lift, sd, mk, col, adm in rows:
        if zoom and mk == 's':
            continue
        fc = col if adm else 'white'
        ec = 'black' if adm else col
        if zoom and sd is not None:
            ax.errorbar(sl, lift, yerr=sd, fmt='none', ecolor=col, elinewidth=0.9, capsize=2, zorder=4)
        dx = 0.0
        if zoom and name == 'CaSiNo': dx = -0.0025
        if zoom and name == 'HH-RLHF': dx = 0.0025
        ax.plot(sl + dx, lift, mk, ms=8 if mk != 'D' else 7, mfc=fc, mec=ec, mew=0.6 if adm else 1.1, zorder=5)
    ax.axhline(0, color=PALETTE['neutral_light'], lw=0.6, zorder=1)
    ax.axvline(0.10, color=PALETTE['neutral_mid'], ls='--', lw=0.8, zorder=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.6, 2.7), gridspec_kw={'width_ratios': [1, 1.1], 'wspace': 0.30})
fig.subplots_adjust(left=0.08, right=0.80, top=0.95, bottom=0.17)

panel_label(ax1, 'a', x=-0.30, y=1.0)
draw(ax1, zoom=False)
ax1.set_xlim(-0.05, 0.70); ax1.set_ylim(-0.03, 0.60)
ax1.set_xlabel('State-lift (Ridge regression, minutes)')
ax1.set_ylabel('PRM$-$ORM lift (Llama-8B training)')
ax1.text(0.115, 0.43, 'threshold\n0.10', fontsize=6.5, color=PALETTE['neutral_dark'], va='top')
ax1.add_patch(Rectangle((-0.03, -0.02), 0.12, 0.40, fill=False, ec=PALETTE['neutral_mid'], lw=0.6, ls=':'))
ax1.text(0.095, 0.385, 'panel b', fontsize=6.5, color=PALETTE['neutral_mid'], ha='right', va='bottom')

panel_label(ax2, 'b', x=-0.30, y=1.0)
draw(ax2, zoom=True)
ax2.set_xlim(-0.03, 0.09); ax2.set_ylim(-0.02, 0.40)
ax2.set_xlabel('State-lift (genuine and constructed labels)')
ax2.set_ylabel('PRM$-$ORM lift, mean $\\pm$ SD over seeds')

shape_handles = [
    Line2D([], [], marker='s', ls='', mfc=PALETTE['neutral_light'], mec='black', ms=7, label='Context-correct.'),
    Line2D([], [], marker='o', ls='', mfc=PALETTE['neutral_light'], mec='black', ms=7, label='Genuine label'),
    Line2D([], [], marker='D', ls='', mfc=PALETTE['neutral_light'], mec='black', ms=6, label='Constructed verifier'),
    Line2D([], [], marker='o', ls='', mfc='white', mec=PALETTE['neutral_dark'], ms=7, label='hollow: inadmissible'),
]
dom_handles = [Line2D([], [], marker=mk, ls='', mfc=(col if adm else 'white'), mec=('black' if adm else col),
                      ms=7 if mk != 'D' else 6, label=name) for name, _, _, _, mk, col, adm in rows]
leg1 = ax1.legend(handles=shape_handles, loc='center right', bbox_to_anchor=(0.99, 0.42), fontsize=6.4, frameon=False, handletextpad=0.35, borderaxespad=0.0, labelspacing=0.5)
ax1.add_artist(leg1)
ax2.legend(handles=dom_handles, loc='center left', bbox_to_anchor=(1.03, 0.5), fontsize=6.6, frameon=False, ncol=1, handletextpad=0.4,
           labelspacing=0.55, borderaxespad=0.0, title='Domain', title_fontsize=6.8)

save_fig(fig, os.path.join(OUT_PAPER, 'fig_sl_vs_lift_cr'))
print('saved fig_sl_vs_lift_cr.pdf')

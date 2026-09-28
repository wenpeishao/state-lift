"""
Shared plot style for State-Lift NeurIPS paper figures.
Matches the approved 4-panel robustness figure style exactly.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- RC params matching approved style ----
STYLE = {
    'font.family': 'serif',
    'font.size': 8,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'lines.linewidth': 1.5,
}

def apply_style():
    """Apply the approved NeurIPS paper style to matplotlib."""
    plt.rcParams.update(STYLE)

# ---- Okabe-Ito colorblind-safe palette ----
C_MATH = '#D55E00'    # vermillion
C_ESCONV = '#009E73'  # green
C_HH = '#CC79A7'      # pink
C_TOOL = '#E69F00'    # orange
C_CODE = '#0072B2'    # blue
C_NEG = '#56B4E9'     # light blue (negotiation)
C_WEB = '#999999'     # gray (web navigation)

# Bound / simulation colors
C_UPPER = '#0072B2'   # blue
C_LOWER = '#E69F00'   # orange
C_SIM = '#009E73'     # green

# Convenience dict
COLORS = {
    'math': C_MATH,
    'esconv': C_ESCONV,
    'hh': C_HH,
    'tool': C_TOOL,
    'code': C_CODE,
    'neg': C_NEG,
    'web': C_WEB,
    'upper': C_UPPER,
    'lower': C_LOWER,
    'sim': C_SIM,
}

# ---- Standard bar kwargs ----
BAR_KW = dict(edgecolor='k', lw=0.3)

# ---- Panel label helper ----
def panel_label(ax, label, title=''):
    """Add bold panel label like '(a) Title' to axis."""
    text = f'({label}) {title}' if title else f'({label})'
    ax.set_title(text, loc='left', fontsize=9, fontweight='bold')

# ---- Save helper ----
def save_fig(fig, path_stem, dpi=300):
    """Save figure as both PDF and PNG."""
    fig.savefig(f'{path_stem}.pdf', dpi=dpi)
    fig.savefig(f'{path_stem}.png', dpi=dpi)
    plt.close(fig)
    print(f"  Saved: {path_stem}.pdf / .png")

"""
Nature-style plot aesthetics for State-Lift paper figures.
Following the Nature Figure Making skill specifications exactly.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ---- Nature PALETTE (from skill spec) ----
PALETTE = {
    "blue_main":      "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE",
    "green_2": "#AADCA9",
    "green_3": "#8BCF8B",
    "red_1":   "#F6CFCB",
    "red_2":   "#E9A6A1",
    "red_strong": "#B64342",
    "neutral_light": "#CFCECE",
    "neutral_mid":   "#767676",
    "neutral_dark":  "#4D4D4D",
    "neutral_black": "#272727",
    "gold":   "#FFD700",
    "teal":   "#42949E",
    "violet": "#9A4D8E",
    "magenta":"#EA84DD",
}

# Domain-specific semantic colors (kept distinct for 7-domain story)
C_MATH   = '#B64342'    # red_strong — high state-lift, hero
C_TOOL   = '#E69F00'    # amber/gold — high SL
C_NEG    = '#3775BA'    # blue_secondary — moderate SL
C_ESCONV = '#42949E'    # teal — low SL
C_CODE   = '#0F4D92'    # blue_main — near-zero SL
C_HH     = '#9A4D8E'    # violet — near-zero SL
C_WEB    = '#767676'    # neutral_mid — negative SL

# Bound / simulation colors
C_UPPER = '#0F4D92'     # blue_main
C_LOWER = '#E69F00'     # gold/amber
C_SIM   = '#42949E'     # teal

COLORS = {
    'math': C_MATH, 'esconv': C_ESCONV, 'hh': C_HH,
    'tool': C_TOOL, 'code': C_CODE, 'neg': C_NEG, 'web': C_WEB,
    'upper': C_UPPER, 'lower': C_LOWER, 'sim': C_SIM,
}


def apply_style(font_size=9, axes_linewidth=1.5):
    """Apply Nature-style rcParams (dense journal-width multi-panel preset)."""
    # MANDATORY: editable SVG text
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    plt.rcParams['svg.fonttype'] = 'none'
    # Layout & style
    plt.rcParams['font.size'] = font_size
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.linewidth'] = axes_linewidth
    plt.rcParams['legend.frameon'] = False
    # Additional
    plt.rcParams['axes.labelsize'] = font_size + 1
    plt.rcParams['axes.titlesize'] = font_size + 1
    plt.rcParams['xtick.labelsize'] = font_size - 1
    plt.rcParams['ytick.labelsize'] = font_size - 1
    plt.rcParams['legend.fontsize'] = font_size - 1
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.08
    plt.rcParams['xtick.major.width'] = axes_linewidth * 0.6
    plt.rcParams['ytick.major.width'] = axes_linewidth * 0.6
    plt.rcParams['xtick.major.size'] = 4
    plt.rcParams['ytick.major.size'] = 4
    plt.rcParams['lines.linewidth'] = 1.8
    plt.rcParams['lines.markersize'] = 6


# ---- Standard bar kwargs (Nature: black edge, 1.5 lw) ----
BAR_KW = dict(edgecolor='black', linewidth=1.0)


def panel_label(ax, label, x=-0.08, y=1.05, fontsize=12):
    """Place a Nature-style bold lowercase panel label near top-left."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=fontsize, fontweight='bold', va='top', ha='left',
            fontfamily='sans-serif', color='black')


def save_fig(fig, path_stem, dpi=300):
    """Save as SVG (primary), PDF, and PNG."""
    fig.tight_layout(pad=1.5)
    fig.savefig(f'{path_stem}.svg', bbox_inches='tight')
    fig.savefig(f'{path_stem}.pdf', dpi=dpi, bbox_inches='tight')
    fig.savefig(f'{path_stem}.png', dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path_stem} (.svg/.pdf/.png)")

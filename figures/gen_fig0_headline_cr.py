"""
Camera-ready Figure 1 = the reviewed headline figure (fig0_headline.png) with minimal
in-place edits; everything else is pixel-identical:
  * green box: drop "Negotiation (0.22)" (withdrawn value)
  * lower branch: "SL < 0.10 -> ORM is Sufficient" replaced by "SL ~ 0 -> Escalate"
    (the two-branch rule was withdrawn in the author response)
  * bar chart: corrected registry values (Negotiation 0.22 withdrawn; DoND 0.057)
Regions are found by colour segmentation so no hand-measured pixel coordinates are needed.
"""
import os, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib import font_manager
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'source', 'fig0_headline.png')   # base PNG drawn with an external tool
OUT_PAPER = os.path.join(HERE, 'out'); os.makedirs(OUT_PAPER, exist_ok=True)
FAM = 'Arial' if any('Arial' in f.name for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family': FAM})

img = mpimg.imread(SRC)[:, :, :3].astype(float)
Hpx, Wpx = img.shape[:2]

def region(seed_xy, tol=0.06):
    """bbox + fill colour of the flat-colour region containing seed (x, y)."""
    sx, sy = seed_xy
    c = img[sy, sx]
    mask = np.linalg.norm(img - c, axis=2) < tol
    lab, _ = ndimage.label(mask)
    m = lab == lab[sy, sx]
    ys, xs = np.where(m)
    return (xs.min(), ys.min(), xs.max(), ys.max()), tuple(c)

# seeds: a blank spot inside each region (fractions of image size, from the reviewed figure's layout)
green_bb, GREEN = region((int(Wpx * 0.955), int(Hpx * 0.20)))     # top-right green box
gray_bb, GRAY = region((int(Wpx * 0.885), int(Hpx * 0.66)))       # lower-right gray box
panel_bb, PANEL = region((int(Wpx * 0.76), int(Hpx * 0.62)))      # orange decision panel bg
bar_bb, BARBG = (1640, 965, 3140, 1315), tuple(img[1300, 1700])   # bar-chart panel (bg gradient defeats segmentation)
print('green', green_bb, 'gray', gray_bb, 'panel', panel_bb, 'bar', bar_bb)

DPI = 300
fig = plt.figure(figsize=(Wpx / DPI, Hpx / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(img); ax.set_xlim(0, Wpx); ax.set_ylim(Hpx, 0); ax.axis('off')
DARK = '#1a1a1a'
def cover(bb, color, inset=6, top=0, rounded=False):
    x0, y0, x1, y1 = bb
    if rounded:  # match the rounded-rectangle boxes so the cover never pokes into their corners
        ax.add_patch(FancyBboxPatch((x0 + inset, y0 + inset + top), x1 - x0 - 2 * inset, y1 - y0 - 2 * inset - top,
                                    boxstyle='round,pad=0,rounding_size=34', fc=color, ec='none', zorder=3))
    else:
        ax.add_patch(Rectangle((x0 + inset, y0 + inset + top), x1 - x0 - 2 * inset, y1 - y0 - 2 * inset - top, fc=color, ec='none', zorder=3))
def text(x, y, s, fs, bold=False, color=DARK, ha='center', style='normal', ls=1.2):
    ax.text(x, y, s, fontsize=fs, fontweight='bold' if bold else 'normal', color=color, ha=ha, va='center', style=style, linespacing=ls, zorder=4)
def cx(bb): return (bb[0] + bb[2]) / 2

# ---- green box: keep title row, replace the example list below it ----
gx0, gy0, gx1, gy1 = green_bb; gh = gy1 - gy0
cover((gx0, gy0 + int(0.34 * gh), gx1, gy1), GREEN, inset=10, rounded=True)
text(cx(green_bb), gy0 + 0.66 * gh, 'e.g., Math (0.58),\nTool-use (0.40)', 9.4, ls=1.3)

# ---- gray box: replace entirely ----
kx0, ky0, kx1, ky1 = gray_bb; kh = ky1 - ky0
cover(gray_bb, GRAY, inset=10, rounded=True)
text(cx(gray_bb), ky0 + 0.29 * kh, 'Escalate,\nDo Not Conclude', 8.6, bold=True, ls=1.1)
text(cx(gray_bb), ky0 + 0.71 * kh, 'truncated training:\n~10% budget, 2 seeds,\nstate-permutation control\nESConv null; Arithmetic +0.34', 6.6, ls=1.25)

# ---- lower branch label: "SL < 0.10" sits left of the gray box, below the arrow ----
lx0 = int(Wpx * 0.785); lx1 = kx0 - 12
# Locate the lower arrow's horizontal shaft: first contiguous run of dark rows in the label column.
col = img[ky0:ky1, lx0 + 20:lx1 - 20]
rowmean = col.mean(axis=(1, 2))
dark = np.where(rowmean < 0.62)[0]
dark = dark[(dark > 0.40 * kh) & (dark < 0.75 * kh)]
runs = np.split(dark, np.where(np.diff(dark) > 1)[0] + 1)
shaft = runs[0]
s0, s1 = ky0 + int(shaft.min()), ky0 + int(shaft.max())
thick = s1 - s0 + 1
yc = (s0 + s1) / 2
# vertical segment x: dark columns in the band just above the shaft
band = img[s0 - 40:s0 - 8, lx0:lx1].mean(axis=(0, 2))
vcols = np.where(band < 0.62)[0]
xv = lx0 + (vcols.min() + vcols.max()) / 2.0
wv = vcols.max() - vcols.min() + 1
shaft_color = tuple(img[int(yc), int(xv) + 60])
print(f'shaft rows {s0}-{s1} (thick {thick}px), x_vertical {xv}, colour {np.round(shaft_color, 2)}')
# cover shaft + old label, then redraw the shaft and arrowhead exactly
cover((lx0, s0, int(xv + wv), ky0 + int(0.82 * kh)), PANEL, inset=0)            # under the joint, keep the vertical shaft above
cover((int(xv + wv), s0 - 26, kx0 + 1, ky0 + int(0.82 * kh)), PANEL, inset=0)  # old shaft + whole old arrowhead
# restore the gray box's left border where the cover touched it
bcol = img[int(yc) - 60:int(yc) - 40, kx0 - 6:kx0 + 12].mean(axis=(0, 2)); bcols = np.where(bcol < 0.5)[0]
bx = kx0 - 6 + (bcols.min() + bcols.max()) / 2.0; bwid = bcols.max() - bcols.min() + 1
ax.plot([bx, bx], [s0 - 30, ky0 + int(0.82 * kh)], color=tuple(img[int(yc) - 50, int(bx)]), lw=bwid * 72 / DPI, solid_capstyle='butt', zorder=5)
from matplotlib.patches import FancyArrowPatch
ax.plot([xv, xv], [s0 - 14, yc + wv / 2], color=shaft_color, lw=(wv + 1) * 72 / DPI, solid_capstyle='butt', zorder=4)
ax.add_patch(FancyArrowPatch((xv - wv / 2, yc), (kx0 - 3, yc), arrowstyle='-|>', mutation_scale=thick * 1.9,
                             lw=thick * 72 / DPI, color=shaft_color, capstyle='butt', joinstyle='miter', zorder=4))
ly0, ly1 = s1 + 8, ky0 + int(0.82 * kh)
text((lx0 + lx1) / 2, (ly0 + ly1) / 2, r'$SL \approx 0$', 9.6, style='italic')

# ---- bar chart: keep panel title, redraw bars/labels below it ----
bx0, by0, bx1, by1 = bar_bb; bh = by1 - by0
cover((bx0, by0 + int(0.31 * bh), bx1, by1), BARBG)
cover((bx0, by0 + int(0.10 * bh), bx0 + int(0.24 * (bx1 - bx0)), by1), BARBG)
bars = [('Math', 0.584, True), ('Tool-use', 0.398, True), ('DoND', 0.057, False), ('ESConv', 0.028, False),
        ('Prosocial', 0.024, False), ('USS', 0.012, False), ('Code', 0.003, True), ('AirDial.', 0.001, False), ('HH-RLHF', -0.003, False)]
base_y = by0 + 0.78 * bh; top_y = by0 + 0.40 * bh; scale = (base_y - top_y) / 0.584
n = len(bars); left = bx0 + 0.035 * (bx1 - bx0); right = bx1 - 0.16 * (bx1 - bx0); gap = (right - left) / n; bw = 0.52 * gap
ax.plot([left - 0.2 * gap, right + 0.2 * gap], [base_y, base_y], color='#333333', lw=0.9, zorder=4)
G1, B1 = '#4E8F4C', '#9FC1DC'
for i, (name, v, ctx) in enumerate(bars):
    x = left + i * gap + (gap - bw) / 2; h = max(v, 0) * scale
    ax.add_patch(Rectangle((x, base_y - h), bw, h, fc=G1 if ctx else B1, ec='none', zorder=4))
    text(x + bw / 2, base_y - h - 0.05 * bh, f'{v:.3f}' if abs(v) < 0.1 else f'{v:.2f}', 7.2)
    text(x + bw / 2, base_y + 0.08 * bh, name, 6.6)
ax.plot([left - 0.2 * gap, right + 0.2 * gap], [base_y - 0.10 * scale] * 2, color='#B64342', lw=0.8, ls='--', zorder=4)
text(right + 0.25 * gap, base_y - 0.10 * scale - 0.04 * bh, '0.10', 6.0, color='#B64342', ha='left')
lx = bx1 - 0.15 * (bx1 - bx0)
ax.add_patch(Rectangle((lx, by0 + 0.40 * bh), 0.014 * (bx1 - bx0), 0.06 * bh, fc=G1, ec='none', zorder=4)); text(lx + 0.02 * (bx1 - bx0), by0 + 0.43 * bh, 'context-correct.', 6.2, ha='left')
ax.add_patch(Rectangle((lx, by0 + 0.52 * bh), 0.014 * (bx1 - bx0), 0.06 * bh, fc=B1, ec='none', zorder=4)); text(lx + 0.02 * (bx1 - bx0), by0 + 0.55 * bh, 'genuine', 6.2, ha='left')

for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(OUT_PAPER, f'fig0_headline_cr.{ext}'), dpi=DPI)
print('font', FAM, '| saved fig0_headline_cr')

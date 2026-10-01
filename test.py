"""
pode_test.py
============
Single-polygon decomposition script using the `pode` library.
https://github.com/GeorgySk/pode

HOW TO USE
----------
Edit only the PARAMETERS section below (Section 0).
Everything else runs automatically.

TIMING
------
Only the divide() call is timed — polygon construction and
visualization are excluded so you get a clean measure of
the decomposition algorithm itself.
"""

import math
import time

# ── matplotlib: show window if display available, else save-only ──
import matplotlib
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.figure(); plt.close()          # probe — raises if no display
except Exception:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.patches import Polygon as MplPoly
from matplotlib.collections import PatchCollection

from pode import (
    divide,
    Contour,
    Point,
    Polygon,
    Requirement,
    joined_constrained_delaunay_triangles,
)


# ===========================================================================
# 0.  PARAMETERS  ← only section you need to edit
# ===========================================================================

# --- Polygon shape ---
N_SIDES = 4          # number of sides  (3=triangle, 6=hex, 8=oct, 12=dodec …)
RADIUS  = 31800.0        # radius from centre to each corner (any positive number)

# --- Partitions ---
# One weight per partition.  Raw numbers — they are normalised automatically.
# [1, 1, 1]      → equal thirds  (33.3% each)
# [1, 2, 3]      → 16.7% / 33.3% / 50%
# [1, 2, 3, 4]   → 10% / 20% / 30% / 40%
z= 25          #number of partitions
ANCHOR_XY = []
WEIGHTS = []

for i in range(1,z):
    ANCHOR_XY.append((None))
    WEIGHTS.append(i)


# --- Anchor points ---
# One entry per partition, matching WEIGHTS order.
# Use None for a free (unanchored) partition.
# Use (x, y) to force that partition to contain that point.
#
# Rule: the point must lie inside the polygon and must be reachable
# from the partition it is assigned to without crossing another anchor's zone.
# Safe default: place anchors near the polygon centre or evenly spread inside.
#
# Example — anchor every partition to a corner halfway to the edge:
#   ANCHOR_XY = [(2.5, 0), (-2.5, 0), (0, 2.5), (0, -2.5)]
#
# Example — mix of free and anchored:
#   ANCHOR_XY = [None, None, (1.0, 1.0), None]


# --- Output ---
OUTPUT_FILE = "pode_test_output.png"   # saved in the folder you run from


# ===========================================================================
# 1.  VALIDATION  (catches bad input before spending time on computation)
# ===========================================================================

def validate():
    assert N_SIDES >= 3, \
        f"N_SIDES must be >= 3, got {N_SIDES}"
    assert RADIUS > 0, \
        f"RADIUS must be > 0, got {RADIUS}"
    assert len(WEIGHTS) >= 2, \
        f"Need at least 2 weights (partitions), got {len(WEIGHTS)}"
    assert all(w > 0 for w in WEIGHTS), \
        f"All weights must be positive, got {WEIGHTS}"
    assert len(ANCHOR_XY) == len(WEIGHTS), \
        (f"ANCHOR_XY length ({len(ANCHOR_XY)}) must match "
         f"WEIGHTS length ({len(WEIGHTS)})")


# ===========================================================================
# 2.  HELPERS
# ===========================================================================

# one distinct colour per partition (cycles if you have > 8 partitions)
COLORS = ['#4363d8', '#e6194b', '#3cb44b', '#f58231',
          '#911eb4', '#42d4f4', '#f032e6', '#bfef45']


def make_regular_polygon(n, r):
    """
    Build a regular n-sided polygon centred at the origin.

    Points are placed starting from the top (−π/2 offset) and go
    counter-clockwise, which is the winding order pode expects.
    """
    pts = [
        Point(r * math.cos(2 * math.pi * k / n - math.pi / 2),
              r * math.sin(2 * math.pi * k / n - math.pi / 2))
        for k in range(n)
    ]
    return Polygon(Contour(pts))


def build_requirements(weights, anchor_xy_list):
    """
    Convert raw weights + optional anchor coordinates → list of Requirement.

    Parameters
    ----------
    weights        : list of positive numbers
    anchor_xy_list : list of (x, y) tuples or None, same length as weights

    How Requirement works
    ---------------------
    Requirement(fraction)              — free partition, no location constraint
    Requirement(fraction, point=p)    — anchored: this part must contain p
    fraction = weight / sum(weights)  — normalised automatically here
    """
    total = sum(weights)
    reqs  = []
    for w, xy in zip(weights, anchor_xy_list):
        frac = w / total
        if xy is not None:
            reqs.append(Requirement(frac, point=Point(xy[0], xy[1])))
        else:
            reqs.append(Requirement(frac))
    return reqs


def polygon_to_xy(polygon):
    """
    Extract border vertices as (x, y) float tuples.

    pode stores coordinates as exact Fractions internally for numerical
    stability — we convert to float only here, at the matplotlib boundary.
    """
    return [(float(v.x), float(v.y)) for v in polygon.border.vertices]


# ===========================================================================
# 3.  VISUALIZATION  (deliberately separate from timing)
# ===========================================================================

def visualize(polygon, parts, elapsed_s, anchor_xy_list):
    """
    Draw the decomposed polygon and print a timing summary.

    This function is called AFTER timing is complete — it has no effect
    on the measured processing time.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    total_area = float(polygon.area)

    # Title encodes the key parameters
    anchor_count = sum(1 for a in anchor_xy_list if a is not None)
    ax.set_title(
        f"{N_SIDES}-sided polygon  |  r = {RADIUS}  |  "
        f"{len(parts)} partitions  |  {anchor_count} anchor(s)\n"
        f"divide() time: {elapsed_s * 1000:.4f} ms",
        fontsize=10, fontweight='bold', pad=10
    )
    ax.set_aspect('equal')

    legend_handles = []

    # --- draw each partition ---
    for i, part in enumerate(parts):
        xy    = polygon_to_xy(part)
        color = COLORS[i % len(COLORS)]

        ax.add_collection(PatchCollection(
            [MplPoly(xy, closed=True)],
            facecolor=color, edgecolor='black',
            linewidths=1.2, alpha=0.55
        ))

        # label at centroid (average of vertices)
        cx  = sum(p[0] for p in xy) / len(xy)
        cy  = sum(p[1] for p in xy) / len(xy)
        pct = float(part.area) / total_area * 100

        ax.text(cx, cy,
                f'P{i + 1}\n{pct:.1f}%',
                ha='center', va='center',
                fontsize=9, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor=color, alpha=0.75, edgecolor='none'))

        legend_handles.append(
            mpatches.Patch(
                color=color, alpha=0.6,
                label=(f'P{i + 1}: weight {WEIGHTS[i]}  →  {pct:.1f}%  '
                       f'({float(part.area):.4f} u²)')
            )
        )

    # --- draw anchor points ---
    for i, xy in enumerate(anchor_xy_list):
        if xy is not None:
            ax.plot(xy[0], xy[1],
                    marker='*', color='gold', markersize=15,
                    markeredgecolor='black', markeredgewidth=0.8, zorder=6)

    if any(a is not None for a in anchor_xy_list):
        legend_handles.append(
            mlines.Line2D([0], [0], marker='*', color='w',
                          markerfacecolor='gold', markersize=11,
                          markeredgecolor='black', label='Anchor point')
        )

    # --- original polygon border ---
    bxy = polygon_to_xy(polygon)
    bxs = [p[0] for p in bxy] + [bxy[0][0]]
    bys = [p[1] for p in bxy] + [bxy[0][1]]
    ax.plot(bxs, bys, 'k-', linewidth=2.0, zorder=4)

    # --- axis limits ---
    all_xs = [p[0] for p in bxy]
    all_ys = [p[1] for p in bxy]
    span   = max(max(all_xs) - min(all_xs), max(all_ys) - min(all_ys))
    m      = 0.20 * span
    ax.set_xlim(min(all_xs) - m, max(all_xs) + m)
    ax.set_ylim(min(all_ys) - m, max(all_ys) + m)

    ax.legend(handles=legend_handles, fontsize=8,
              loc='upper left', framealpha=0.85)
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylabel('y', fontsize=9)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(OUTPUT_FILE, dpi=150, bbox_inches='tight')
    print(f"Plot saved → {OUTPUT_FILE}")
    plt.show()


# ===========================================================================
# 4.  MAIN
# ===========================================================================

def main():
    # ── validate user input ──────────────────────────────────────────────────
    validate()

    n_parts = len(WEIGHTS)
    total_w = sum(WEIGHTS)

    print("\n" + "=" * 52)
    print("  pode — single polygon decomposition")
    print("=" * 52)
    print(f"  Polygon  : {N_SIDES} sides,  radius = {RADIUS}")
    print(f"  Parts    : {n_parts}")
    print(f"  Weights  : {WEIGHTS}  (sum = {total_w})")
    for i, (w, xy) in enumerate(zip(WEIGHTS, ANCHOR_XY)):
        anchor_str = f"anchor @ {xy}" if xy is not None else "free"
        print(f"    P{i+1}: {w}/{total_w} = {w/total_w*100:.1f}%  [{anchor_str}]")
    print()

    # ── build polygon (not timed) ─────────────────────────────────────────────
    polygon = make_regular_polygon(N_SIDES, RADIUS)
    reqs    = build_requirements(WEIGHTS, ANCHOR_XY)

    # ── TIMED: only divide() ──────────────────────────────────────────────────
    # time.perf_counter() is the highest-resolution timer Python provides.
    # It measures wall-clock time (not CPU time), so it includes any OS
    # scheduling delays — but for a single-threaded call like this it is
    # accurate to microsecond level on modern hardware.
    t0    = time.perf_counter()
    parts = divide(polygon, reqs, joined_constrained_delaunay_triangles)
    t1    = time.perf_counter()

    elapsed_s  = t1 - t0
    elapsed_ms = elapsed_s * 1000

    # ── print results ─────────────────────────────────────────────────────────
    print(f"  Polygon area : {float(polygon.area):.6f} u²")
    print(f"  Parts found  : {len(parts)}")
    print()

    total_area = float(polygon.area)
    for i, part in enumerate(parts):
        a   = float(part.area)
        pct = a / total_area * 100
        print(f"    P{i+1}: {a:.6f} u²  ({pct:.2f}%)")

    print()
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │  divide() time : {elapsed_ms:>10.4f} ms        │")
    print(f"  │               = {elapsed_s:>10.6f} s         │")
    print(f"  └─────────────────────────────────────┘")
    print()

    # ── visualize (not timed) ─────────────────────────────────────────────────
    visualize(polygon, parts, elapsed_s, ANCHOR_XY)


if __name__ == "__main__":
    main()
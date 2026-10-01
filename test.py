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

POLYGON WITH A HOLE
--------------------
pode supports polygons with a hole (like a donut shape).
Set HOLE_VERTICES to a list of (x, y) tuples defining the hole boundary.
Set HOLE_VERTICES = None if you don't want a hole.

Rules for holes:
  1. The hole must be fully inside the outer polygon.
  2. The hole must not touch the outer boundary.
  3. Only ONE hole is supported reliably by pode.
     (Multiple holes produce incorrect area splits — library limitation.)
  4. Winding order of the hole does not matter — pode handles it.

The area that pode works with = outer polygon area - hole area.
All partition weights and percentages are relative to this net area.
"""

import math
import time

# ── matplotlib: show window if display available, else save-only ──
import matplotlib
try:
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.figure(); plt.close()
except Exception:
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.path import Path
from matplotlib.patches import PathPatch
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

# --- Outer polygon shape ---
N_SIDES = 8          # number of sides  (3=triangle, 6=hex, 8=oct, 12=dodec …)
RADIUS  = 31800.0        # radius from centre to each corner (any positive number)

# --- Hole (optional) ---
# List of (x, y) tuples tracing the hole boundary — counter-clockwise or
# clockwise, pode accepts both.
# Set to None if you do not want a hole.
#
# Example — small square hole in the centre of an octagon (r=5):
#   HOLE_VERTICES = [(-1, -1), (1, -1), (1, 1), (-1, 1)]
#
# Example — off-centre rectangular hole:
#   HOLE_VERTICES = [(1, 1), (3, 1), (3, 3), (1, 3)]
#
# Rule: all hole points must lie strictly inside the outer polygon.
HOLE_VERTICES = [(-15000, -15000), (15000, -15000), (15000, 15000), (-15000, 15000)]

# --- Partitions ---
# One weight per partition.  Raw numbers — normalised to % automatically.
# [1, 1, 1]      → equal thirds  (33.3% each)
# [1, 2, 3]      → 16.7% / 33.3% / 50%
# [1, 2, 3, 4]   → 10% / 20% / 30% / 40%
WEIGHTS = []
ANCHOR_XY = []
z = 25
for i in range(1,z):
    WEIGHTS.append(i)
    ANCHOR_XY.append(None)
# --- Anchor points ---
# One entry per partition, matching WEIGHTS order.
# None  → free partition (no location constraint)
# (x,y) → that partition is guaranteed to contain this point

# --- Output ---
OUTPUT_FILE = "pode_test_output.png"   # saved in the folder you run from


# ===========================================================================
# 1.  VALIDATION
# ===========================================================================

def validate():
    assert N_SIDES >= 3, \
        f"N_SIDES must be >= 3, got {N_SIDES}"
    assert RADIUS > 0, \
        f"RADIUS must be > 0, got {RADIUS}"
    assert len(WEIGHTS) >= 2, \
        f"Need at least 2 weights, got {len(WEIGHTS)}"
    assert all(w > 0 for w in WEIGHTS), \
        f"All weights must be positive, got {WEIGHTS}"
    assert len(ANCHOR_XY) == len(WEIGHTS), \
        (f"ANCHOR_XY length ({len(ANCHOR_XY)}) must match "
         f"WEIGHTS length ({len(WEIGHTS)})")
    if HOLE_VERTICES is not None:
        assert len(HOLE_VERTICES) >= 3, \
            "HOLE_VERTICES must have at least 3 points"


# ===========================================================================
# 2.  HELPERS
# ===========================================================================

COLORS = ['#4363d8', '#e6194b', '#3cb44b', '#f58231',
          '#911eb4', '#42d4f4', '#f032e6', '#bfef45']


def make_polygon(n, r, hole_vertices=None):
    """
    Build a regular n-sided polygon centred at the origin.
    Optionally punch a hole in it.

    Parameters
    ----------
    n              : int   — number of sides
    r              : float — radius to each corner
    hole_vertices  : list of (x, y) tuples or None

    How holes work internally
    -------------------------
    pode's Polygon takes:
        Polygon(border_contour)                         — no hole
        Polygon(border_contour, holes=[hole_contour])   — with hole

    The net area that divide() works with = outer area - hole area.
    """
    outer_pts = [
        Point(r * math.cos(2 * math.pi * k / n - math.pi / 2),
              r * math.sin(2 * math.pi * k / n - math.pi / 2))
        for k in range(n)
    ]
    outer_contour = Contour(outer_pts)

    if hole_vertices is None:
        return Polygon(outer_contour)

    hole_pts     = [Point(x, y) for x, y in hole_vertices]
    hole_contour = Contour(hole_pts)
    return Polygon(outer_contour, holes=[hole_contour])


def build_requirements(weights, anchor_xy_list):
    """Convert raw weights + optional anchor coords → Requirement list."""
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
    """Return border vertices as (x, y) float tuples."""
    return [(float(v.x), float(v.y)) for v in polygon.border.vertices]


def contour_to_xy(contour):
    """Return contour vertices as (x, y) float tuples."""
    return [(float(v.x), float(v.y)) for v in contour.vertices]


def make_donut_path(outer_xy, hole_xy):
    """
    Build a matplotlib Path that fills the outer polygon but
    leaves the hole empty (even-odd fill rule).

    Parameters
    ----------
    outer_xy : list of (x, y) — outer boundary vertices
    hole_xy  : list of (x, y) — hole boundary vertices

    Returns
    -------
    matplotlib.path.Path
    """
    def ring_codes(verts):
        return ([Path.MOVETO]
                + [Path.LINETO] * (len(verts) - 1)
                + [Path.CLOSEPOLY])

    outer_closed = outer_xy + [outer_xy[0]]
    hole_closed  = hole_xy  + [hole_xy[0]]

    all_verts = outer_closed + hole_closed
    all_codes = ring_codes(outer_closed) + ring_codes(hole_closed)
    return Path(all_verts, all_codes)


# ===========================================================================
# 3.  VISUALIZATION  (not included in timing)
# ===========================================================================

def visualize(polygon, parts, elapsed_s, anchor_xy_list, hole_vertices):
    fig, ax = plt.subplots(figsize=(8, 7))
    total_area   = float(polygon.area)
    anchor_count = sum(1 for a in anchor_xy_list if a is not None)
    hole_str     = f"1 hole" if hole_vertices else "no hole"

    ax.set_title(
        f"{N_SIDES}-sided polygon  |  r = {RADIUS}  |  {hole_str}  |  "
        f"{len(parts)} partitions  |  {anchor_count} anchor(s)\n"
        f"divide() time: {elapsed_s * 1000:.4f} ms",
        fontsize=10, fontweight='bold', pad=10
    )
    ax.set_aspect('equal')

    legend_handles = []
    outer_xy = polygon_to_xy(polygon)   # used for axis scaling

    # --- draw each partition ---
    for i, part in enumerate(parts):
        xy    = polygon_to_xy(part)
        color = COLORS[i % len(COLORS)]

        # A partition piece may itself contain a piece of the hole.
        # If so, draw it as a donut patch; otherwise as a plain polygon.
        part_holes = list(part.holes)
        if part_holes:
            # draw with hole cut out
            for h in part_holes:
                h_xy  = contour_to_xy(h)
                path  = make_donut_path(xy, h_xy)
                patch = PathPatch(path,
                                  facecolor=color, edgecolor='black',
                                  linewidth=1.2, alpha=0.55)
                ax.add_patch(patch)
        else:
            ax.add_collection(PatchCollection(
                [MplPoly(xy, closed=True)],
                facecolor=color, edgecolor='black',
                linewidths=1.2, alpha=0.55
            ))

        # label at centroid
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
    for xy in anchor_xy_list:
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

    # --- draw outer polygon border ---
    bxs = [p[0] for p in outer_xy] + [outer_xy[0][0]]
    bys = [p[1] for p in outer_xy] + [outer_xy[0][1]]
    ax.plot(bxs, bys, 'k-', linewidth=2.0, zorder=4)

    # --- draw hole border (dashed) ---
    if hole_vertices:
        hxs = [p[0] for p in hole_vertices] + [hole_vertices[0][0]]
        hys = [p[1] for p in hole_vertices] + [hole_vertices[0][1]]
        ax.plot(hxs, hys, 'k--', linewidth=1.5, zorder=5,
                label='Hole boundary')
        legend_handles.append(
            mlines.Line2D([0], [0], color='black', linestyle='--',
                          linewidth=1.5, label='Hole boundary')
        )

    # --- axis limits ---
    all_xs = [p[0] for p in outer_xy]
    all_ys = [p[1] for p in outer_xy]
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
    validate()

    n_parts = len(WEIGHTS)
    total_w = sum(WEIGHTS)

    print("\n" + "=" * 52)
    print("  pode — single polygon decomposition")
    print("=" * 52)
    print(f"  Polygon  : {N_SIDES} sides,  radius = {RADIUS}")
    if HOLE_VERTICES:
        print(f"  Hole     : {len(HOLE_VERTICES)} vertices")
    else:
        print(f"  Hole     : none")
    print(f"  Parts    : {n_parts}")
    print(f"  Weights  : {WEIGHTS}  (sum = {total_w})")
    for i, (w, xy) in enumerate(zip(WEIGHTS, ANCHOR_XY)):
        anchor_str = f"anchor @ {xy}" if xy is not None else "free"
        print(f"    P{i+1}: {w}/{total_w} = {w/total_w*100:.1f}%  [{anchor_str}]")
    print()

    # ── build polygon (not timed) ─────────────────────────────────────────
    polygon = make_polygon(N_SIDES, RADIUS, HOLE_VERTICES)
    reqs    = build_requirements(WEIGHTS, ANCHOR_XY)

    outer_area = math.pi   # placeholder — we use polygon.area below
    net_area   = float(polygon.area)
    print(f"  Net area (outer − hole) : {net_area:.6f} u²")
    print()

    # ── TIMED: only divide() ──────────────────────────────────────────────
    t0    = time.perf_counter()
    parts = divide(polygon, reqs, joined_constrained_delaunay_triangles)
    t1    = time.perf_counter()

    elapsed_s  = t1 - t0
    elapsed_ms = elapsed_s * 1000

    # ── print results ─────────────────────────────────────────────────────
    print(f"  Parts found : {len(parts)}")
    for i, part in enumerate(parts):
        a   = float(part.area)
        pct = a / net_area * 100
        print(f"    P{i+1}: {a:.6f} u²  ({pct:.2f}%)")

    print()
    print(f"  ┌─────────────────────────────────────┐")
    print(f"  │  divide() time : {elapsed_ms:>10.4f} ms        │")
    print(f"  │               = {elapsed_s:>10.6f} s         │")
    print(f"  └─────────────────────────────────────┘")
    print()

    # ── visualize (not timed) ─────────────────────────────────────────────
    visualize(polygon, parts, elapsed_s, ANCHOR_XY, HOLE_VERTICES)


if __name__ == "__main__":
    main()
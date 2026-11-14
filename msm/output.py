"""Showing a solved layout: plot, Excel file and text summaries."""

from __future__ import annotations

import os
import warnings

import numpy as np

from .grid import cell_name, square
from .solver import Layout

STRUCTURE = "#4C78A8"
COLLECTION = "#E45756"
FILL = "#54A24B"
BLOCKED = "#2B2B2B"
AXIS = "#F2B701"
EXTRA_PALETTE = ("#9467BD", "#F58518", "#17BECF", "#EECA3B", "#FF9DA6", "#8C564B", "#B5BD61", "#7F7F7F")


def piece_colors(layout: Layout) -> dict[str, str]:
    """Colour per kind: structures blue, collection red, each extra its own colour, fill green."""
    colors, extras = {}, 0
    for k in layout.kinds:
        if k.group == "extra":
            colors[k.name] = EXTRA_PALETTE[extras % len(EXTRA_PALETTE)]
            extras += 1
        else:
            colors[k.name] = {"structure": STRUCTURE, "collection": COLLECTION, "fill": FILL}[k.group]
    return colors


def _label(p) -> str | None:
    """Name shown on a piece: named structure types and extras (generic pieces stay blank)."""
    generic = p.kind.startswith(("structure ", "collection ", "fill "))
    return None if generic else p.kind


def _headline(layout: Layout) -> str:
    if layout.island.income_mode:
        mix = ", ".join(f"{n} x{c}" for n, c in layout.mix.items() if c) or "none"
        return f"income {layout.income:,} (best possible {layout.income_bound:,})   extras: {mix}"
    return f"fill {layout.island.fill}x{layout.island.fill}: {layout.fill} (best possible {layout.fill_bound})"


# --------------------------------------------------------------------------- #
def draw(layout: Layout, ax=None, show: bool = True):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    grid = layout.grid
    H, W = grid.shape
    if ax is None:
        _fig, ax = plt.subplots(figsize=(max(12, W / 2.2), max(10, H / 2.2)))
    ax.imshow(np.where(grid[:, :, None], [1.0, 1.0, 1.0], [43 / 255] * 3), interpolation="none")

    colors = piece_colors(layout)
    for p in layout.placements:
        ax.add_patch(mpatches.Rectangle((p.x - 0.5, p.y - 0.5), p.s, p.s, facecolor=colors[p.kind], alpha=0.85,
                                        edgecolor="#111111", linewidth=2.6 if p.locked else 1.2,
                                        hatch="///" if p.locked else None))
        name = _label(p)
        if p.s >= 2 or name:
            text = f"{p.s}" if not name or p.s < 3 else f"{p.s}\n{name[:9]}"
            ax.text(p.x + (p.s - 1) / 2, p.y + (p.s - 1) / 2, text, ha="center", va="center",
                    fontsize=7 + p.s, color="white", fontweight="bold")

    axis, height = layout.axis, layout.height
    xs = np.array([-0.5, W - 0.5])
    ax.plot(xs, (axis.c - xs) if axis.kind == "anti" else (xs - axis.c), "--", color=AXIS, lw=2.2, zorder=5)
    top = height.top_cell(grid)
    dx, dy = height.up_vector()
    tail = (top[0] - 7 * dx, top[1] - 7 * dy)
    ax.annotate("", xy=top, xytext=tail, zorder=6, arrowprops=dict(arrowstyle="-|>", color=AXIS, lw=3))
    ax.text(*tail, "in-game up", color=AXIS, fontsize=11, fontweight="bold", ha="center", va="center", zorder=6,
            bbox=dict(facecolor=BLOCKED, edgecolor="none", alpha=0.75, pad=2))

    title = f"{layout.island.name}\n{_headline(layout)}"
    if layout.mirrored is not None:
        title += f"\nmirrored structures: {layout.mirrored}/{layout.structures_total}  (yellow line = mirror axis)"
    ax.set_title(title, pad=16)
    ax.set_xlim(-0.5, W - 0.5)
    ax.set_ylim(H - 0.5, -0.5)
    ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
    ax.grid(which="minor", color="#DDDDDD", linewidth=0.35)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)

    handles = [mpatches.Patch(color=STRUCTURE, label="Structure"), mpatches.Patch(color=COLLECTION, label="Collection")]
    for k in layout.kinds:
        if k.group == "extra" and k.count:
            handles.append(mpatches.Patch(color=colors[k.name], label=f"{k.name} (extra x{k.count})"))
        elif k.group == "fill":
            handles.append(mpatches.Patch(color=FILL, label=f"Fill {k.size}x{k.size} x{k.count}"))
    if any(p.locked for p in layout.placements):
        handles.append(mpatches.Patch(facecolor="white", edgecolor="#111111", hatch="///", label="Locked"))
    handles.append(mpatches.Patch(color=BLOCKED, label="Blocked"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.03), ncol=min(len(handles), 6), frameon=False)
    if show:
        plt.tight_layout()
        plt.show()
    return ax


# --------------------------------------------------------------------------- #
_SHADES = {  # lighter = smaller, so sizes are easy to tell apart in Excel
    "structure": {1: "DCE8F5", 2: "A9C4E2", 3: "6F9FD1", 4: "3E79B2"},
    "collection": {1: "F8D6D6", 2: "F2A7A7", 3: "E46F6F", 4: "D64545"},
    "fill": {1: "D9F0D2", 2: "A9D99B", 3: "78C266", 4: "4AA64A"},
}


def export_excel(layout: Layout, path: str | None = None) -> str:
    """Write the layout (sheet 'Layout') and its numbers (sheet 'Summary') to an .xlsx file."""
    from openpyxl import Workbook
    from openpyxl.styles import Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter

    path = path or layout.island.excel_path
    grid = layout.grid
    H, W = grid.shape
    colors = piece_colors(layout)
    fills = {}

    def fill(hex6):
        if hex6 not in fills:
            fills[hex6] = PatternFill(start_color="FF" + hex6, end_color="FF" + hex6, fill_type="solid")
        return fills[hex6]

    wb = Workbook()
    ws = wb.active
    ws.title = "Layout"
    for y in range(H):
        ws.row_dimensions[y + 1].height = 18
        for x in range(W):
            ws.cell(y + 1, x + 1).fill = fill("FFFFFF" if grid[y, x] else BLOCKED[1:])
    for x in range(W):
        ws.column_dimensions[get_column_letter(x + 1)].width = 4

    small = Font(size=7, bold=True)
    for p in layout.placements:
        hex6 = _SHADES[p.group][p.s] if p.group in _SHADES else colors[p.kind][1:]
        for cx, cy in square(p.x, p.y, p.s):
            ws.cell(cy + 1, cx + 1).fill = fill(hex6)
        name = _label(p)
        if name or p.locked:
            c = ws.cell(p.y + 1, p.x + 1)
            c.value = (name or "") + (" (locked)" if p.locked else "")
            c.font = small

    # mirror axis: yellow outline on the tiles it runs through; "UP" on the top tile
    side = Side(style="medium", color="FF" + AXIS[1:])
    for y in range(H):
        for x in range(W):
            if layout.axis.on_axis(x, y):
                ws.cell(y + 1, x + 1).border = Border(left=side, right=side, top=side, bottom=side)
    tx, ty = layout.height.top_cell(grid)
    top = ws.cell(ty + 1, tx + 1)
    top.value = (top.value + " " if top.value else "") + "UP"
    top.font = small

    s = wb.create_sheet("Summary")
    for row in _summary_rows(layout):
        s.append(row)
    s.column_dimensions["A"].width = 26
    s.column_dimensions["B"].width = 60
    for r in range(1, s.max_row + 1):
        s.cell(r, 1).font = Font(bold=True)
    return _save(wb, path)


def _save(wb, path: str) -> str:
    """
    Save the workbook. If the file is open in Excel (Windows locks it), save
    next to it as "<name> (1).xlsx", "(2)", ... instead of losing the result.
    """
    root, ext = os.path.splitext(path)
    for n in range(0, 100):
        target = path if n == 0 else f"{root} ({n}){ext}"
        try:
            wb.save(target)
        except PermissionError:
            continue
        if n:
            warnings.warn(f"{os.path.basename(path)} is open in another program (close it in Excel);"
                          f" saved as {os.path.basename(target)} instead.", stacklevel=3)
        return target
    raise PermissionError(f"could not save {path}: it and 99 numbered copies are all in use or read-only.")


def _summary_rows(layout: Layout) -> list[tuple]:
    isl = layout.island
    rows = [("Island", isl.name), ("Grid", isl.csv),
            ("Mode", "income" if isl.income_mode else f"fill {isl.fill}x{isl.fill}")]
    if isl.income_mode:
        rows += [("Income", layout.income), ("Best possible income", layout.income_bound),
                 ("Income proven best", "yes" if layout.packing_optimal else "no (time limit)")]
        rows += [(f"Extra: {n}", c) for n, c in layout.mix.items()]
        if isl.beds is not None:
            rows.append(("Beds used", f"{layout.beds_used}/{isl.beds}"))
    else:
        rows += [("Fill squares", layout.fill), ("Best possible", layout.fill_bound),
                 ("Fill proven best", "yes" if layout.packing_optimal else "no (time limit)")]
    if layout.mirrored is not None:
        rows += [("Mirrored structures", f"{layout.mirrored}/{layout.structures_total}"),
                 ("Symmetry proven best", "yes" if layout.symmetry_optimal else f"no (up to {layout.mirrored_bound})")]
        if _named(layout):
            rows.append(("Same type opposite", f"{layout.same_type}/{layout.structures_total}"))
    if layout.arrange_gap is not None:
        rows.append(("Arrangement", f"{layout.arrange_gap:.1%} from the solver's bound (cosmetic only)"))
    rows += [("Free tiles", layout.free_cells),
             ("Mirror axis", f"{layout.axis.kind} diagonal, {layout.axis.matched}/{layout.axis.total} tiles mirror"),
             ("In-game top tile", cell_name(*layout.height.top_cell(layout.grid)))]
    rows += [(f"Locked: {p.kind}", cell_name(p.x, p.y)) for p in layout.placements if p.locked]
    rows += [(f"Time {k}", f"{v:.1f}s") for k, v in layout.seconds.items()]
    return rows


def _named(layout: Layout) -> bool:
    """True when some size has more than one structure type."""
    sizes = [k.size for k in layout.kinds if k.group == "structure" and k.count]
    return len(sizes) != len(set(sizes))


def report(layout: Layout) -> str:
    """Multi-line text summary of one layout."""
    return "\n".join(f"{k:22s} {v}" for k, v in _summary_rows(layout))


def summary(results: dict) -> str:
    """One line per island from run_all's results (a Layout or an error message)."""
    lines = [f"{'Island':10s} {'Result':42s} {'Proven':7s} {'Mirrored':9s} {'Arrange':8s} {'Time':>7s}"]
    for name, r in results.items():
        if isinstance(r, str):
            lines.append(f"{name:10s} FAILED: {r}")
            continue
        if r.island.income_mode:
            res = f"income {r.income:,}" + (f", beds {r.beds_used}/{r.island.beds}" if r.island.beds is not None else "")
        else:
            res = f"fill {r.fill}/{r.fill_bound}"
        mir = "-" if r.mirrored is None else f"{r.mirrored}/{r.structures_total}"
        arr = "-" if r.arrange_gap is None else f"{r.arrange_gap:.1%}"
        lines.append(f"{name:10s} {res:42s} {'yes' if r.packing_optimal else 'no':7s} {mir:9s} {arr:8s} "
                     f"{r.seconds.get('total', 0):6.0f}s")
    return "\n".join(lines)

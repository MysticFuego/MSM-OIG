"""
Island settings, as written in the notebook's ISLANDS cell.

    "Plant": {
        "csv": "Original.csv",            # island mask
        "excel": "Plant.xlsx",            # output file (default: <name>.xlsx)
        "structures": {4: 3, 3: 8, 1: 10},   # count by size, or named (see below)
        "monsters": {4: 3, 3: 3, 2: 52, 1: 23},  # your collection, by size
        "beds": 36,                       # income mode: extras use exactly this many beds
        "catalog": [ {extra}, ... ],      # income mode: optional extras (see Extra)
        "fill": 2,                        # fill mode only: size of the fill squares
        "locked": [ {lock}, ... ],        # pieces pinned to a cell (see Lock)
        "flip_up": False,                 # True if the 'in-game up' arrow points the wrong way
    }

With a catalog the island is in *income mode* (the solver picks extras for the
most income); without one it is in *fill mode* (as many fill squares as fit).

Structures can also be named. Symmetry mirrors shapes (same size), then puts
the same type opposite itself wherever possible:
    "structures": [("Castle", 4, 1), ("Bakery", 3, 2), ("Decoration", 1, 10)]
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .grid import parse_cell

SIZES = (1, 2, 3, 4)


@dataclass
class Kind:
    """One type of piece. Pieces of the same kind are interchangeable."""
    name: str
    group: str        # "structure" | "collection" | "extra" | "fill"
    size: int
    count: int = 0    # how many to place (extras/fill: decided by the solver)
    income: int = 0   # extras only
    beds: int = 0
    min: int = 0
    max: int = 0
    back: bool = False  # pulled to the in-game back (structures, and extras by default)


@dataclass
class Lock:
    kind: str
    x: int
    y: int


@dataclass
class Island:
    name: str
    csv: str = "Original.csv"
    structures: dict | list = field(default_factory=dict)
    monsters: dict = field(default_factory=dict)
    excel: str | None = None
    beds: int | None = None
    catalog: list = field(default_factory=list)
    fill: int = 2
    locked: list = field(default_factory=list)
    flip_up: bool = False

    @classmethod
    def from_dict(cls, name: str, d: dict) -> "Island":
        d = dict(d)
        monsters = dict(d.pop("monsters", {}))
        if "beds" in monsters:  # old style: beds inside the monsters dict
            d.setdefault("beds", monsters.pop("beds"))
        unknown = set(d) - {f for f in cls.__dataclass_fields__ if f != "name"}
        if unknown:
            raise ValueError(f"{name}: unknown setting(s) {sorted(unknown)}.")
        return cls(name=name, monsters=monsters, **d)

    @property
    def income_mode(self) -> bool:
        return bool(self.catalog)

    @property
    def excel_path(self) -> str:
        return self.excel or f"{self.name}.xlsx"

    # ------------------------------------------------------------------ #
    def kinds(self, n_open: int) -> list[Kind]:
        """Every piece type on this island, with counts and limits checked."""
        kinds: list[Kind] = []

        if isinstance(self.structures, dict):
            items = [(f"structure {s}x{s}", s, n) for s, n in self.structures.items()]
        else:
            items = [tuple(t) if not isinstance(t, dict) else (t["name"], t["size"], t.get("count", 1))
                     for t in self.structures]
        for name, s, n in items:
            kinds.append(Kind(str(name), "structure", _size(s, name), _count(n, name), back=True))

        for s, n in self.monsters.items():
            kinds.append(Kind(f"collection {s}x{s}", "collection", _size(s, "monsters"), _count(n, "monsters")))

        if self.catalog:
            for e in self.catalog:
                name, s = str(e["name"]), _size(e["size"], e["name"])
                lo = _count(e.get("min", 0), name)
                hi = e.get("max")
                hi = n_open // (s * s) if hi is None else _count(hi, name)
                beds = _count(e.get("beds", 1), name)
                if self.beds is not None and beds > 0:
                    hi = min(hi, self.beds // beds)
                if hi < lo:
                    raise ValueError(f"{self.name}: {name} needs min {lo} but at most {hi} fit the limits/beds.")
                kinds.append(Kind(name, "extra", s, income=int(round(float(e.get("income", 0)))),
                                  beds=beds, min=lo, max=hi, back=bool(e.get("back", True))))
        else:
            kinds.append(Kind(f"fill {self.fill}x{self.fill}", "fill", _size(self.fill, "fill")))

        names = [k.name for k in kinds]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"{self.name}: duplicate piece names {sorted(dupes)}.")
        return [k for k in kinds if k.count > 0 or k.group in ("extra", "fill")]

    def locks(self, kinds: list[Kind]) -> list[Lock]:
        """
        Locked pieces: {"what": <name>, "at": "N3"} with <name> a structure
        type, an extra's name, or "structure"/"collection" plus "size".
        "at" is the piece's top-left cell, as in the Excel output.
        """
        by_name = {k.name: k for k in kinds}
        out = []
        for d in self.locked:
            what = str(d["what"])
            if what in ("structure", "collection"):
                what = f"{what} {d['size']}x{d['size']}"
            if what not in by_name:
                raise ValueError(f"{self.name}: locked piece {d['what']!r} is not a structure, extra or collection size here.")
            if "size" in d and int(d["size"]) != by_name[what].size:
                raise ValueError(f"{self.name}: locked {what} is {by_name[what].size}x{by_name[what].size}, not size {d['size']}.")
            x, y = parse_cell(d["at"])
            out.append(Lock(what, x, y))
        return out


def _size(s, label) -> int:
    s = int(s)
    if s not in SIZES:
        raise ValueError(f"{label}: size must be 1-4, got {s}.")
    return s


def _count(n, label) -> int:
    n = int(n)
    if n < 0:
        raise ValueError(f"{label}: counts must be >= 0, got {n}.")
    return n

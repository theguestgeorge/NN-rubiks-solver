"""Cubie-level model of a 3x3x3 Rubik's cube.

A state records, for every corner and edge *slot*, which piece is sitting there
and how that piece is twisted/flipped. States are immutable and hashable, so they
can be used as dict keys / set members (handy for search and dataset generation).

Example:
    >>> cube = SOLVED_CUBE.apply_moves("R U R' U'")
    >>> cube.apply_moves(invert_moves("R U R' U'")).is_solved()
    True
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

# --------------------------------------------------------------------------- #
# Slot numbering
#
# Slots are numbered using the 3BLD lettering, read from the top-down view.
#
# Corners: a b c d -> 0 1 2 3 (U layer), u v w x -> 4 5 6 7 (D layer)
# Edges:   a b c d -> 0 1 2 3 (U layer), g h p o -> 4 5 6 7 (E slice),
#          u v w x -> 8 9 10 11 (D layer)
#
# Orientation:
#   Edges   - 0 if the piece's U/D sticker (or F/B sticker for E-slice pieces)
#             faces U/D (F/B), 1 if flipped. Only F and B quarter turns flip edges.
#   Corners - 0 if the piece's U/D sticker faces U/D, 1 if twisted clockwise,
#             2 if twisted anticlockwise. U and D turns never twist corners.
# --------------------------------------------------------------------------- #

CORNER_NAMES = ("UBL", "UBR", "UFR", "UFL", "DFL", "DFR", "DBR", "DBL")
EDGE_NAMES = ("UB", "UR", "UF", "UL", "FL", "BL", "FR", "BR", "DF", "DR", "DB", "DL")

NUM_CORNERS = len(CORNER_NAMES)
NUM_EDGES = len(EDGE_NAMES)


@dataclass(frozen=True)
class CubeState:
    """`corners[i]` is the piece in corner slot i, `corner_orientations[i]` its twist.
    Same for edges. The default-constructed state is the solved cube."""

    corners: tuple[int, ...] = tuple(range(NUM_CORNERS))
    edges: tuple[int, ...] = tuple(range(NUM_EDGES))
    corner_orientations: tuple[int, ...] = (0,) * NUM_CORNERS
    edge_orientations: tuple[int, ...] = (0,) * NUM_EDGES

    def __post_init__(self):
        # Accept lists for convenience, but store tuples so the state stays hashable.
        for name in ("corners", "edges", "corner_orientations", "edge_orientations"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

        if sorted(self.corners) != list(range(NUM_CORNERS)):
            raise ValueError(f"corners must be a permutation of 0..{NUM_CORNERS - 1}: {self.corners}")
        if sorted(self.edges) != list(range(NUM_EDGES)):
            raise ValueError(f"edges must be a permutation of 0..{NUM_EDGES - 1}: {self.edges}")
        if len(self.corner_orientations) != NUM_CORNERS or not all(o in (0, 1, 2) for o in self.corner_orientations):
            raise ValueError(f"corner_orientations must be {NUM_CORNERS} values in 0..2: {self.corner_orientations}")
        if len(self.edge_orientations) != NUM_EDGES or not all(o in (0, 1) for o in self.edge_orientations):
            raise ValueError(f"edge_orientations must be {NUM_EDGES} values in 0..1: {self.edge_orientations}")

    # ------------------------------------------------------------------ moves

    def apply_move(self, move: str) -> CubeState:
        """Return the state after a single move such as "R", "U'" or "F2"."""
        try:
            transform = MOVES[move]
        except KeyError:
            raise ValueError(f"unknown move {move!r}; expected one of {', '.join(MOVE_NAMES)}") from None
        return self.compose(transform)

    def apply_moves(self, moves: str | Iterable[str]) -> CubeState:
        """Return the state after a sequence, e.g. "R U R' U'" or ["R", "U", "R'"]."""
        state = self
        for move in parse_moves(moves):
            state = state.apply_move(move)
        return state

    def compose(self, transform: CubeState) -> CubeState:
        """Apply `transform` on top of this state.

        Any state can act as a transform: it is the move (or sequence) that takes
        the solved cube to that state. The piece that ends up in slot i is the one
        that was in slot transform.corners[i], twisted further by
        transform.corner_orientations[i] (likewise for edges).
        """
        c_src, e_src = transform.corners, transform.edges
        return CubeState(
            corners=tuple(self.corners[s] for s in c_src),
            edges=tuple(self.edges[s] for s in e_src),
            corner_orientations=tuple(
                (self.corner_orientations[s] + t) % 3 for s, t in zip(c_src, transform.corner_orientations)
            ),
            edge_orientations=tuple(
                (self.edge_orientations[s] + f) % 2 for s, f in zip(e_src, transform.edge_orientations)
            ),
        )

    # ---------------------------------------------------------------- queries

    def is_solved(self) -> bool:
        return self == SOLVED_CUBE

    def is_solvable(self) -> bool:
        """Whether this state can be reached from solved using legal moves.

        A state can be valid as a data structure but impossible on a real cube,
        e.g. one twisted corner, one flipped edge, or two swapped edges.
        """
        return (
            sum(self.corner_orientations) % 3 == 0
            and sum(self.edge_orientations) % 2 == 0
            and _parity(self.corners) == _parity(self.edges)
        )


def _parity(perm: tuple[int, ...]) -> int:
    """0 for an even permutation, 1 for an odd one."""
    seen = [False] * len(perm)
    parity = 0
    for start in range(len(perm)):
        if seen[start]:
            continue
        length = 0
        i = start
        while not seen[i]:
            seen[i] = True
            i = perm[i]
            length += 1
        parity ^= (length - 1) & 1
    return parity


# --------------------------------------------------------------------------- #
# Move definitions
#
# Each clockwise quarter turn is given as a corner cycle and an edge cycle.
# A cycle (a, b, c, d) means the piece in slot a moves to b, b to c, c to d and
# d back to a.
#
# For side faces the corner cycle starts with the U-layer corner that stays in
# the U layer, so the twists picked up at the destination slots b, c, d, a are
# always +1, +2, +1, +2.
# --------------------------------------------------------------------------- #

_QUARTER_TURNS = {
    #      corner cycle    edge cycle
    "U": ((2, 3, 0, 1), (2, 3, 0, 1)),
    "D": ((4, 5, 6, 7), (8, 9, 10, 11)),
    "R": ((2, 1, 6, 5), (6, 1, 7, 9)),
    "L": ((0, 3, 4, 7), (3, 4, 11, 5)),
    "F": ((3, 2, 5, 4), (2, 6, 8, 4)),
    "B": ((1, 0, 7, 6), (0, 5, 10, 7)),
}
_TWISTS_CORNERS = {"R", "L", "F", "B"}
_FLIPS_EDGES = {"F", "B"}


def _build_quarter_turn(face: str) -> CubeState:
    corner_cycle, edge_cycle = _QUARTER_TURNS[face]
    corners, edges = list(range(NUM_CORNERS)), list(range(NUM_EDGES))
    co, eo = [0] * NUM_CORNERS, [0] * NUM_EDGES

    for k in range(4):
        src, dst = corner_cycle[k], corner_cycle[(k + 1) % 4]
        corners[dst] = src
        if face in _TWISTS_CORNERS:
            co[dst] = 1 if k % 2 == 0 else 2

        src, dst = edge_cycle[k], edge_cycle[(k + 1) % 4]
        edges[dst] = src
        if face in _FLIPS_EDGES:
            eo[dst] = 1

    return CubeState(tuple(corners), tuple(edges), tuple(co), tuple(eo))


def _build_moves() -> dict[str, CubeState]:
    # Rather than writing out all 3 variations of a move (R, R2, R'), we repeat
    # the quarter turn: twice for R2, three times for R'.
    moves = {}
    for face in _QUARTER_TURNS:
        quarter = _build_quarter_turn(face)
        half = quarter.compose(quarter)
        moves[face] = quarter
        moves[face + "2"] = half
        moves[face + "'"] = half.compose(quarter)
    return moves


SOLVED_CUBE = CubeState()
MOVES: dict[str, CubeState] = _build_moves()
MOVE_NAMES: tuple[str, ...] = tuple(MOVES)  # the 18 face moves, useful as an NN action space


# --------------------------------------------------------------------------- #
# Notation helpers
# --------------------------------------------------------------------------- #

def parse_moves(moves: str | Iterable[str]) -> list[str]:
    """Split "R U R' U'" into ["R", "U", "R'", "U'"]; lists pass straight through."""
    tokens = moves.split() if isinstance(moves, str) else list(moves)
    for token in tokens:
        if token not in MOVES:
            raise ValueError(f"unknown move {token!r}; expected one of {', '.join(MOVE_NAMES)}")
    return tokens


def invert_move(move: str) -> str:
    if move.endswith("2"):
        return move
    if move.endswith("'"):
        return move[:-1]
    return move + "'"


def invert_moves(moves: str | Iterable[str]) -> list[str]:
    """The sequence that undoes `moves`, e.g. "R U" -> ["U'", "R'"]."""
    return [invert_move(m) for m in reversed(parse_moves(moves))]

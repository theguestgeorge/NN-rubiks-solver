import random
import unittest

from solver.cube import (
    CORNER_NAMES,
    EDGE_NAMES,
    MOVE_NAMES,
    SOLVED_CUBE,
    CubeState,
    invert_moves,
)


def order(sequence: str) -> int:
    """How many times `sequence` must be repeated to return to solved."""
    state = SOLVED_CUBE.apply_moves(sequence)
    n = 1
    while not state.is_solved():
        state = state.apply_moves(sequence)
        n += 1
    return n


class TestMoves(unittest.TestCase):
    def test_there_are_18_moves(self):
        self.assertEqual(len(MOVE_NAMES), 18)

    def test_every_quarter_turn_has_order_4(self):
        for face in "URFDLB":
            self.assertEqual(order(face), 4, face)

    def test_move_variants(self):
        for face in "URFDLB":
            self.assertEqual(SOLVED_CUBE.apply_moves([face, face]), SOLVED_CUBE.apply_move(face + "2"))
            self.assertTrue(SOLVED_CUBE.apply_moves([face, face + "'"]).is_solved())

    def test_r_moves_edges_like_original_apply_r(self):
        # Original hand-written R: edge[1] <- edge[6] <- edge[9] <- edge[7] <- edge[1]
        cube = SOLVED_CUBE.apply_move("R")
        self.assertEqual((cube.edges[1], cube.edges[6], cube.edges[9], cube.edges[7]), (6, 9, 7, 1))

    def test_known_sequence_orders(self):
        # These only come out right if both permutations and orientations are correct.
        self.assertEqual(order("R U R' U'"), 6)
        self.assertEqual(order("R U"), 105)
        self.assertEqual(order("R U2 D' B D'"), 1260)

    def test_t_perm(self):
        cube = SOLVED_CUBE.apply_moves("R U R' U' R' F R2 U' R' U' R U R' F'")
        ul, ur = EDGE_NAMES.index("UL"), EDGE_NAMES.index("UR")
        ubr, ufr = CORNER_NAMES.index("UBR"), CORNER_NAMES.index("UFR")

        expected_edges = list(range(12))
        expected_edges[ul], expected_edges[ur] = ur, ul
        expected_corners = list(range(8))
        expected_corners[ubr], expected_corners[ufr] = ufr, ubr

        self.assertEqual(cube, CubeState(corners=expected_corners, edges=expected_edges))

    def test_scramble_then_inverse_is_solved(self):
        rng = random.Random(0)
        for _ in range(100):
            scramble = [rng.choice(MOVE_NAMES) for _ in range(25)]
            cube = SOLVED_CUBE.apply_moves(scramble)
            self.assertTrue(cube.is_solvable())
            self.assertTrue(cube.apply_moves(invert_moves(scramble)).is_solved())


class TestState(unittest.TestCase):
    def test_states_are_hashable_and_immutable(self):
        a = SOLVED_CUBE.apply_moves("R U")
        b = SOLVED_CUBE.apply_moves(["R", "U"])
        self.assertEqual(len({a, b, SOLVED_CUBE}), 2)
        with self.assertRaises(AttributeError):
            a.corners = SOLVED_CUBE.corners

    def test_invalid_states_rejected(self):
        with self.assertRaises(ValueError):
            CubeState(corners=(0,) * 8)
        with self.assertRaises(ValueError):
            CubeState(edge_orientations=(2,) * 12)

    def test_unknown_move_rejected(self):
        with self.assertRaises(ValueError):
            SOLVED_CUBE.apply_moves("R X")

    def test_unsolvable_states_detected(self):
        self.assertFalse(CubeState(corner_orientations=(1,) + (0,) * 7).is_solvable())
        self.assertFalse(CubeState(edge_orientations=(1,) + (0,) * 11).is_solvable())
        self.assertFalse(CubeState(edges=(1, 0) + tuple(range(2, 12))).is_solvable())


if __name__ == "__main__":
    unittest.main()

"""CPU-only tests for state coordinates, ranking, padding and invalid observations."""
import unittest
import numpy as np
from chapter3.perception.hawp import pack_lines


class TokenTests(unittest.TestCase):
    def test_background_rejected_but_silhouette_kept(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[25:75, 25:75] = True
        lines = [[0, 5, 99, 5], [25, 25, 75, 25], [50, 25, 50, 75],
                 [0, 80, 99, 80]]
        tokens = pack_lines(lines, [.99, .9, .8, .7], 100, 100,
                            max_lines=4, foreground=mask)
        self.assertEqual(int(tokens[:, 5].sum()), 2)
        np.testing.assert_allclose(tokens[:2, 4], [.9, .8])

    def test_empty_foreground_and_shape_mismatch(self):
        tokens = pack_lines([[0, 50, 99, 50]], [.9], 100, 100,
                            foreground=np.zeros((100, 100), dtype=bool))
        self.assertEqual(int(tokens[:, 5].sum()), 0)
        with self.assertRaises(ValueError):
            pack_lines([], [], 100, 100, foreground=np.zeros((99, 100)))

    def test_empty(self):
        np.testing.assert_array_equal(pack_lines([], [], 100, 50), np.zeros((64, 6)))

    def test_reverse_and_normalization(self):
        a = pack_lines([[80, 40, 20, 10]], [.8], 100, 50)
        b = pack_lines([[20, 10, 80, 40]], [.8], 100, 50)
        np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(a[0], [.2, .2, .8, .8, .8, 1])
        self.assertEqual(a[:, 5].sum(), 1)

    def test_filter_rank_and_padding(self):
        lines = [[0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 30, 30],
                 [1, 1, 1, 1], [float('nan'), 1, 2, 3]]
        tokens = pack_lines(lines, [.8, .9, .2, 1, 1], 100, 100, max_lines=1)
        np.testing.assert_allclose(tokens[0], [0, 0, .2, .2, .9, 1])


if __name__ == '__main__':
    unittest.main()

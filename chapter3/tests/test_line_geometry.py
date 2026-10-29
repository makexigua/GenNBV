"""Synthetic geometry counterexamples; no detector, CUDA, Gym or GT assets needed."""
import unittest
import numpy as np
from chapter3.perception.hawp import pack_lines
from chapter3.perception.geometry import add_evidence, positive_gym_depth, unknown_evidence


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.h = self.w = 128
        self.k = [100., 100., 64., 64.]
        self.tokens = pack_lines([[64, 20, 64, 108]], [.9], 128, 128, max_lines=4)

    def evidence(self, depth, tokens=None):
        return add_evidence(self.tokens if tokens is None else tokens, depth, self.k)

    def test_flat_texture_is_weak_not_structural(self):
        result = self.evidence(np.full((128, 128), 5.))
        self.assertLess(result[0, 6], .001)
        self.assertGreater(result[0, 7], .99)
        np.testing.assert_array_equal(result[1:], 0)
        np.testing.assert_allclose(result[0, 6:].sum(), 1.)

    def test_sloped_plane_does_not_become_edge(self):
        y, x = np.mgrid[:128, :128]
        depth = 1. / (.2 + .09 * (x - 64) / 100 + .03 * (y - 64) / 100)
        result = self.evidence(depth)
        self.assertLess(result[0, 6], .001)
        self.assertGreater(result[0, 7], .99)

    def test_depth_step_is_supported(self):
        depth = np.full((128, 128), 5.)
        depth[:, 64:] = 8.
        self.assertGreater(self.evidence(depth)[0, 6], .99)

    def test_continuous_roof_fold_is_supported(self):
        y, x = np.mgrid[:128, :128]
        depth = 1. / (.2 + .10 * np.abs((x - 64) / 100))
        self.assertGreater(self.evidence(depth)[0, 6], .99)

    def test_missing_side_is_unknown_and_kept(self):
        depth = np.full((128, 128), 5.)
        depth[:, 64:] = np.nan
        result = self.evidence(depth)
        np.testing.assert_array_equal(result[0, 6:], [0, 0, 1])
        np.testing.assert_array_equal(result[:, :6], self.tokens)

    def test_invalid_range_and_gym_sign(self):
        z = positive_gym_depth([[-5., -np.inf, np.nan, 0., 5., -51., -50.]])
        self.assertEqual(z[0, 0], 5.)
        self.assertTrue(np.isnan(z[0, 1:]).all())
        for value in (np.nan, np.inf, 0, -5, 50, 100):
            np.testing.assert_array_equal(self.evidence(np.full((128, 128), value))[0, 6:], [0, 0, 1])

    def test_reversing_endpoints_and_line_order(self):
        depth = np.full((128, 128), 5.)
        reverse = self.tokens.copy()
        reverse[:, :4] = reverse[:, [2, 3, 0, 1]]
        np.testing.assert_allclose(self.evidence(depth, reverse)[:, 6:], self.evidence(depth)[:, 6:], atol=1e-5)
        np.testing.assert_allclose(self.evidence(depth, self.tokens[::-1]), self.evidence(depth)[::-1])

    def test_image_boundary_short_and_empty_lines(self):
        tokens = pack_lines([[0, 20, 0, 108], [64, 64, 65, 65]], [.9, .8], 128, 128, max_lines=4)
        result = self.evidence(np.full((128, 128), 5.), tokens)
        np.testing.assert_array_equal(result[:2, 6:], [[0, 0, 1], [0, 0, 1]])
        np.testing.assert_array_equal(result[2:], 0)

    def test_control_and_noisy_depth_are_finite(self):
        result = unknown_evidence(self.tokens)
        np.testing.assert_array_equal(result[0, 6:], [0, 0, 1])
        np.testing.assert_array_equal(result[1:], 0)
        rng = np.random.RandomState(7)
        result = self.evidence(rng.uniform(2., 15., (128, 128)))
        self.assertTrue(np.isfinite(result).all())
        self.assertGreater(result[0, 8], .5)
        np.testing.assert_allclose(result[0, 6:].sum(), 1., atol=1e-6)


if __name__ == '__main__':
    unittest.main()

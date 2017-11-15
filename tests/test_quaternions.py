import unittest
from hypothesis import given, assume
from hypothesis.strategies import floats
import numpy as np
import json
import os

from quaternions import Quaternion, QuaternionError

basedir = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data')

class QuaternionTest(unittest.TestCase):
    # Schaub, Chapter 3
    schaub_example_dcm = np.array([[.892539, .157379, -.422618],
                                   [-.275451, .932257, -.23457],
                                   [.357073, .325773, .875426]])
    schaub_result = np.array([.961798, -.14565, .202665, .112505])

    def test_matrix_respects_product(self):
        q1 = Quaternion.exp(Quaternion(0, .1, .02, -.3))
        q2 = Quaternion.exp(Quaternion(0, -.2, .21, .083))
        np.testing.assert_allclose((q1 * q2).matrix, q1.matrix.dot(q2.matrix))

    def test_quaternion_rotates_vector(self):
        q1 = Quaternion.exp(Quaternion(0, .1, .02, -.3))
        vector = QuaternionTest.schaub_example_dcm[:, 1]
        rotated_vector = q1 * vector
        np.testing.assert_allclose(rotated_vector, q1.matrix.dot(vector), atol=1e-5, rtol=0)

    def test_from_matrix(self):
        q = Quaternion.from_matrix(QuaternionTest.schaub_example_dcm)
        np.testing.assert_allclose(QuaternionTest.schaub_result, q.coordinates, atol=1e-5, rtol=0)

    def test_from_matrix_twisted(self):
        q = Quaternion.from_matrix(QuaternionTest.schaub_example_dcm * [-1, -1, 1])
        e1 = Quaternion(*QuaternionTest.schaub_result)
        expected = e1 * Quaternion(0, 0, 0, 1)
        np.testing.assert_allclose(expected.coordinates, q.coordinates, atol=1e-5, rtol=0)

    def test_from_rotation_vector_to_matrix(self):
        phi = np.array([-.295067, .410571, .227921])
        expected = np.array([
            [.892539, .157379, -.422618],
            [-.275451, .932257, -.23457],
            [.357073, .325773, .875426]])
        q = Quaternion.from_rotation_vector(phi)
        np.testing.assert_allclose(expected, q.matrix, atol=1e-5, rtol=0)

    def test_qmethod(self):
        frame_1 = np.array([[2 / 3, 2 / 3, 1 / 3], [2 / 3, -1 / 3, -2 / 3]])
        frame_2 = np.array([[0.8, 0.6, 0], [-0.6, 0.8, 0]])
        q = Quaternion.from_qmethod(frame_1.T, frame_2.T, np.ones(2))

        for a1 in np.arange(0, 1, .1):
            for a2 in np.arange(0, 1, .1):
                v1 = a1 * frame_1[0] + a2 * frame_1[1]
                v2 = a1 * frame_2[0] + a2 * frame_2[1]
                np.testing.assert_allclose(q.matrix.dot(v1), v2, atol=1e-10)

    def test_qmethod_with_probability(self):
        frame_1 = np.array([[2 / 3, 2 / 3, 1 / 3], [2 / 3, -1 / 3, -2 / 3]])
        frame_2 = np.array([[0.8, 0.6, 0], [-0.6, 0.8, 0]])
        q = Quaternion.from_qmethod(frame_1.T, frame_2.T, np.ones(2))

        for a1 in np.arange(0, 1, .1):
            for a2 in np.arange(0, 1, .1):
                v1 = a1 * frame_1[0] + a2 * frame_1[1]
                v2 = a1 * frame_2[0] + a2 * frame_2[1]
                np.testing.assert_allclose(q.matrix.dot(v1), v2, atol=1e-10)

    def test_ra_dec_roll(self):
        for ra in np.linspace(-170, 180, 8):
            for dec in np.linspace(-90, 90, 8):
                for roll in np.linspace(10, 360, 8):

                    xyz = np.deg2rad(np.array([ra, dec, roll]))
                    c3, c2, c1 = np.cos(xyz)
                    s3, s2, s1 = np.sin(xyz)
                    expected = np.array([
                        [c2 * c3,               -c2 * s3,                 s2],       # noqa
                        [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],  # noqa
                        [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3,  c1 * c2]   # noqa
                    ])

                    obtained = Quaternion.from_ra_dec_roll(ra, dec, roll)

                    np.testing.assert_allclose(expected, obtained.matrix, atol=1e-15)

    def test_to_rdr(self):
        for ra in np.linspace(-170, 170, 8):
            for dec in np.linspace(-88, 88, 8):
                for roll in np.linspace(-170, 170, 8):
                    q = Quaternion.from_ra_dec_roll(ra, dec, roll)

                    np.testing.assert_allclose([ra, dec, roll], q.ra_dec_roll)

    def test_average_easy(self):
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion(-1, 0, 0, 0)
        avg = Quaternion.average(q1, q2)

        np.testing.assert_allclose(q1.coordinates, avg.coordinates)

    def test_average_mild(self):
        q1 = Quaternion.exp(Quaternion(0, .1, .3, .7))
        quats_l = []
        quats_r = []
        for i in np.arange(-.1, .11, .05):
            for j in np.arange(-.1, .11, .05):
                for k in np.arange(-.1, .11, .05):
                    try:
                        q = Quaternion(0, i, j, k)
                    except QuaternionError:  # wrong quaternion
                        continue
                    q = Quaternion.exp(q)
                    quats_l.append(q1 * q)
                    quats_r.append(q * q1)

        avg_l = Quaternion.average(*quats_l)
        avg_r = Quaternion.average(*quats_r)
        np.testing.assert_allclose(q1.coordinates, avg_l.coordinates)
        np.testing.assert_allclose(q1.coordinates, avg_r.coordinates)

    def test_average_weights_easy(self):
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion(-1, 0, 0, 0)
        weights = [1, 1]
        avg = Quaternion.average(q1, q2, weights=weights)
        np.testing.assert_allclose(q1.coordinates, avg.coordinates)

    def test_average_weights_easy_2(self):
        q1 = Quaternion(1, 0, 0, 0)
        q2 = Quaternion(0.707, 0, 0.707, 0)
        weights = [1, 0]
        avg = Quaternion.average(q1, q2, weights=weights)
        np.testing.assert_allclose(q1.coordinates, avg.coordinates)

    def test_average_weights_mild(self):
        q1 = Quaternion.exp(Quaternion(0, .1, .3, .7))
        quats_l = []
        quats_r = []
        weights = []
        for i in np.arange(-.1, .11, .05):
            for j in np.arange(-.1, .11, .05):
                for k in np.arange(-.1, .11, .05):
                    try:
                        q = Quaternion(0, i, j, k)
                    except QuaternionError:  # wrong quaternion
                        continue
                    q = Quaternion.exp(q)
                    quats_l.append(q1 * q)
                    quats_r.append(q * q1)
                    weights.append(1)

        avg_l = Quaternion.average(*quats_l, weights=weights)
        avg_r = Quaternion.average(*quats_r, weights=weights)
        np.testing.assert_allclose(q1.coordinates, avg_l.coordinates)
        np.testing.assert_allclose(q1.coordinates, avg_r.coordinates)

    def test_optical_axis_first(self):
        v1 = np.array([.02, .01, .99])
        v2 = np.array([-.01, .02, .99])
        oaf = Quaternion.OpticalAxisFirst()
        np.testing.assert_allclose([.99, -.02, -.01], oaf.matrix.dot(v1))
        np.testing.assert_allclose([.99, .01, -.02], oaf.matrix.dot(v2))

    def test_distance(self):
        q = Quaternion.from_rotation_vector([.1, .2, .3])

        for rot_x in np.linspace(-np.pi, np.pi, 7):
            for rot_y in np.linspace(-np.pi / 2, np.pi / 2, 3):
                for rot_z in np.linspace(-np.pi / 2, np.pi / 2, 2):

                    rotation = [rot_x, rot_y, rot_z]
                    rot_quat = Quaternion.from_rotation_vector(rotation)
                    q_rot = q * rot_quat

                    expected = np.linalg.norm(rotation) % (2 * np.pi)
                    if expected > np.pi:
                        expected = 2 * np.pi - expected

                    self.assertAlmostEqual(expected, q.distance(q_rot))


class ParameterizedTests(unittest.TestCase):

    @staticmethod
    def ra_dec_to_xyz(ra, dec):
        cr, sr = np.cos(np.radians(ra)), np.sin(np.radians(ra))
        cd, sd = np.cos(np.radians(dec)), np.sin(np.radians(dec))
        return np.array([cr * cd, sr * cd, sd])

    @staticmethod
    def angle_to_xy(angle):
        return np.cos(np.radians(angle)), np.sin(np.radians(angle))

    @staticmethod
    def from_mrp(xyz):
        N = xyz.dot(xyz)

        def inv_proj(x):
            return 4 * x / (4 + N)

        qi, qj, qk = map(inv_proj, xyz)
        qr = (4 - N) / (4 + N)
        return Quaternion(qr, qi, qj, qk)

    @given(floats(min_value=-180, max_value=180),
           floats(min_value=-89, max_value=89),  # avoid singularities in -90 & 90 degs
           floats(min_value=0, max_value=360))
    def test_quat_ra_dec_roll(self, ra, dec, roll):
        q = Quaternion.from_ra_dec_roll(ra, dec, roll)
        ob_ra, ob_dec, ob_roll = q.ra_dec_roll
        np.testing.assert_almost_equal(self.ra_dec_to_xyz(ob_ra, ob_dec),
                                       self.ra_dec_to_xyz(ra, dec))
        np.testing.assert_almost_equal(self.angle_to_xy(ob_roll),
                                       self.angle_to_xy(roll), decimal=2)

    @given(floats(min_value=-2, max_value=2),
           floats(min_value=-2, max_value=2),
           floats(min_value=-2, max_value=2))
    def test_quat_rotation_vector(self, rx, ry, rz):
        # ignore numerically unstable quaternions:
        assume(np.linalg.norm([rx, ry, rz]) > Quaternion.tolerance)
        rot = np.array([rx, ry, rz])
        q = Quaternion.from_rotation_vector(rot)
        distance = np.linalg.norm(rot - q.rotation_vector)

        assert (distance % 2 * np.pi) < 1e-8

    @given(floats(min_value=-1, max_value=1),
           floats(min_value=-1, max_value=1),
           floats(min_value=-1, max_value=1))
    def test_matrix(self, ma, mb, mc):
        q = self.from_mrp(np.array([ma, mb, mc]))
        self.assertTrue(q.is_unitary())

        m = q.matrix
        np.testing.assert_almost_equal(np.identity(3), m.dot(m.T))

        obtained = Quaternion.from_matrix(m)
        self.assertTrue(obtained.is_unitary())

        np.testing.assert_almost_equal(q.positive_representant.coordinates,
                                       obtained.positive_representant.coordinates,
                                       decimal=8)

    @given(floats(min_value=-1, max_value=1),
           floats(min_value=-1, max_value=1),
           floats(min_value=-1, max_value=1))
    def test_log_exp(self, qi, qj, qk):
        # ignore numerically unstable quaternions:
        assume(np.linalg.norm([qi, qj, qk]) > Quaternion.tolerance)
        q = Quaternion(0, qi, qj, qk)
        expq = q.exp()
        qback = expq.log()

        np.testing.assert_almost_equal(q.coordinates,
                                       qback.coordinates,
                                       decimal=8)

    @given(floats(min_value=-5, max_value=5),
           floats(min_value=-5, max_value=5),
           floats(min_value=-5, max_value=5),
           floats(min_value=-5, max_value=5))
    def test_exp_log(self, qr, qi, qj, qk):
        # ignore numerically unstable quaternions:
        assume(np.linalg.norm([qr, qi, qj, qk]) > Quaternion.tolerance)
        q = Quaternion(qr, qi, qj, qk)
        if q.norm() == 0:
            return

        logq = q.log()
        qback = logq.exp()

        np.testing.assert_almost_equal(q.coordinates,
                                       qback.coordinates,
                                       decimal=8)

    @given(floats(min_value=-2, max_value=2),
           floats(min_value=-2, max_value=2),
           floats(min_value=-2, max_value=2))
    def test_from_qmethod(self, rx, ry, rz):
        # ignore numerically unstable quaternions:
        assume(np.linalg.norm([rx, ry, rz]) > Quaternion.tolerance)
        q = Quaternion.from_rotation_vector(np.array([rx, ry, rz]))

        vectors = np.random.normal(scale=1.0, size=(3, 6))
        norms = np.linalg.norm(vectors, axis=0)
        vectors /= norms

        errors = np.random.normal(scale=1e-6, size=(3, 6))
        rotated_vectors = q.matrix.dot(vectors) + errors

        qback = Quaternion.from_qmethod(vectors, rotated_vectors, np.ones(6))
        q_diff = (q / qback).positive_representant

        np.testing.assert_almost_equal(q_diff.coordinates,
                                       Quaternion.Unit().coordinates,
                                       decimal=4)

    @given(floats(min_value=-5, max_value=5),
           floats(min_value=-5, max_value=5),
           floats(min_value=-5, max_value=5),
           floats(min_value=-5, max_value=5))
    def test_eq(self, qr, qi, qj, qk):
        # ignore numerically unstable quaternions:
        assume(np.linalg.norm([qr, qi, qj, qk]) > Quaternion.tolerance)
        q = Quaternion(qr, qi, qj, qk)

        small = Quaternion(1, .5 * Quaternion.tolerance, 0, 0)
        not_small = Quaternion(1, 2 * Quaternion.tolerance, 0, 0)

        assert q == q
        assert q * small == q
        assert q * not_small != q

    @given(floats(min_value=-5, max_value=5),
           floats(min_value=-5, max_value=5),
           floats(min_value=-5, max_value=5),
           floats(min_value=-5, max_value=5))
    def test_invert(self, qr, qi, qj, qk):
        """ verify all inverse methods are identical, and indeed they invert. """
        # ignore numerically unstable quaternions:
        assume(np.linalg.norm([qr, qi, qj, qk]) > Quaternion.tolerance)

        q = Quaternion(qr, qi, qj, qk)
        assert ~q == q.inverse() == q.conjugate()
        assert q * ~q == ~q * q == Quaternion.Unit()


class QuaternionStdDevTests(unittest.TestCase):
    # tolerance is this big because average_and_std_naive gives slightly different results than matlab implementation
    # this may be due to the way weights are taken into account, as in matlab implementation weights were not being used
    tolerance_deg = 1e-3
    matlab_basedir = os.path.join(basedir, 'matlab_results')

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(cls.matlab_basedir, 'matlab_results.json'), 'r') as fid:
            cls.results = json.load(fid)

        cls.quat_diff_matlab_quats = {}
        for star_vectors_noise_arcsec in cls.results.keys():
            quat_diff_matlab = np.loadtxt(
                os.path.join(cls.matlab_basedir, 'quat_diff_ST1_ST2_{}_arcsec.txt'.format(star_vectors_noise_arcsec)))
            cls.quat_diff_matlab_quats[star_vectors_noise_arcsec] = [Quaternion(*q) for q in quat_diff_matlab]

    def test_average_std_naive(self):

        for star_vectors_noise_arcsec in self.results.keys():
            quat_diff_matlab_quats = self.quat_diff_matlab_quats[star_vectors_noise_arcsec]
            _, mean_total_rotation_angle_deg = Quaternion.average_and_std_naive(*quat_diff_matlab_quats)

            assert abs(mean_total_rotation_angle_deg - self.results[star_vectors_noise_arcsec]['mean_total_angle'])\
                   < self.tolerance_deg

    def test_average_std_sigma_lerner(self):
        for star_vectors_noise_arcsec in self.results.keys():
            quat_diff_matlab_quats = self.quat_diff_matlab_quats[star_vectors_noise_arcsec]
            _, sigma_lerner_deg = Quaternion.average_and_std_lerner(*quat_diff_matlab_quats)

            assert abs(sigma_lerner_deg - self.results[star_vectors_noise_arcsec]['sigma_lerner']) \
                   < self.tolerance_deg

    def test_average_and_covariance(self):
        # This tests that the trace of the resultant covariance matrix of the
        # averaged test is around the same value than the input covariance matrix
        # if an individual quaternion divided np.sqrt(N-1), where N is the number of
        # quaternions
        for star_vectors_noise_arcsec in self.results.keys():
            quat_diff_matlab_quats = self.quat_diff_matlab_quats[star_vectors_noise_arcsec]
            sigma_lerner_in_deg = self.results[star_vectors_noise_arcsec]['sigma_lerner']
            _, covariance_rad = Quaternion.average_and_covariance(
                *quat_diff_matlab_quats, R=np.deg2rad(sigma_lerner_in_deg)**2*np.eye(3))
            sigma_lerner_out_deg = np.rad2deg(np.sqrt(np.trace(covariance_rad)/3))\
                        * np.sqrt(len(quat_diff_matlab_quats)-1)

            assert abs(sigma_lerner_in_deg - sigma_lerner_out_deg ) \
                   < self.tolerance_deg

import unittest

from hypothesis import given, assume, strategies
import numpy as np
import pytest
import os
import json

from quaternions import Quaternion, QuaternionError, GeneralQuaternion
from quaternions.general_quaternion import DEFAULT_TOLERANCE, exp, log


ANY_QUATERNION = strategies.lists(elements=strategies.floats(min_value=-5, max_value=5), min_size=4, max_size=4)
ANY_ROTATION_VECTOR = strategies.lists(elements=strategies.floats(min_value=-5, max_value=5), min_size=3, max_size=3)


class QuaternionTest(unittest.TestCase):

    schaub_example_dcm = np.array([[.892539, .157379, -.422618],
                                   [-.275451, .932257, -.23457],
                                   [.357073, .325773, .875426]])

    schaub_result = np.array([.961798, -.14565, .202665, .112505])

    @given(ANY_QUATERNION)
    def test_constructor(self, arr):
        assume(GeneralQuaternion(*arr).norm() > DEFAULT_TOLERANCE)
        q = Quaternion(*arr)
        assert isinstance(q, Quaternion)
        assert q.is_unitary()

    def test_constructor_zero_raises(self):
        with pytest.raises(QuaternionError):
            Quaternion(0, 0, 0, 0)

    @given(ANY_QUATERNION)
    def test_is_equal(self, arr):
        assume(GeneralQuaternion(*arr).norm() > DEFAULT_TOLERANCE)
        q = Quaternion(*arr)
        assert q == q
        assert q == -q
        assert q != q + Quaternion(1, 2, 3, 4)
        assert q == GeneralQuaternion(*q.coordinates)

    @given(ANY_QUATERNION, strategies.floats(min_value=0, max_value=2 * np.math.pi-1e-4))
    def test_distance(self, arr, angle_rad):
        assume(GeneralQuaternion(*arr).norm() > DEFAULT_TOLERANCE)
        q = Quaternion(*arr)
        assert q.distance(-q) == pytest.approx(0) or q.distance(-q) == pytest.approx(2 * np.math.pi)

        for diff in [Quaternion.from_ra_dec_roll(np.degrees(angle_rad), 0, 0),
                     Quaternion.from_ra_dec_roll(0, np.degrees(angle_rad), 0),
                     Quaternion.from_ra_dec_roll(0, 0, np.degrees(angle_rad))]:
            assert q.distance(q * diff) == pytest.approx(angle_rad)
            assert q.distance(diff * q) == pytest.approx(angle_rad)

    @given(ANY_QUATERNION)
    def test_positive_representant(self, arr):
        assume(GeneralQuaternion(*arr).norm() > DEFAULT_TOLERANCE)
        q = Quaternion(*arr)
        assert q.positive_representant == q or q.positive_representant == -q
        assert q.positive_representant == (-q).positive_representant

    @given(ANY_QUATERNION)
    def test_mul(self, arr):
        assume(GeneralQuaternion(*arr).norm() > DEFAULT_TOLERANCE)
        q = Quaternion(*arr)

        # multiply by scalar:
        assert isinstance(2 * q, Quaternion)
        assert isinstance(q * 2, Quaternion)
        assert q == q * 2 == 2 * q  # Note: up to scale; differs from GeneralQuaternion() * 2

        # multiply by Quaternion:
        other = Quaternion(1, 2, 3, 4)
        assert isinstance(q * other, Quaternion)
        assert (q * other).is_unitary()

        # multiply by GeneralQuaternion:
        other = GeneralQuaternion(1, 2, 3, 4)
        for mul in [other * q, q * other]:
            assert isinstance(mul, GeneralQuaternion) and not isinstance(mul, Quaternion)
            assert mul.norm() == pytest.approx(other.norm())

    @given(ANY_QUATERNION)
    def test_rotation_vector(self, arr):
        assume(np.linalg.norm(np.asarray(arr[1:])) > DEFAULT_TOLERANCE)
        q = Quaternion(*arr)

        complex_part = np.array([q.qi, q.qj, q.qk])
        complex_norm = np.linalg.norm(complex_part)

        # test alternative, direct way to calculate rotation axis:
        np.testing.assert_allclose(complex_part / complex_norm, q.rotation_axis())

        # test alternative, direct way to calculate rotation angle:
        angle = 2 * np.math.atan2(complex_norm, q.qr)
        assert angle == pytest.approx(q.rotation_angle())

        # test rotation of q^2 is 2*rotation:
        assert q*q == Quaternion.from_rotation_vector(2 * np.asarray(q.rotation_vector))

    @given(ANY_QUATERNION)
    def test_from_rotation_vector(self, arr):
        assume(np.linalg.norm(np.asarray(arr[1:])) > DEFAULT_TOLERANCE)
        q = Quaternion(*arr)
        assert q.from_rotation_vector(q.rotation_vector) == q

    def test_rotate_vector_schaub(self):
        q1 = exp(Quaternion(0, .1, .02, -.3))
        vector = QuaternionTest.schaub_example_dcm[:, 1]
        rotated_vector = q1 * vector
        np.testing.assert_allclose(rotated_vector, q1.matrix.dot(vector), atol=1e-5, rtol=0)

    def test_from_matrix_schaub(self):
        q = Quaternion.from_matrix(QuaternionTest.schaub_example_dcm)
        np.testing.assert_allclose(QuaternionTest.schaub_result, q.coordinates, atol=1e-5, rtol=0)

    @given(ANY_QUATERNION)
    def test_matrix(self, arr):
        assume(GeneralQuaternion(*arr).norm() > DEFAULT_TOLERANCE)
        q = Quaternion(*arr)
        m = q.matrix
        np.testing.assert_almost_equal(np.identity(3), m.dot(m.T))

    @given(ANY_QUATERNION)
    def test_from_matrix(self, arr):
        assume(GeneralQuaternion(*arr).norm() > DEFAULT_TOLERANCE)
        q = Quaternion(*arr)
        assert q.from_matrix(q.matrix) == q

    @given(ANY_QUATERNION)
    def test_basis(self, arr):
        assume(GeneralQuaternion(*arr).norm() > DEFAULT_TOLERANCE)
        q = Quaternion(*arr)
        b1, b2, b3 = q.basis
        assert np.array_equal([b1, b2, b3], q.matrix)

    @given(ANY_ROTATION_VECTOR)
    def test_from_ra_dec_roll(self, arr):
        xyz = np.deg2rad(arr)
        c3, c2, c1 = np.cos(xyz)
        s3, s2, s1 = np.sin(xyz)
        expected = (np.array([
            [c2 * c3,               -c2 * s3,                 s2],       # noqa
            [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],  # noqa
            [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3,  c1 * c2]   # noqa
        ]))

        assert Quaternion.from_ra_dec_roll(*arr) == Quaternion.from_matrix(expected)

    @given(ANY_QUATERNION)
    def test_ra_dec_roll(self, arr):
        assume(np.linalg.norm(arr) > 3 * DEFAULT_TOLERANCE)
        q = Quaternion(*arr)
        ra, dec, roll = q.ra_dec_roll
        assume(abs(abs(dec) - 90) > 1e-3)  # avoid singularity at dec==+_90
        assert Quaternion.from_ra_dec_roll(ra, dec, roll) == q

    def test_qmethod(self):
        v1, v2 = [2 / 3, 2 / 3, 1 / 3], [2 / 3, -1 / 3, -2 / 3]
        w1, w2 = [0.8, 0.6, 0], [-0.6, 0.8, 0]
        q = Quaternion.from_qmethod(np.array([v1, v2]).T, np.array([w1, w2]).T)

        np.testing.assert_allclose(q(v1), w1, atol=1e-10)
        np.testing.assert_allclose(q(v2), w2, atol=1e-10)

    @given(ANY_ROTATION_VECTOR)
    def test_from_qmethod_with_noise(self, r):
        assume(np.linalg.norm(r) > Quaternion.tolerance)
        q = Quaternion.from_rotation_vector(r)

        vectors = np.random.normal(scale=1.0, size=(3, 6))
        norms = np.linalg.norm(vectors, axis=0)
        vectors /= norms

        noise_sigma = 1e-6
        errors = np.random.normal(scale=noise_sigma, size=(3, 6))
        rotated_vectors = q.matrix.dot(vectors) + errors

        q_calculated = Quaternion.from_qmethod(vectors, rotated_vectors, np.ones(6))
        assert q.is_equal(q_calculated, tolerance=10*noise_sigma)

    @given(ANY_ROTATION_VECTOR)
    def test_optical_axis_first(self, v):
        oaf = Quaternion.OpticalAxisFirst()
        np.testing.assert_allclose(oaf(v), [v[2], -v[0], -v[1]])

    def test_type(self):
        # Unit quaternion can be unitary or general:
        assert isinstance(GeneralQuaternion.unit(), GeneralQuaternion)
        assert isinstance(Quaternion.unit(), Quaternion)

        # Unit quaternion can not be unitary:
        assert isinstance(GeneralQuaternion.zero(), GeneralQuaternion)
        assert not isinstance(Quaternion.zero(), Quaternion)
        assert isinstance(Quaternion.zero(), GeneralQuaternion)

        assert isinstance(exp(GeneralQuaternion(1, 2, 3, 4)), GeneralQuaternion)
        assert isinstance(exp(Quaternion(1, 2, 3, 4)), Quaternion)

        assert isinstance(log(Quaternion(1, 2, 3, 4)), GeneralQuaternion)
        assert not isinstance(log(Quaternion(1, 2, 3, 4)), Quaternion)

    def test_exp_identity(self):
        assert exp(GeneralQuaternion.zero()) == Quaternion.unit()

    def test_log_identity(self):
        assert log(Quaternion.unit()) == GeneralQuaternion.zero()

    @given(ANY_QUATERNION)
    def test_exp_log(self, arr):
        assume(np.linalg.norm(arr) > DEFAULT_TOLERANCE)
        q = GeneralQuaternion(*arr).normalized()
        assert exp(log(q)) == q
        assert (log(exp(q)).imaginary.tolist() == pytest.approx(q.imaginary.tolist()))  # log defined up to real

    NUM_ELEMENTS = 25

    @given(strategies.lists(elements=strategies.floats(min_value=-5, max_value=5),
                            min_size=4+4*NUM_ELEMENTS, max_size=4+4*NUM_ELEMENTS))
    def test_average(self, arr):

        q = GeneralQuaternion(*arr[:4])
        assume(q.norm() > DEFAULT_TOLERANCE)  # ignore quaternions of norm==0, whose inverse is numerically unstable

        q = q.normalized()
        randoms = [GeneralQuaternion(*arr[4*i: 4*i+4]) for i in range(1, self.NUM_ELEMENTS+1)]
        q_with_noise = [q + n * (.1 * DEFAULT_TOLERANCE) for n in randoms]

        # test without weights:
        average = Quaternion.average(*q_with_noise)
        assert average == q or average == -q

        # test with weights:
        weights = [1] + (self.NUM_ELEMENTS-1) * [0]  # only uses q_with_noise[0]
        average = Quaternion.average(*q_with_noise, weights=weights)
        assert average == q_with_noise[0] or average == -q_with_noise[0]

    def test_apply(self):
        q = Quaternion(1, 2, 3, 4)

        v = [1, 2, 3]
        assert (q(v) == q * v).all()
        assert isinstance(q(v), np.ndarray)
        assert len(q(v)) == 3

    def test_apply_wrong_type(self):
        q = Quaternion(1, 2, 3, 4)
        with pytest.raises(QuaternionError):
            q([1, 2])
        with pytest.raises(QuaternionError):
            q({1, 2, 3})

    def test_repr(self):
        unit_quat = Quaternion(1, 2, 3, 4)
        assert repr(unit_quat).startswith('Quaternion(')
        assert eval(repr(unit_quat)) == unit_quat


class QuaternionStdDevTests(unittest.TestCase):
    # tolerance is this big because average_and_std_naive gives slightly different results than matlab implementation
    # this may be due to the way weights are taken into account, as in matlab implementation weights were not being used
    tolerance_deg = 1e-3
    basedir = os.path.join(os.path.split(os.path.abspath(__file__))[0], 'data')
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

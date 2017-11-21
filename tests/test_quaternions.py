import unittest

from hypothesis import given, assume, strategies
import numpy as np
import pytest

from quaternions import Quaternion, QuaternionError, GeneralQuaternion
from quaternions.general_quaternion import DEFAULT_TOLERANCE


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

    @given(ANY_QUATERNION)
    def test_is_equal(self, arr):
        assume(GeneralQuaternion(*arr).norm() > DEFAULT_TOLERANCE)
        q = Quaternion(*arr)
        assert q.distance(q) == pytest.approx(0)
        assert q.distance(-q) == pytest.approx(0)

    @given(ANY_QUATERNION)
    def test_is_equal(self, arr):
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
        q1 = Quaternion.exp([0, .1, .02, -.3])
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
        assert np.array_equal([*q.basis], q.matrix)

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
        assume(np.linalg.norm(arr) > DEFAULT_TOLERANCE)
        q = Quaternion(*arr)
        assert Quaternion.from_ra_dec_roll(*q.ra_dec_roll) == q

    def test_qmethod(self):
        v1, v2 = [2 / 3, 2 / 3, 1 / 3], [2 / 3, -1 / 3, -2 / 3]
        w1, w2 = [0.8, 0.6, 0], [-0.6, 0.8, 0]
        q = Quaternion.from_qmethod(np.array([v1, v2]).T, np.array([w1, w2]).T, np.ones(2))

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
        assert isinstance(GeneralQuaternion.unit(), GeneralQuaternion)
        assert isinstance(Quaternion.unit(), Quaternion)
        assert isinstance(GeneralQuaternion.exp([1, 2, 3, 4]), GeneralQuaternion)
        assert isinstance(Quaternion.exp([1, 2, 3, 4]), Quaternion)

    def test_exp_identity(self):
        assert Quaternion.exp(GeneralQuaternion.zero()) == Quaternion.unit()

    def test_log_identity(self):
        assert Quaternion.log(Quaternion.unit()) == GeneralQuaternion.zero()

    @given(ANY_QUATERNION)
    def test_exp_2ways(self, arr):
        assume(np.linalg.norm(arr) > Quaternion.tolerance)
        q = GeneralQuaternion(*arr).normalized()
        assert GeneralQuaternion.exp(q) == GeneralQuaternion.exp(q.coordinates)

    @given(ANY_QUATERNION)
    def test_exp_log(self, arr):
        assume(np.linalg.norm(arr) > DEFAULT_TOLERANCE)
        for q in [GeneralQuaternion(*arr).normalized(), Quaternion(*arr)]:  # both ways are supported
            assert Quaternion.log(GeneralQuaternion.exp(q)) == q
            assert GeneralQuaternion.exp(Quaternion.log(q)) == q

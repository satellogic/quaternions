import unittest

import numpy as np
from hypothesis import given, assume, strategies
import pytest

from quaternions.general_quaternion import GeneralQuaternion, QuaternionError, DEFAULT_TOLERANCE, exp, log


ANY_QUATERNION = strategies.lists(elements=strategies.floats(min_value=-5, max_value=5), min_size=4, max_size=4)


class GeneralQuaternionTest(unittest.TestCase):

    @given(ANY_QUATERNION)
    def test_equal(self, arr):
        q = GeneralQuaternion(*arr)
        assert q == q
        assert q == q + GeneralQuaternion(0.9 * DEFAULT_TOLERANCE, 0, 0, 0)
        assert q != q + GeneralQuaternion(1.1 * DEFAULT_TOLERANCE, 0, 0, 0)

    @given(ANY_QUATERNION)
    def test_real_imaginary(self, arr):
        q = GeneralQuaternion(*arr)
        i, j, k = q.imaginary
        assert (q.coordinates == [q.real, i, j, k]).all()

    @given(ANY_QUATERNION)
    def test_raises(self, arr):
        q = GeneralQuaternion(*arr)
        with pytest.raises(QuaternionError):
            q + 3

    @given(ANY_QUATERNION)
    def test_arithmetics(self, arr):
        q = GeneralQuaternion(*arr)
        assert q + q == 2 * q == q * 2
        assert q - q == GeneralQuaternion.zero()
        assert q * GeneralQuaternion.zero() == q * 0 == GeneralQuaternion.zero()
        assert q * GeneralQuaternion.unit() == q * 1 == q
        assert -(-q) == q

    @given(ANY_QUATERNION, ANY_QUATERNION)
    def test_sum_commutative(self, arr1, arr2):
        q1 = GeneralQuaternion(*arr1)
        q2 = GeneralQuaternion(*arr2)
        assert q1 + q2 == q2 + q1
        assert q1 - q2 == - (q2 - q1)

    @given(ANY_QUATERNION)
    def test_conjugate(self, arr):
        q = GeneralQuaternion(*arr)
        assert (q + q.conjugate()).is_real()

    @given(ANY_QUATERNION)
    def test_inverse(self, arr):
        q = GeneralQuaternion(*arr)
        assume(q.norm() > DEFAULT_TOLERANCE)  # ignore quaternions of norm==0, whose inverse is numerically unstable
        assert q * q.inverse() == q.inverse() * q == GeneralQuaternion.unit()
        assert q * ~q == ~q * q == GeneralQuaternion.unit()

    @given(ANY_QUATERNION)
    def test_distance(self, arr):
        q = GeneralQuaternion(*arr)
        assert q.euclidean_distance(q) == pytest.approx(0)
        assert q.norm() == q.euclidean_distance(GeneralQuaternion.zero()) == q.euclidean_distance(2 * q)

    @given(ANY_QUATERNION)
    def test_normalized(self, arr):
        q = GeneralQuaternion(*arr)
        assume(q.norm() > DEFAULT_TOLERANCE)  # ignore quaternions of norm==0, whose inverse is numerically unstable
        assert q.normalized().norm() == pytest.approx(1, DEFAULT_TOLERANCE)

    @given(ANY_QUATERNION)
    def test_is_unitary(self, arr):
        q = GeneralQuaternion(*arr)
        assume(q.norm() > DEFAULT_TOLERANCE)  # ignore quaternions of norm==0, whose inverse is numerically unstable
        assert q.normalized().is_unitary()
        assert not (2 * q.normalized()).is_unitary()

    @given(ANY_QUATERNION)
    def test_coordinates(self, arr):
        q = GeneralQuaternion(*arr)
        assert q == GeneralQuaternion(*q.coordinates)

    @given(ANY_QUATERNION)
    def test_print(self, arr):
        """ make sure all coordinates are printed. """
        q = GeneralQuaternion(*arr)
        for elem in q.coordinates:
            expected_string = '{elem:.6g}'.format(**{'elem': elem})
            assert expected_string in str(q)

    def test_exp_identity(self):
        assert exp(GeneralQuaternion.zero()) == GeneralQuaternion.unit()

    def test_log_identity(self):
        assert log(GeneralQuaternion.unit()) == GeneralQuaternion.zero()

    @given(ANY_QUATERNION)
    def test_exp_norm(self, arr1):
        q1 = GeneralQuaternion(*arr1)
        assert exp(q1).norm() == pytest.approx(np.exp(q1.qr))  # |exp(q)| == exp(real(q)|

    @given(ANY_QUATERNION)
    def test_exp_log(self, arr):
        assume(np.linalg.norm(arr) > DEFAULT_TOLERANCE)
        q = GeneralQuaternion(*arr).normalized()
        assert exp(log(q)) == q
        assert log(exp(q)) == GeneralQuaternion(*q.coordinates)

    @given(ANY_QUATERNION)
    def test_exp_identical_both_ways(self, arr):
        q = GeneralQuaternion(*arr)
        assert exp(q) == q.exp()

    @given(ANY_QUATERNION)
    def test_log_identical_both_ways(self, arr):
        assume(np.linalg.norm(arr) > DEFAULT_TOLERANCE)
        q = GeneralQuaternion(*arr)
        assert log(q) == q.log()

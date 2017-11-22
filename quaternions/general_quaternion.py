import numpy as np


class QuaternionError(Exception):
    pass


DEFAULT_TOLERANCE = 1e-8


class GeneralQuaternion(object):
    """
    represents a quaternion, considered a member in Lie algebra of quaternions.
    for backward compatibility shares the same interface with Quaternion.
    unlike quaternion, it's not defined up to scale
    """
    def __init__(self, qr, qi, qj, qk):
        self.qr = qr
        self.qi = qi
        self.qj = qj
        self.qk = qk

    def __add__(self, p):
        if not is_quaternion(p):
            raise QuaternionError('expected quaternion, got %s' % p.__class__.__name__)
        return GeneralQuaternion(self.qr + p.qr, self.qi + p.qi, self.qj + p.qj, self.qk + p.qk)

    def __sub__(self, p):
        if not is_quaternion(p):
            raise QuaternionError('expected quaternion, got %s' % p)
        return GeneralQuaternion(self.qr - p.qr, self.qi - p.qi, self.qj - p.qj, self.qk - p.qk)

    def __neg__(self):
        return self.__class__(-self.qr, -self.qi, -self.qj, -self.qk)

    def __mul__(self, p):
        if is_quaternion(p):
            mat = np.array([
                [self.qr, -self.qi, -self.qj, -self.qk],  # noqa
                [self.qi,  self.qr,  self.qk, -self.qj],  # noqa
                [self.qj, -self.qk,  self.qr,  self.qi],  # noqa
                [self.qk,  self.qj, -self.qi,  self.qr]   # noqa
            ])
            result = mat.dot(np.array(p.coordinates))
            return self.__class__(*result)
        else:
            return self.__class__(self.qr * p, self.qi * p, self.qj * p, self.qk * p)

    def __rmul__(self, p):
        return self.__mul__(p)

    def __truediv__(self, p):
        return self * (1 / p)

    def __rtruediv__(self, p):
        return p * self.inverse()

    def conjugate(self):
        return self.__class__(self.qr, -self.qi, -self.qj, -self.qk)

    def inverse(self):
        return self.conjugate() * (1 / self._squarenorm())

    def __invert__(self):
        return self.inverse()

    def _squarenorm(self):
        return self.qr * self.qr + self.qi * self.qi + self.qj * self.qj + self.qk * self.qk

    def __repr__(self):
        return 'GeneralQuaternion{}'.format(tuple(self.coordinates))

    def __str__(self):
        return '({qr:.6g}{qi:+.6g}i{qj:+.6g}j{qk:+.6g}k)'.format(**self.__dict__)

    def is_real(self, tolerance=DEFAULT_TOLERANCE):
        """ True if i, j, k components are zero. """
        complex_norm = np.linalg.norm([self.qi, self.qj, self.qk])
        return complex_norm < tolerance

    def is_equal(self, other, tolerance=DEFAULT_TOLERANCE):
        """
        compares as quaternions up to tolerance.
        Note: tolerance in coords, not in quaternions metrics.
        Note: unlike quaternions, equality is not up to scale.
        """
        return np.linalg.norm(self.coordinates - other.coordinates) < tolerance

    def __eq__(self, other):
        return self.is_equal(other)

    def norm(self):
        return np.sqrt(self._squarenorm())

    def normalized(self):
        return self / self.norm()

    def euclidean_distance(self, other):
        """ Returns the euclidean distance between two quaternions. Note: differs from unitary quaternions distance. """
        return (self - other).norm()

    def is_unitary(self, tolerance=DEFAULT_TOLERANCE):
        return abs(self.norm() - 1) < tolerance

    @property
    def coordinates(self):
        return np.array([self.qr, self.qi, self.qj, self.qk])

    @property
    def basis(self):
        qr, qi, qj, qk = self.coordinates
        b0 = np.array([
            qr ** 2 + qi ** 2 - qj ** 2 - qk ** 2,
            2 * qr * qk + 2 * qi * qj,
            -2 * qr * qj + 2 * qi * qk
        ])
        b1 = np.array([
            -2 * qr * qk + 2 * qi * qj,
            qr ** 2 - qi ** 2 + qj ** 2 - qk ** 2,
            2 * qr * qi + 2 * qj * qk
        ])
        b2 = np.array([
            2 * qr * qj + 2 * qi * qk,
            -2 * qr * qi + 2 * qj * qk,
            qr ** 2 - qi ** 2 - qj ** 2 + qk ** 2
        ])
        return b0, b1, b2

    @classmethod
    def _first_eigenvector(cls, matrix):
        """ matrix must be a 4x4 symmetric matrix. """
        vals, vecs = np.linalg.eigh(matrix)
        # q is the eigenvec with heighest eigenvalue (already normalized)
        q = vecs[:, -1]
        if q[0] < 0:
            q = -q
        return cls(*q)

    @staticmethod
    def average(*quaternions, weights=None):
        """
        Return the quaternion such that its matrix minimizes the square distance
        to the matrices of the quaternions in the argument list.

        See Averaging Quaternions, by Markley, Cheng, Crassidis, Oschman.
        """
        b = np.array([q.coordinates for q in quaternions])
        if weights is None:
            weights = np.ones(len(quaternions))
        m = b.T.dot(np.diag(weights)).dot(b)

        return GeneralQuaternion._first_eigenvector(m)

    @classmethod
    def unit(cls):
        return cls(1, 0, 0, 0)

    @classmethod
    def zero(cls):
        return cls(0, 0, 0, 0)

    @classmethod
    def exp(cls, q):
        """
        exponent quaternion
        :param q: list of 4 items or Quaternion (or any derived)
        :return: Quaternion (or any derived)
        """
        if is_quaternion(q):
            real, imag = q.coordinates[0], q.coordinates[1:]
        else:
            real, imag = q[0], np.asarray(q[1:])

        exp_norm = np.exp(real)

        imag_norm = np.linalg.norm(imag)
        if imag_norm == 0:
            return cls(exp_norm, 0, 0, 0)

        j, k, l = np.sin(imag_norm) * imag / imag_norm
        return exp_norm * cls(np.cos(imag_norm), j, k, l)


def is_quaternion(q):
    return issubclass(type(q), GeneralQuaternion)

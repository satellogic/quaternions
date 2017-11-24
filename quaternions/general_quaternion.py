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

    @property
    def real(self):
        return self.qr

    @property
    def imaginary(self):
        return np.array([self.qi, self.qj, self.qk])

    def __add__(self, p):
        validate_is_quaternion(p)
        return GeneralQuaternion(self.qr + p.qr, self.qi + p.qi, self.qj + p.qj, self.qk + p.qk)

    def __sub__(self, p):
        validate_is_quaternion(p)
        return GeneralQuaternion(self.qr - p.qr, self.qi - p.qi, self.qj - p.qj, self.qk - p.qk)

    def __neg__(self):
        return self.__class__(-self.qr, -self.qi, -self.qj, -self.qk)

    def __mul__(self, p):
        if isinstance(p, GeneralQuaternion):
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

    @classmethod
    def unit(cls):
        return cls(1, 0, 0, 0)

    @classmethod
    def zero(cls):
        return GeneralQuaternion(0, 0, 0, 0)

    def exp(self):
        return exp(self)

    def log(self):
        return log(self)


def validate_is_quaternion(q):
    if not isinstance(q, GeneralQuaternion):
        raise QuaternionError('expected quaternion, got %s' % q.__class__.__name__)


def exp(q):
    """
    exponent quaternion
    :param q: GeneralQuaternion (or any derived)
    :return: same class as q (GeneralQuaternion or any derived)
    """
    validate_is_quaternion(q)
    cls = type(q)

    exp_norm = np.exp(q.real)

    imag_norm = np.linalg.norm(q.imaginary)
    if imag_norm == 0:
        i, j, k = 0, 0, 0
    else:
        i, j, k = np.sin(imag_norm) * q.imaginary / imag_norm
    q_exp = GeneralQuaternion(*(exp_norm * np.array([np.cos(imag_norm), i, j, k])))
    return cls(*q_exp.coordinates)  # to enable derived classes


def log(q):
    """
    logarithm of quaternion
    :param q: GeneralQuaternion (or any derived)
    :return: GeneralQuaternion
    """
    validate_is_quaternion(q)

    norm = q.norm()
    imag = np.array((q.qi, q.qj, q.qk)) / norm
    imag_norm = np.linalg.norm(imag)
    if imag_norm == 0:
        i, j, k = 0 if q.qr > 0 else np.pi, 0, 0
    else:
        i, j, k = imag / imag_norm * np.arctan2(imag_norm, q.qr / norm)
    return GeneralQuaternion(np.log(norm), i, j, k)

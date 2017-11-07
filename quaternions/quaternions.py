import functools
import numpy as np
from collections import Iterable


class QuaternionError(Exception):
    pass


class Quaternion(object):
    ''' A class that holds quaternions. It actually holds Q^op, as
    this is the way Schaub-Jenkins work with them.
    '''
    tolerance = 1e-8

    def __init__(self, qr, qi, qj, qk, validate_numeric_stability=True):
        self.qr = qr
        self.qi = qi
        self.qj = qj
        self.qk = qk

        if validate_numeric_stability:
            if self._squarenorm() < self.tolerance * self.tolerance:
                raise QuaternionError('provided numerically unstable quaternion: %s' % self)

    def __add__(self, p):
        assert isinstance(p, Quaternion)
        return Quaternion(self.qr + p.qr, self.qi + p.qi, self.qj + p.qj, self.qk + p.qk)

    def __sub__(self, p):
        assert isinstance(p, Quaternion)
        return Quaternion(self.qr - p.qr, self.qi - p.qi, self.qj - p.qj, self.qk - p.qk)

    def __neg__(self):
        return Quaternion(-self.qr, -self.qi, -self.qj, -self.qk)

    def __mul__(self, p):
        if isinstance(p, Quaternion):
            mat = np.array([
                [self.qr, -self.qi, -self.qj, -self.qk],  # noqa
                [self.qi,  self.qr,  self.qk, -self.qj],  # noqa
                [self.qj, -self.qk,  self.qr,  self.qi],  # noqa
                [self.qk,  self.qj, -self.qi,  self.qr]   # noqa
            ])
            result = mat.dot(np.array(p.coordinates))
            return Quaternion(*result)
        elif isinstance(p, Iterable):
            return self.matrix.dot(p)
        else:
            return Quaternion(self.qr * p, self.qi * p, self.qj * p, self.qk * p)

    def __rmul__(self, p):
        return self.__mul__(p)

    def __truediv__(self, p):
        return self * (1 / p)

    def __rtruediv__(self, p):
        return p * self.inverse()

    def conjugate(self):
        return Quaternion(self.qr, -self.qi, -self.qj, -self.qk)

    def inverse(self):
        return self.conjugate() * (1 / self._squarenorm())

    def __invert__(self):
        return self.inverse()

    def normalized(self):
        return self / np.sqrt(self._squarenorm())

    def _squarenorm(self):
        return self.qr * self.qr + self.qi * self.qi + self.qj * self.qj + self.qk * self.qk

    def __repr__(self):
        return 'Quaternion{}'.format(self.coordinates)

    def __str__(self):
        return '({qr:.6g}{qi:+.6g}i{qj:+.6g}j{qk:+.6g}k)'.format(**self.__dict__)

    def is_equal(self, other):
        """
        compares as quaternions up to tolerance.
        Note: tolerance in coords, not in quaternions metrics.
        """
        q1 = np.asarray(self.normalized().coordinates)
        q2 = np.asarray(other.normalized().coordinates)
        dist = min(np.linalg.norm(q1 - q2), np.linalg.norm(q1 - (-q2)))
        return dist < self.tolerance

    def __eq__(self, other):
        return self.is_equal(other)

    def norm(self):
        return np.sqrt(self._squarenorm())

    def exp(self):
        exp_norm = np.exp(self.qr)

        imag = np.array([self.qi, self.qj, self.qk])
        imag_norm = np.linalg.norm(imag)
        if imag_norm == 0:
            return Quaternion(exp_norm, 0, 0, 0)

        imag_renorm = np.sin(imag_norm) * imag / imag_norm
        q = Quaternion(np.cos(imag_norm), *imag_renorm)

        return exp_norm * q

    def log(self):
        norm = self.norm()
        imag = np.array((self.qi, self.qj, self.qk)) / norm
        imag_norm = np.linalg.norm(imag)
        if imag_norm == 0:
            i_part = 0 if self.qr > 0 else np.pi
            return Quaternion(np.log(norm), i_part, 0, 0)
        imag = imag / imag_norm * np.arctan2(imag_norm, self.qr / norm)
        return Quaternion(np.log(norm), *imag)

    def distance(self, other):
        '''Returns the distance in radians between two unitary quaternions'''
        quot = (self * other.conjugate()).positive_representant
        return 2 * quot.log().norm()

    def is_unitary(self):
        return abs(self._squarenorm() - 1) < self.tolerance

    @property
    def coordinates(self):
        return self.qr, self.qi, self.qj, self.qk

    @property
    def positive_representant(self):
        '''Unitary quaternions q and -q correspond to the same element in SO(3).
        In order to perform some computations (v.g., distance), it is important
        to fix one of them.

        Though the following computations can be done for any quaternion, we allow them
        only for unitary ones.
        '''

        assert self.is_unitary(), 'This method makes sense for unitary quaternions'

        for coord in self.coordinates:
            if coord > 0:
                return self
            if coord < 0:
                return -self
        # add a return here if you remove the assert

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

    @property
    def matrix(self):
        qr, qi, qj, qk = self.coordinates
        return np.array([
            [qr * qr + qi * qi - qj * qj - qk * qk,
                2 * (qi * qj + qr * qk),
                2 * (qi * qk - qr * qj)],
            [2 * (qi * qj - qr * qk),
                qr * qr - qi * qi + qj * qj - qk * qk,
                2 * (qj * qk + qr * qi)],
            [2 * (qi * qk + qr * qj),
                2 * (qj * qk - qr * qi),
                qr * qr - qi * qi - qj * qj + qk * qk]
        ])

    @property
    def rotation_vector(self):
        return (2 * self.log()).coordinates[1:]

    @property
    def ra_dec_roll(self):
        '''Returns ra, dec, roll for quaternion.
        The Euler angles are those called Tait-Bryan XYZ, as defined in
        https://en.wikipedia.org/wiki/Euler_angles#Tait-Bryan_angles
        '''
        m = self.matrix
        ra_rad = np.arctan2(-m[0][1], m[0][0])
        dec_rad = np.arctan2(m[0][2], np.sqrt(m[1][2] ** 2 + m[2][2] ** 2))
        roll_rad = np.arctan2(-m[1][2], m[2][2])
        return np.rad2deg(np.array([ra_rad, dec_rad, roll_rad]))

    @property
    def astrometry_ra_dec_roll(self):
        '''Returns ra, dec, roll as reported by astrometry.
        Notice that Tetra gives a different roll angle, so this is not
        a fixed standard.
        '''
        twisted = self.OpticalAxisFirst() * self
        ra, dec, roll = twisted.ra_dec_roll
        return np.array([-ra, dec, roll - 180])

    @staticmethod
    def from_matrix(mat):
        '''
        Returns the quaternion corresponding to the unitary matrix mat
        '''
        mat = np.array(mat)
        tr = np.trace(mat)
        d = 1 + 2 * mat.diagonal() - tr
        qsquare = 1 / 4 * np.array([1 + tr, d[0], d[1], d[2]])
        qsquare = qsquare.clip(0, None)  # avoid numerical errors
        # compute signs matrix
        signs = np.sign([mat[1, 2] - mat[2, 1], mat[2, 0] - mat[0, 2], mat[0, 1] - mat[1, 0],
                         mat[0, 1] + mat[1, 0], mat[2, 0] + mat[0, 2], mat[1, 2] + mat[2, 1]])
        signs_m = np.zeros((4, 4))
        signs_m[np.triu_indices(4, 1)] = signs
        signs_m += signs_m.T
        signs_m[np.diag_indices(4)] = 1.
        # choose appropriate signs
        max_idx = qsquare.argmax()
        coords = np.sqrt(qsquare) * signs_m[max_idx]
        return Quaternion(*coords)

    @staticmethod
    def from_rotation_vector(xyz):
        '''
        Returns the quaternion corresponding to the rotation xyz.
        Explicitly: rotation occurs along the axis xyz and has angle
        norm(xyz)

        This corresponds to the exponential of the quaternion with
        real part 0 and imaginary part 1/2 * xyz.
        '''
        xyz_half = .5 * np.array(xyz)
        return Quaternion(0, *xyz_half).exp()

    @staticmethod
    def _first_eigenvector(matrix):
        '''matrix must be a 4x4 symmetric matrix'''
        vals, vecs = np.linalg.eigh(matrix)
        # q is the eigenvec with heighest eigenvalue (already normalized)
        q = vecs[:, -1]
        if q[0] < 0:
            q = -q
        return Quaternion(*q)

    @staticmethod
    def from_qmethod(source, target, probabilities=None):
        '''
        Returns the quaternion corresponding to solving with qmethod.

        See: Closed-form solution of absolute orientation using unit quaternions,
        Berthold K. P. Horn,
        J. Opt. Soc. Am. A, Vol. 4, No. 4, April 1987

        It "sends" the (3xn) matrix source to the (3xn) matrix target.
        Vectors are multiplied by probabilities too, if available.

        "sends" means that if q = Quaternion.from_qmethod(s, t)
        then q.matrix will be a rotation matrix (not a coordinate changing matrix).
        In other words, q.matrix.dot(s) ~ t

        The method can also produce the change of basis quaternion
        in this way: assume that there are vectors v1,..., vn for which we have coordinates
        in two frames, F1 and F2.
        If s and t are the 3xn matrices of v1,..., vn in frames F1 and F2, then
        Quaternion.from_qmethod(s, t) is the quaternion corresponding to the change of basis
        from F1 to F2.
        '''
        if probabilities is not None:
            B = source.dot(np.diag(probabilities)).dot(target.T)
        else:
            B = source.dot(target.T)
        sigma = np.trace(B)
        S = B + B.T
        Z = B - B.T
        i, j, k = Z[2, 1], Z[0, 2], Z[1, 0]
        K = np.zeros((4, 4))
        K[0] = [sigma, i, j, k]
        K[1:4, 0] = [i, j, k]
        K[1:4, 1:4] = S - sigma * np.identity(3)
        return Quaternion._first_eigenvector(K)

    @staticmethod
    def average(*quaternions, weights=None):
        '''
        Return the quaternion such that its matrix minimizes the square distance
        to the matrices of the quaternions in the argument list.

        See Averaging Quaternions, by Markley, Cheng, Crassidis, Oschman.
        '''
        B = np.array([q.coordinates for q in quaternions])
        if not weights:
            weights = np.ones(len(quaternions))
        M = B.T.dot(np.diag(weights)).dot(B)

        return Quaternion._first_eigenvector(M)

    @staticmethod
    def Unit():
        return Quaternion(1, 0, 0, 0)

    @staticmethod
    def integrate_from_velocity_vectors(vectors):
        '''vectors must be an iterable of 3-d vectors.
        This method just exponentiates all vectors/2, multiplies them and takes 2*log.
        Thus, the return value corresponds to the resultant rotation vector of a body
        under all rotations in the iterable.
        '''
        qs = list(map(Quaternion.from_rotation_vector, vectors))[::-1]
        prod = functools.reduce(Quaternion.__mul__, qs, Quaternion.Unit())
        return prod.rotation_vector

    @staticmethod
    def from_ra_dec_roll(ra, dec, roll):
        '''constructs a quaternion from ra/dec/roll params
        using Tait-Bryan angles XYZ.

        ra stands for right ascencion, and usually lies in [0, 360]
        dec stands for declination, and usually lies in [-90, 90]
        roll stands for rotation/rolling, and usually lies in [0, 360]
        '''
        raq = Quaternion.exp(Quaternion(0, 0, 0, -np.deg2rad(ra) / 2,
                                        validate_numeric_stability=False))
        decq = Quaternion.exp(Quaternion(0, 0, -np.deg2rad(dec) / 2, 0,
                                         validate_numeric_stability=False))
        rollq = Quaternion.exp(Quaternion(0, -np.deg2rad(roll) / 2, 0, 0,
                                          validate_numeric_stability=False))
        return rollq * decq * raq

    @staticmethod
    def OpticalAxisFirst():
        '''
        This quaternion is useful for changing from camera coordinates in
        two standard frames:

        Let the sensor plane have axes
        R (pointing horizontally to the right)
        D (pointing vertically down)
        and let P be the optical axis, pointing "outwards", i.e., from the
        focus to the center of the focal plane.

        One typical convention is taking the frame [R, D, P].
        The other one is taking the frame [P, -R, -D].

        This quaternion gives the change of basis from the first to the second.
        '''
        return Quaternion(0.5, 0.5, -.5, 0.5)

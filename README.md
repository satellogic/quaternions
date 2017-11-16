Quaternions library
===================

[![Coverage Status](https://coveralls.io/repos/github/satellogic/quaternions/badge.svg)](https://coveralls.io/github/satellogic/quaternions)
[![Build Status](https://travis-ci.org/satellogic/quaternions.svg?branch=add_travis_ci)](https://travis-ci.org/satellogic/quaternions)

This is a library for dealing with quaternions in python in a unified way.

To install it:

```bash
$ pip install satellogic_quaternions
```

Examples of code:

```python
In [1]: from quaternions import Quaternion

In [2]: q1 = Quaternion(1, 2, 3, 4)

In [3]: q2 = Quaternion(2, 3, 5, 8)

In [4]: q1 + q2
Out[4]: Quaternion(3, 5, 8, 12)

In [5]: q1 * q2
Out[5]: Quaternion(-51, 3, 15, 15)

In [6]: q1 / q2
Out[6]: Quaternion(0.53921568627450989, 0.049019607843137247, -0.029411764705882353, 0.0098039215686274439)

In [7]: print(q1)
(1+2i+3j+4k)
```

~~Most of~~ all the quaternions we use are unitary (not like the example above)

```python
In [8]: q1.is_unitary()
Out[8]: False

In [9]: q1 / q1.norm()
Out[9]: Quaternion(0.18257418583505536, 0.36514837167011072, 0.54772255750516607, 0.73029674334022143)

In [10]: q1 /= q1.norm()

In [11]: q1.is_unitary()
Out[11]: True
```

Usually, quaternions are used for rotating vectors. This is done with `numpy`:
```python
In [12]: q1.matrix
Out[12]:
array([[-0.66666667,  0.66666667,  0.33333333],
       [ 0.13333333, -0.33333333,  0.93333333],
       [ 0.73333333,  0.66666667,  0.13333333]])

In [14]: q1.matrix.dot([2, 3, -4])
Out[14]: array([-0.66666667, -4.46666667,  2.93333333])

In [16]: q1.matrix.dot([1, 0, 0])
Out[16]: array([-0.66666667,  0.13333333,  0.73333333])
```

A unitary quaternion matrix is unitary. The inverse is the transpose and it is also the
matrix of the inverse quaternion. And the inverse quaternion of a unitary quaternion is
the conjugate:

```python
In [17]: q1.conjugate()
Out[17]: Quaternion(0.18257418583505536, -0.36514837167011072, -0.54772255750516607, -0.73029674334022143)

In [18]: q1
Out[18]: Quaternion(0.18257418583505536, 0.36514837167011072, 0.54772255750516607, 0.73029674334022143)

In [19]: q1 * q1.conjugate()
Out[19]: Quaternion(0.99999999999999978, 0.0, 0.0, 0.0)

In [20]: q1.conjugate().matrix
Out[20]:
array([[-0.66666667,  0.13333333,  0.73333333],
       [ 0.66666667, -0.33333333,  0.66666667],
       [ 0.33333333,  0.93333333,  0.13333333]])
```

License
=======

quaternions is Satellogic SA Copyright 2017. All our code is GPLv3 licensed.

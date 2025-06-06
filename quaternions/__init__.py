import warnings

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # for Python < 3.8

from quaternions.quaternion import Quaternion  # NOQA
from quaternions.general_quaternion import GeneralQuaternion, QuaternionError  # NOQA

warnings.simplefilter('once')

try:
    __version__ = version("quaternions")
except PackageNotFoundError:
    __version__ = "0.0.0"  # for when running tests or in development

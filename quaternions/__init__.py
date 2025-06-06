import warnings

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

from quaternions.quaternion import Quaternion  # NOQA
from quaternions.general_quaternion import GeneralQuaternion, QuaternionError  # NOQA

warnings.simplefilter('once')

__version__ = version("quaternions")

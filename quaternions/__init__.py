from quaternions.quaternion import Quaternion  # NOQA
from quaternions.general_quaternion import GeneralQuaternion, QuaternionError  # NOQA

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

import warnings
warnings.simplefilter('once')

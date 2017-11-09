import numpy as np


def covariance_matrix_from_angles(angles_list):
    """
    Computes covariance matrix from 3d vectors list
    """
    covariance_matrix = np.zeros((3, 3))
    for angles in angles_list:
        covariance_matrix += np.outer(angles.T, angles)
    covariance_matrix /= (angles_list.shape[0] + 1)
    return covariance_matrix


def sigma_lerner(covariance_matrix):
    """ Computes sigma lerner from covariance matrix"""
    values, _ = np.linalg.eig(covariance_matrix)
    return 1.87 * np.sqrt(max(values))


def rotation_matrix(quaternion):
    """
    Computes rotation matrix form quaternion.
    Function related to average_and_std_theorical
    """
    return np.array([[0, -quaternion.qk, quaternion.qj],
                     [quaternion.qk, 0, -quaternion.qi],
                     [-quaternion.qj, quaternion.qi, 0]])


def orthogonal_matrix(quaternion):
    """
    Computes orthogonal matrix form quaternion.
    Function related to average_and_std_theorical
    """
    return np.vstack((quaternion.qr * np.eye(3) + rotation_matrix(quaternion),
                      -np.array([quaternion.qi, quaternion.qj, quaternion.qk])))

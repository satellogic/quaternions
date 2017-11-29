import numpy as np


def covariance_matrix_from_angles(angles_list):
    """
    Computes covariance matrix from 3d vectors list
    :param angles_list: list of NumPy arrays
    """
    covariance_matrix = sum(np.outer(angles, angles) for angles in angles_list)
    covariance_matrix /= (len(angles_list) - 1)
    return covariance_matrix


def sigma_lerner(covariance_matrix):
    """
    Calculates the std. dev. as 1.87 of the root of the maximum
    eigenvalue of the covariance matrix, called sigma lerner.
    """
    values, _ = np.linalg.eig(covariance_matrix)
    return 1.87 * np.sqrt(max(values))


def cross_product_matrix(quaternion):
    """
    Auxiliary matrix for average_and_std_theorical calculations.
    equation (7) from Averaging Quaternions, by Markley, Cheng, Crassidis, Oschman
    """
    return np.array([[0, -quaternion.qk, quaternion.qj],
                     [quaternion.qk, 0, -quaternion.qi],
                     [-quaternion.qj, quaternion.qi, 0]])


def xi_matrix(quaternion):
    """
    Auxiliary matrix for average_and_std_theorical calculations.
    equation (15) from Averaging Quaternions, by Markley, Cheng, Crassidis, Oschman
    """
    return np.vstack((quaternion.qr * np.eye(3) + cross_product_matrix(quaternion),
                      -np.array([quaternion.qi, quaternion.qj, quaternion.qk])))

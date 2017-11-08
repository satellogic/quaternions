import numpy as np


def covariance_matrix_from_angles(angles_list):
    covariance_matrix = np.zeros((3, 3))
    for angles in angles_list:
        covariance_matrix += np.outer(angles.T, angles)
    covariance_matrix /= (angles_list.shape[0] + 1)
    return covariance_matrix


def sigma_lerner(covariance_matrix):
    """ Calcula sigma lerner en grados"""
    values, _ = np.linalg.eig(covariance_matrix)
    return 1.87 * np.sqrt(max(values))
"""
Lorentz boosts

"""
import numpy as np


def _velocity(gamma):
    return np.sqrt(1 - (1 / (gamma ** 2)))


def _magnitude(vector: np.ndarray):
    return np.sqrt(np.sum(vector ** 2))


def direction(vector: np.ndarray):
    """
    Return a unit vector parallel to the input - three dimensional

    """
    return vector / _magnitude(vector)


def _masses(energies, px, py, pz):
    """
    Relativistic mass

    """
    return np.sqrt((energies ** 2) - (px ** 2) - (py ** 2) - (pz ** 2))


def _multiply(array, matrix):
    """
    multiply each row of a matrix by each element of an array

    e.g. [1, 2] * [[1, 2, 3], [4, 5, 6]] = [[1, 2, 3], [8, 10, 12]]

    """
    return array.reshape((-1,) + (1,) * (matrix.ndim - 1)) * matrix


def gamma(mass, energy):
    """
    Easiest way to calculate gamma, i reckon

    Both in GeV

    """
    return energy / mass


def boost(target, gamma, direction: np.ndarray):
    """
    Boost a 4-vector (original) in a direction

    4-vector in GeV: (x, y, z, t)

    """
    assert len(target) == 4

    # Work out v from gamma
    v = _velocity(gamma)

    new_energy = gamma * (target[3] - v * np.dot(direction, target[:3]))
    new_p = (
        target[:3]
        + (gamma - 1) * np.dot(direction, target[:3]) * direction
        - gamma * target[3] * v * direction
    )

    return [*new_p, new_energy]

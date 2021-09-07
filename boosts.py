"""
Lorentz boosts

"""
import numpy as np


def _gamma(mass, energy):
    """
    Easiest way to calculate gamma, i reckon

    Both in GeV

    """
    return energy / mass


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


def boosts(*particles, target: np.ndarray):
    """
    Boost some particles into the rest frame of targets

    Each particle in particles should be an array of [[px1, px2,...], [py1, py2, ...], [py1, pz2, ...], [E1, E2, ...]]

    target should be in the same format

    :param *particles: parameter pack of array of particles to boost
    :param target: array of particles to boost into. Should all be the same particle, i.e. a D meson
    :param mass: mass of the target particles
    :returns: tuple of arrays of boosted particles

    """
    # Number of target particles - i.e. number of boosts we need to perform
    n = target.shape[1]

    # Work out masses of target particles
    masses = _masses(target[3], *target[0:3])

    # Find gamma for each target particle
    gammas = target[3] / masses
    betas = np.sqrt(1 - 1 / gammas ** 2)

    # Find direction for each target particle
    directions = (
        target[0:3] / np.sqrt(target[0] ** 2 + target[1] ** 2 + target[2] ** 2)
    ).T

    boosted_particles = []

    # For each particle:
    for particle in particles:
        particle_energies = particle[3]
        particle_3momenta = particle[0:3].T

        # Init empty array for solutions
        boosted_particle = np.zeros(target.shape)

        # Work out energies
        n_dot_p = (directions * particle_3momenta).sum(
            1
        )  # This is the dot product directions . 3-momenta
        energies = gammas * (particle_energies - betas * n_dot_p)

        # Work out momenta
        momenta = (
            particle_3momenta
            + _multiply((gammas - 1), _multiply(n_dot_p, directions))
            - _multiply(particle_energies * betas * gammas, directions)
        )
        boosted_particle[0:3] = momenta.T
        boosted_particle[3] = energies

        boosted_particles.append(boosted_particle)

    return tuple(boosted_particles)

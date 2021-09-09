import utils
import numpy as np
from typing import Tuple


def boosts(particles: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Boost multiple particles into the rest frame of different particle(s).

    Can either boost all particles into the rest frame of one target particle or many targets.

    `particles` should be a numpy array with shape (4, N), representing N 4-vectors.
    This should be [x, y, z, t], where each element is a length-N array.
    `target` can either be a shape (4, N) array representing many target particles whose frames we will boost into, or one
    target particle [x, y, z, t] whose frame too boost all particles into.

    :param particles: shape (4, N) array of particles to boost- (x, y, z, t)
    :param target: either one particle (x, y, z, t) whose frame we will boost all particles to, or a shape (4, N) array of target particles whose frames we will boost into.
    :return: shape (4, N) array of particles after boosting

    """
    # TODO: figure out whether target is one particle or several

    # Work out masses of target particles
    masses = utils._masses(target[3], *target[0:3])

    # Find gamma for each target particle
    gammas = target[3] / masses
    betas = np.sqrt(1 - 1 / gammas ** 2)

    # Find direction for each target particle
    directions = (
        target[0:3] / np.sqrt(target[0] ** 2 + target[1] ** 2 + target[2] ** 2)
    ).T

    particle_energies = particles[3]
    particle_3momenta = particles[0:3].T

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
        + utils._multiply((gammas - 1), utils._multiply(n_dot_p, directions))
        - utils._multiply(particle_energies * betas * gammas, directions)
    )
    boosted_particle[0:3] = momenta.T
    boosted_particle[3] = energies

    return boosted_particle


def boost_one_particle(particle, target) -> Tuple[float, float, float, float]:
    """
    Boost a single particle into the rest frame of a target particle

    :param particle: 4-vector (x, y, z, t) of particle to boost
    :param target: 4-vector (x, y, z, t) of particle whose rest frame we will boost to
    :return: 4-vector (x, y, x, t) of particle after boosting

    """
    pass

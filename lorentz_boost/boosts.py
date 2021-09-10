import utils
import numpy as np
from typing import Tuple


def _to_arrays(particle, count: int):
    """
    I want to deal with arrays of target particles

    This fcn converts a particle to an array of (the same) particles
    or if an array is passed in just returns it

    :param particle: either 1 particle or a numpy array of particles
    :param count: how many boosts we're dealing with
    :return: an array of particles to boost to
    :raises ValueError: if bad stuff is passed in

    """
    if (
        isinstance(particle, np.ndarray)
        and len(particle.shape) == 2
        and particle.shape[0] == 4
    ):
        # Multiple particles provided
        return particle

    elif len(particle) == 4:
        # One particle
        out = np.zeros((4, count))
        out[0] += particle[0]
        out[1] += particle[1]
        out[2] += particle[2]
        out[3] += particle[3]

        return out

    raise ValueError(
        f"target shape invalid: should either be a length-4 iterable [x, y, z, t] or a shape (4, N) array\nGot {type(particle)}"
    )


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
    n_particles = len(particles[0])

    # Create a 4xN numpy array of target particles to boost to
    targets = _to_arrays(target, n_particles)

    # Work out masses of target particles
    masses = utils._masses(targets[3], *targets[0:3])

    # Find gamma for each target particle
    gammas = targets[3] / masses
    betas = np.sqrt(1 - 1 / gammas ** 2)

    # Find direction for each target particle
    directions = (
        targets[0:3] / np.sqrt(targets[0] ** 2 + targets[1] ** 2 + targets[2] ** 2)
    ).T

    particle_energies = particles[3]
    particle_3momenta = particles[0:3].T

    # Init empty array for solutions
    boosted_particle = np.zeros(targets.shape)

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
    # Convert particle to 2d numpy array
    particle_array = np.array(
        [[particle[0]], [particle[1]], [particle[2]], [particle[3]]]
    )

    # Pass to multiple particle boost fcn
    boosted_particle = boosts(particle_array, target)

    # Convert to tuple
    return (
        boosted_particle[0][0],
        boosted_particle[1][0],
        boosted_particle[2][0],
        boosted_particle[3][0],
    )

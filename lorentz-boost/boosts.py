import utils
import numpy as np


def boosts(*particles: np.ndarray, target: np.ndarray) -> tuple:
    """
    Boost some particles into the rest frame of targets

    Each particle in particles should be an array of [[px1, px2,...], [py1, py2, ...], [py1, pz2, ...], [E1, E2, ...]]

    target should be in the same format

    :param *particles: parameter pack of array of particles to boost
    :param target: array of particles to boost into. Should all be the same particle, i.e. a D meson
    :param mass: mass of the target particles
    :returns: tuple of arrays of boosted particles

    """
    # Work out masses of target particles
    masses = utils._masses(target[3], *target[0:3])

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
            + utils._multiply((gammas - 1), utils._multiply(n_dot_p, directions))
            - utils._multiply(particle_energies * betas * gammas, directions)
        )
        boosted_particle[0:3] = momenta.T
        boosted_particle[3] = energies

        boosted_particles.append(boosted_particle)

    return tuple(boosted_particles)

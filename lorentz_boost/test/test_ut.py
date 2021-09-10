import numpy as np
from .. import utils
from .. import boosts
import pytest

D_MASS_GEV = 1.86484
K_MASS_GEV = 0.493677
PI_MASS_GEV = 0.139570


def test_array_to_array():
    N = 10

    # oOoOo random numbers in a unit test OoOooo
    target = np.random.random((4, N))

    assert np.allclose(target, boosts._to_arrays(target, N))


def test_particle_to_array():
    N = 3

    target = [1, 2, 3, 4]
    expected = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])

    assert np.allclose(expected, boosts._to_arrays(target, N))

    target = np.array([1, 2, 3, 4])
    assert np.allclose(expected, boosts._to_arrays(target, N))


def test_to_array_bad_shape():
    N = 3
    with pytest.raises(ValueError):
        boosts._to_arrays([1, 2, 3], N)

    with pytest.raises(ValueError):
        boosts._to_arrays([1, 2, 3, 4, 5], N)

    with pytest.raises(ValueError):
        boosts._to_arrays(np.random.random((3, N)), N)

    with pytest.raises(ValueError):
        boosts._to_arrays(np.random.random((5, N)), N)


def test_beta():
    assert utils._velocity(5.0 / 3.0) == 0.8


def test_magnitude():
    assert utils._magnitude(np.array([2.0, 7.0, 26.0])) == 27.0


def test_direction():
    assert np.allclose(
        utils.direction(np.array([2.0, 7.0, 26.0])),
        np.array([2.0 / 27.0, 7.0 / 27.0, 26.0 / 27.0]),
    )


def test_mass():
    assert utils._masses(27.0, 2.0, 7.0, 26.0) == 0.0


def test_masses():
    energies = np.array([27.0, 28.0, 10.0, 15.0])
    px = np.array([2.0, 2.0, 1.0, 2.0])
    py = np.array([7.0, 7.0, -2.0, 0.0])
    pz = np.array([26.0, 26.0, 3.0, 4.0])

    assert np.allclose(
        utils._masses(energies, px, py, pz),
        np.array([0.0, np.sqrt(55), np.sqrt(86), np.sqrt(205)]),
    )


def test_multiply():
    assert np.allclose(
        utils._multiply(np.array([1, 2]), np.array([[1, 2, 3], [4, 5, 6]])),
        np.array([[1, 2, 3], [8, 10, 12]]),
    )


def test_no_boost():
    # Stationary D
    d_momentum = np.array([0.0, 0.0, 0.0, D_MASS_GEV])

    gamma = utils.gamma(D_MASS_GEV, d_momentum[3])
    dirn = d_momentum[:3]

    # vector to boost
    target = np.array([1, 2, 3, 4])

    # Check that it's unaffected
    assert np.all(utils.boost(target, gamma, dirn) == target)


def test_x_boost():
    """
    Boost along the x axis

    """
    d_3momentum = [5.0, 0.0, 0.0]
    d_energy = np.sqrt(np.linalg.norm(d_3momentum) ** 2 + D_MASS_GEV ** 2)
    d_4momentum = np.array([*d_3momentum, d_energy])

    k_3momentum = [0.0, 3.0, -4.0]
    k_energy = np.sqrt(np.linalg.norm(k_3momentum) ** 2 + K_MASS_GEV ** 2)
    k_4momentum = np.array([*k_3momentum, k_energy])

    gamma = utils.gamma(D_MASS_GEV, d_energy)
    dirn = np.array([1.0, 0.0, 0.0])

    expected = np.array([-13.47280853, 3.0, -4.0, 14.37916154])

    assert np.allclose(expected, utils.boost(k_4momentum, gamma, dirn), atol=0.01)


def test_general_boost():
    """
    Boost along a not-nice direction

    """
    d_3momentum = np.array([1.0, 2.0, 3.0])
    d_energy = np.sqrt(np.linalg.norm(d_3momentum) ** 2 + D_MASS_GEV ** 2)
    d_4momentum = np.array([*d_3momentum, d_energy])

    k_3momentum = [1.0, 3.0, -4.0]
    k_energy = np.sqrt(np.linalg.norm(k_3momentum) ** 2 + K_MASS_GEV ** 2)
    k_4momentum = np.array([*k_3momentum, k_energy])

    gamma = utils.gamma(D_MASS_GEV, d_energy)
    dirn = utils.direction(d_3momentum)

    expected = np.array([-2.190947286, -3.381894571, -13.57284186, 14.16697618])

    assert np.allclose(expected, utils.boost(k_4momentum, gamma, dirn), atol=0.01)

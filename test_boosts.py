import numpy as np
import boosts
import helpers

D_MASS_GEV = 1.86484
K_MASS_GEV = 0.493677
PI_MASS_GEV = 0.139570


def test_beta():
    assert boosts._velocity(5.0 / 3.0) == 0.8


def test_magnitude():
    assert boosts._magnitude(np.array([2.0, 7.0, 26.0])) == 27.0


def test_direction():
    assert np.allclose(
        boosts.direction(np.array([2.0, 7.0, 26.0])),
        np.array([2.0 / 27.0, 7.0 / 27.0, 26.0 / 27.0]),
    )


def test_no_boost():
    # Stationary D
    d_momentum = np.array([0.0, 0.0, 0.0, D_MASS_GEV])

    gamma = helpers.gamma(D_MASS_GEV, d_momentum[3])
    dirn = d_momentum[:3]

    # vector to boost
    target = np.array([1, 2, 3, 4])

    # Check that it's unaffected
    assert np.all(boosts.boost(target, gamma, dirn) == target)


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

    gamma = helpers.gamma(D_MASS_GEV, d_energy)
    dirn = np.array([1.0, 0.0, 0.0])

    expected = np.array([-13.47280853, 3.0, -4.0, 14.37916154])

    assert np.allclose(expected, boosts.boost(k_4momentum, gamma, dirn), atol=0.01)


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

    gamma = helpers.gamma(D_MASS_GEV, d_energy)
    dirn = boosts.direction(d_3momentum)

    expected = np.array([-2.190947286, -3.381894571, -13.57284186, 14.16697618])

    assert np.allclose(expected, boosts.boost(k_4momentum, gamma, dirn), atol=0.01)


def test_masses():
    energies = np.array([10.0, 15.0])
    px = np.array([1.0, 2.0])
    py = np.array([-2.0, 0.0])
    pz = np.array([3.0, 4.0])

    masses = np.array([np.sqrt(86), np.sqrt(205)])

    assert np.allclose(boosts._masses(energies, px, py, pz), masses)


def test_multiply():
    assert np.allclose(
        boosts._multiply(np.array([1, 2]), np.array([[1, 2, 3], [4, 5, 6]])),
        np.array([[1, 2, 3], [8, 10, 12]]),
    )


def test_boosts():
    """
    Test numpy array of boosts

    """
    d = np.array(
        [
            [5.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [np.sqrt(25.0 + D_MASS_GEV ** 2), np.sqrt(14.0 + D_MASS_GEV ** 2)],
        ]
    )
    k = np.array(
        [
            [0.0, 1.0],
            [3.0, 3.0],
            [-4.0, -4.0],
            [np.sqrt(25.0 + K_MASS_GEV ** 2), np.sqrt(26.0 + K_MASS_GEV ** 2)],
        ]
    )

    expected = np.array(
        [
            [-13.47280853, -2.190947286],
            [3.0, -3.381894571],
            [-4.0, -13.57284186],
            [14.37916154, 14.16697618],
        ]
    )
    (calculated,) = boosts.boosts(k, target=d)

    assert np.allclose(calculated, expected, atol=0.01)


def test_boosts_multiple_particles():
    d = np.array(
        [
            [5.0, 1.0],
            [0.0, 2.0],
            [0.0, 3.0],
            [np.sqrt(25.0 + D_MASS_GEV ** 2), np.sqrt(14.0 + D_MASS_GEV ** 2)],
        ]
    )
    k = np.array(
        [
            [0.0, 1.0],
            [3.0, 3.0],
            [-4.0, -4.0],
            [np.sqrt(25.0 + K_MASS_GEV ** 2), np.sqrt(26.0 + K_MASS_GEV ** 2)],
        ]
    )

    pi = np.array(
        [
            [0.0, 1.0],
            [3.0, 3.0],
            [-4.0, -4.0],
            [np.sqrt(25.0 + PI_MASS_GEV ** 2), np.sqrt(26.0 + PI_MASS_GEV ** 2)],
        ]
    )

    expected_k = np.array(
        [
            [-13.47280853, -2.190947286],
            [3.0, -3.381894571],
            [-4.0, -13.57284186],
            [14.37916154, 14.16697618],
        ]
    )

    # Can only be bothered to test the pion energy
    expected_pi_energy = 14.11780251
    boosted_k, boosted_pi = boosts.boosts(k, pi, target=d)

    assert np.allclose(boosted_k, expected_k, atol=0.01)
    assert np.allclose(boosted_pi[-1, 1], expected_pi_energy, atol=0.01)

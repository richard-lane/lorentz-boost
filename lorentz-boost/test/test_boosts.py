import numpy as np
import utils
import boosts

D_MASS_GEV = 1.86484
K_MASS_GEV = 0.493677
PI_MASS_GEV = 0.139570


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

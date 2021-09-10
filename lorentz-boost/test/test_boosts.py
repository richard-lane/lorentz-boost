import numpy as np
import utils
import boosts

D_MASS_GEV = 1.86484
K_MASS_GEV = 0.493677
PI_MASS_GEV = 0.139570


def test_boost_one_particle():
    target = np.array([1.0, 2.0, 3.0, np.sqrt(14.0 + D_MASS_GEV ** 2)])
    source = (1.0, 3.0, -4.0, np.sqrt(26.0 + K_MASS_GEV ** 2))
    expected = (-2.190947286, -3.381894571, -13.57284186, 14.16697618)

    assert np.allclose(boosts.boost_one_particle(source, target), expected, atol=0.01)

    source = np.array([1.0, 3.0, -4.0, np.sqrt(26.0 + K_MASS_GEV ** 2)])
    assert np.allclose(boosts.boost_one_particle(source, target), expected, atol=0.01)


def test_boosts_one_target():
    target = np.array([1.0, 2.0, 3.0, np.sqrt(14.0 + D_MASS_GEV ** 2)])
    source = np.array(
        [
            [1.0, 1.0],
            [3.0, 2.0],
            [-4.0, 3.0],
            [np.sqrt(26.0 + K_MASS_GEV ** 2), np.sqrt(14.0 + D_MASS_GEV ** 2)],
        ]
    )

    expected = np.array(
        [
            [-2.190947286, 0.0],
            [-3.381894571, 0.0],
            [-13.57284186, 0.0],
            [14.16697618, D_MASS_GEV],
        ]
    )

    calculated = boosts.boosts(source, target=target)

    assert np.allclose(calculated, expected, atol=0.01)


def test_boosts():
    """
    Test boosting multiple particle to multiple targets

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
    calculated = boosts.boosts(k, target=d)

    assert np.allclose(calculated, expected, atol=0.01)

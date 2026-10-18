import numpy as np

from ctf.mappo.buffer import compute_gae


def test_gae_matches_manual():
    r = np.array([[1.0], [0.0], [2.0]])
    v = np.array([[0.5], [0.2], [0.1]])
    d = np.array([[0.0], [1.0], [0.0]])  # el episodio termina en t=1
    last = np.array([0.3])
    g, lam = 0.9, 0.8
    adv, ret = compute_gae(r, v, d, last, g, lam)
    d2 = 2.0 + g * 0.3 - 0.1
    d1 = 0.0 - 0.2                        # terminal: sin bootstrap
    d0 = 1.0 + g * 0.2 - 0.5
    assert np.allclose(adv.ravel(), [d0 + g * lam * d1, d1, d2])
    assert np.allclose(ret, adv + v)

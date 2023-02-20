import pytest
import math
from scipy.stats import binomtest
from rl01.bandits import Bandit, EpsilonDecay, LinearDecay, ExponentialDecay, InverseSqrtDecay


@pytest.fixture(scope="module")
def get_bandit():
    return Bandit(p_true=0.5)


def test_bandit_pull_values(get_bandit):
    assert get_bandit.pull() in (True, False)


def test_bandit_pull_dist(get_bandit):
    results = []
    for _ in range(0, 100):
        results.append(get_bandit.pull())
    tresults = binomtest(k=results.count(1), n=1000)
    assert tresults.pvalue <= 0.05


def test_epsilon_constant():
    e_decay = EpsilonDecay(eps_min=0.05, decay_rate=0.01)
    eps = 0.10
    for i in range(1,6):
        eps_n = e_decay(eps)
        assert eps_n == eps
        eps = eps_n
    assert eps == 0.10


def test_linear_decay():
    l_decay = LinearDecay(eps_min=0.05, decay_rate=0.01)
    eps = 0.10
    all_eps = [eps,]
    for i in range(1, 6):
        eps_n = l_decay(all_eps[-1])
        all_eps.append(eps_n)
        assert all_eps[-1] <= all_eps[-2]
    assert math.isclose(all_eps[-1], 0.05)


def test_exp_decay():
    exp_decay = ExponentialDecay(eps_min=0.05, decay_rate=0.01)
    eps = 0.10
    for i in range(1, 6):
        eps_n = exp_decay(eps)
        assert eps_n <= eps
        eps = eps_n
    assert math.isclose(eps, 0.05)


def test_inverse_sqr_decay():
    is_decay = InverseSqrtDecay(eps_min=0.05, decay_rate=0.01)
    eps = 0.10
    for i in range(1, 6):
        eps_n = is_decay(eps)
        assert eps_n <= eps
        eps = eps_n
    assert math.isclose(eps, 0.05)


@pytest.mark.skip
def test_adaptive_deacy():
    pass
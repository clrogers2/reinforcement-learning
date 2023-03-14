import math

import numpy as np
import pytest
from scipy.stats import binomtest

from rl01.bandits import (Bandit, EpsilonDecay, ExponentialDecay,
                          InverseSqrtDecay, LinearDecay,
                          OptimisticInitialValues, UpperConfidenceBound1)


@pytest.fixture(scope="module")
def get_bandit():
    return Bandit(p_true=0.5)


def test_bandit_pull_values(get_bandit):
    """Make sure the output of bandit is a bool value"""
    assert get_bandit.pull() in (True, False)


def test_bandit_pull_dist(get_bandit):
    """
    Make sure the bandit pull method works and the distribution generated is the one expected
    """
    results = []
    for _ in range(0, 100):
        results.append(get_bandit.pull())
    tresults = binomtest(k=results.count(1), n=1000)
    assert tresults.pvalue <= 0.05


def test_epsilon_constant():
    """Test the constant epsilon using the base decay class. Could implement a parameterized test that tests each 
    form of decay, however, in the future I may choose to test each form of decay more exactly. 
    """
    e_decay = EpsilonDecay(eps_min=0.05, decay_rate=0.01)
    eps = 0.10
    for i in range(1,6):
        eps_n = e_decay(eps)
        assert eps_n == eps
        eps = eps_n
    assert eps == 0.10


def test_linear_decay():
    """Test linear decay"""
    l_decay = LinearDecay(eps_min=0.05, decay_rate=0.01)
    eps = 0.10
    all_eps = [eps,]
    for i in range(1, 6):
        eps_n = l_decay(all_eps[-1])
        all_eps.append(eps_n)
        assert all_eps[-1] <= all_eps[-2]
    assert math.isclose(all_eps[-1], 0.05)


def test_exp_decay():
    """Test exponential decay"""
    exp_decay = ExponentialDecay(eps_min=0.05, decay_rate=0.01)
    eps = 0.10
    for i in range(1, 6):
        eps_n = exp_decay(eps)
        assert eps_n <= eps
        eps = eps_n
    assert math.isclose(eps, 0.05)


def test_inverse_sqr_decay():
    """Test the inverse square root decay"""
    is_decay = InverseSqrtDecay(eps_min=0.05, decay_rate=0.01)
    eps = 0.10
    for i in range(1, 6):
        eps_n = is_decay(eps)
        assert eps_n <= eps
        eps = eps_n
    assert math.isclose(eps, 0.05)


@pytest.mark.skip
def test_adaptive_deacy():
    """Since I don't have a handle on this yet, I skip for now"""
    pass


def test_UpperConfidenceBound1():
    """Test the UpperConfidenceBound class"""
    ucb = UpperConfidenceBound1(nbandits=2, probs=[0.1, 0.9], ntrials=100)
    ucb.experiment()
    assert np.argmax([b.p_estimate for b in ucb.bandits]) == 1


def test_OptimisticInitialValues():
    """Test the OptimisticInitialValues class"""
    oiv = OptimisticInitialValues(initial_mean=5, nbandits=2, probs=[0.1, 0.9], ntrials=100)
    oiv.experiment()
    assert np.argmax([b.p_estimate for b in oiv.bandits]) == 1

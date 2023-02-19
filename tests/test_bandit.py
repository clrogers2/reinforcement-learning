import pytest
from rl01.bandits import Bandit


@pytest.fixture(scope="module")
def get_bandit():
    pass


def test_bandit_pull(get_bandit):
    bandit = get_bandit
    pass


def test_epsilon_decay():
    pass


def test_linear_decay():
    pass


def test_exp_decay():
    pass


def test_inverse_sqr_decay():
    pass


def test_adaptive_deacy():
    pass
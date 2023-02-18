import pytest
from rl01.bandits import Bandit


@pytest.fixture(scope="module")
def get_bandit():
    pass


def test_bandit_pull(get_bandit):
    bandit = get_bandit
    pass

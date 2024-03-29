"""Bandits and Multi-arm Bandits"""
import math
from typing import Callable, List, Union

import matplotlib.pyplot as plt
import numpy as np


class EpsilonDecay(object):
    """
    A generic decay strategy for decreasing the value of epsilon over time
    """

    def __init__(self, name: str = "constant", eps_min: float = 0.01, decay_rate: float = 0.001):
        """
        A generic class for implementing a decay strategy for epsilon. Implementing a class with inheritance is 
        definitly overkill for code geared toward learning. But after being issued the challenge to implement decay, I 
        got thinking about how I would keep my code consistent and meet some of the demands of production use, so I 
        allowed myself an hour or two to try a more robust implementation.

        This base class implements a constant epsilon. To get actual decay, utilize a specific subclass. Once 
        instantiated, call the class with the current estimated epsilon value. Used in this way the class keeps track
        of the number of times the decay has occurred.

        const_decay = EpsilonDecay(eps_min=0.01, decay_rate=0.001)
        new_eps = const_decay(cur_eps)

        
        Alternatively you can keep track of 'n' manually and use the decay() method directly (used internally when 
        calling the class as shown above.)

        const_decay.n += 1
        new_eps = const_decay.decay(eps=cur_eps, n=const_decay.n)

        
        Args:
            name (str): Name of the decay strategy
            eps_min (float, optional): The lower limit to which epsilon might decay. Defaults to 0.001.
            decay_rate (float, optional): The rate of decay. Defaults to 0.01.
        """

        self.name = name
        # Need to verify epsilon min and max values so we set placeholder values then use the property to assign
        self._eps_min = 0.01
        self.eps_min = eps_min
        # need to verify decay rate so we set a placeholder value then use the property to verify and assign
        self._decay_rate = 0.001
        self.decay_rate = decay_rate
        self.n = 0

    def __call__(self, x: float = None):
        self.n += 1
        return self.decay(eps=x, n=self.n)

    @property
    def eps_min(self) -> float:
        """
        The minimum value to which epsilon might be reduced. Must be a value greater than or equal to 0 but less than 1.

        Returns:
            float: Minimum epsilon
        """
        return self._eps_min

    @eps_min.setter
    def eps_min(self, x):
        if 1 > x >= 0:
            self._eps_min = x
        else:
            raise ValueError("eps_min must be less than 1 and greater than or equal to 0")

    @eps_min.deleter
    def eps_min(self):
        self._eps_min = 0.01

    @property
    def decay_rate(self) -> float:
        """
        The rate at which epsilon would decay. Must be less than 1 and greater than 0.

        Returns:
            float: the decay rate for this decay strategy
        """
        return self._decay_rate

    @decay_rate.setter
    def decay_rate(self, x):
        if 1 > x > 0:
            self._decay_rate = x
        else:
            raise ValueError("Decay rate must be less than 1 and greater than 0")

    @decay_rate.deleter
    def decay_rate(self):
        self._decay_rate = 0.001

    def decay(self, eps: float, n: int | None) -> float:
        """
        Implement a constant epsilon

        Args:
            eps (float): The current epsilon value
            n (int): The current number of steps in the decay process. Not always used

        Returns:
            float: return the epsilon value
        """
        return eps


class LinearDecay(EpsilonDecay):
    """Decay epsilon linearly"""

    def __init__(self, *args, **kwargs):
        """
        Epsilon will decay by the product of the difference between the current epsilon and the minimum epsilon value 
        and the current step divided by the decay rate. In otherwords, how much distance we have left to go to reach the
        minimum, times the decay per step

        Args:
            eps_min (float, optional): The lower limit to which epsilon might decay. Defaults to 0.001.
            decay_rate (float, optional): The rate of decay a value between 0 and 1. Defaults to 0.01.
        """
        super().__init__(name='linear', *args, **kwargs)

    def decay(self, eps: float, n: int) -> float:
        return max(eps - (eps - self.eps_min) * (n * self.decay_rate), self.eps_min)


class ExponentialDecay(EpsilonDecay):
    """Decay epsilon exponentially"""

    def __init__(self, *args, **kwargs):
        """
        Epsilon will decay by the product of the difference between the current epsilon and the minimum epsilon value 
        and the negative decay rate for the given step squared.

        Args:
            eps_min (float, optional): The lower limit to which epsilon might decay. Defaults to 0.001.
            decay_rate (float, optional): The rate of decay a value between 0 and 1. Defaults to 0.01.
        """
        super().__init__(name='exponential', *args, **kwargs)

    def decay(self, eps: float, n: int) -> float:
        return max(self.eps_min + (eps - self.eps_min) * math.exp(-self.decay_rate * n), self.eps_min)


class InverseSqrtDecay(EpsilonDecay):
    """Decay epsilon using the inverse square root of n (the current number of steps)"""

    def __init__(self, *args, **kwargs):
        """
        Epsilon will decay by dividing by the square root of the number of steps plus 1.
       
        Args:
            eps_min (float, optional): The lower limit to which epsilon might decay. Defaults to 0.001.
            decay_rate (float, optional): The rate of decay a value between 0 and 1. Defaults to 0.01.
        """
        super().__init__(name='inverse_sqrt', *args, **kwargs)

    def decay(self, eps: float, n: int) -> float:
        return max(eps / math.sqrt(n + 1), self.eps_min)


class AdaptiveDecay(EpsilonDecay):
    """Change epsilon based on it's current performance"""

    def __init__(self, *args, **kwargs):
        """
        While I understand the theory behind adaptive decay, and know about "real-world" implementations like Adagrad, 
        AdaDelta, RMSProp and Adam; it is the decay strategy I understand the least. This implementation is my very 
        naive attempt to implement something that takes into account the performance of epsilon. However, I really lack 
        an understanding and intuition of the details of the implementation.
        """
        super().__init__(name='adaptive', *args, **kwargs)

    def decay(self, eps: float, perf: float, **kwargs):
        """
        Decay using some performance metric

        Args:
            eps (float): The current epsilon value
            perf (float): The current performance of epsilon

        Returns:
            float: the adjusted epsilon
        """
        # TODO return and fix implementation once I better understand adaptive decay
        return max(self.eps_min, eps / (1 + self.decay_rate * perf))


class Bandit(object):
    """
    A simulated Bandit
    """

    def __init__(self, p_true: float, dist: Callable = np.random.random):
        """
        Hold the true and estimated probability (p) for a simulated bandit

        Args:
            p_true (float): The true probability for this simulated bandit between 0 and 1.
        """

        self.dist = dist
        self.p_true = p_true
        self.p_estimate = 0.
        # Number of times this bandit was chosen. Needed to update the online probability (mean) using the previous mean
        # and the new value being added to the mean
        self.n_pulls = 0

    def pull(self) -> bool:
        """
        Generate a win or loss based on the true probability

        Returns:
            bool: The win or loss
        """
        return self.dist() < self.p_true

    def update(self, x: bool) -> None:
        """
        Update the probability estimate and the number of time chosen tracker.

        Args:
            x (bool): The win or loss generated by this pull of the simulated bandit. Automatically converted to an 
            integer of value 0 for False and 1 for True
        """
        self.n_pulls += 1
        self.p_estimate = ((self.n_pulls - 1) * self.p_estimate + x) / self.n_pulls


class MultiArmBandit(object):
    """
    Generic Multi-arm Bandit class to be inherited by the specific algorithmic implementation.
    """
    bandits: list = []

    def __init__(self, nbandits: int, probs: list | None, ntrials: int, dist: Callable = np.random.random, 
                 seed: int = 123):
        """
        Implement the basic framework needed to run a multi-arm bandit simulation. This class lacks the implementation
        of a specific algorithm and should be the super class of algorithm-specific implementations. I chose the 
        inheritance pattern to keep the code simple and readable while still avoiding most code duplication. 
        A seperate EnsembledMAB class could be implemented to allow the user to run simulations using multiple 
        algorithms without having to instantiate every algorithm individually.

        Args:
            nbandits (int): Total number of bandits (arms) to use in the simulation
            probs (list | None): The theoretical probability distribution of each bandit
            ntrials (int): Total number of simulated steps to run
            dist (Callable, optional): The probability distribution to use when determining rewards. 
                Defaults to np.random.random.
            seed (int, optional): Set the random seed for reproduceability. Defaults to 123.
        """

        self.seed = seed
        np.random.seed(self.seed)
        self.MAB = MultiArmBandit  # Shortcut the class
        self.n_bandits = nbandits
        self.i_bandits = list(range(self.n_bandits))
        self.dist = dist
        if not probs:
            probs = self.dist((1, self.n_bandits)).tolist()
        self.bandit_probs: List[float] = probs
        self.n_trials = ntrials
        self.bandits: List[Bandit] = list(map(Bandit, self.bandit_probs))
        self.rewards = np.zeros(self.n_trials)
        self.n_explored = 0
        self.n_exploited = 0
        self.num_optimal = 0
        self.optimal_bandit = np.argmax([b.p_true for b in self.bandits])

    def algorithm(self) -> int:
        """
        Populated by each individual algorithm class
        """
        self.n_explored += 1
        return np.random.choice(self.i_bandits)
        

    def experiment(self):
        """
        Contains the logic necessary to run the experiement and collect the data from the simulation.
        """
        for i in range(self.n_trials):
            #  Run the algorithm
            j = self.algorithm()
            # Since this is a simulation, we can keep track of how often the algorithm choose the optimal solution
            if j == self.optimal_bandit:
                # Record the number of times we select the Optimal bandit
                self.num_optimal += 1

            # Generate a win/loss for the currently selected bandit
            x = self.bandits[j].pull()
            # Update the log of wins and losses
            self.rewards[i] = x
            # Need to update the probability for the selected bandit
            self.bandits[j].update(x)

    def calc_metrics(self):
        """
        Since we are only simulating data, we can keep track of the metrics in order to showcase the performance of 
        each algorithm implementation
        """
        est = []
        p_true = []
        n_select = []
        # Iterate through the bandits and format data for output
        for i, b in enumerate(self.bandits):

            est.append(f"{i+1}: {b.p_estimate}")
            p_true.append(f"{i+1}: {b.p_true}")
            n_select.append(f"{i+1}: {b.n_pulls}")

        rewards = self.rewards.sum()
        win_rate = rewards / self.n_trials

        print(f"Mean Estimate: {est}")
        print(f"True Probability: {p_true}")
        print(f"Total Rewards Earned: {rewards}")
        print(f"Overall Win Rate: {win_rate}")
        print(f"Times Explored: {self.n_explored}")
        print(f"Times Exploited: {self.n_exploited}")
        print(f"Times Selected Optimal Bandit: {self.num_optimal}")
        print(f"Times Selected Each Bandit: {n_select}")

    def plot_results(self, log_scale: bool = False, y_max: int | None = None):
        """
        Plot the cumulative reward for the simulation against the maximum probable likelihood

        Args:
            log_scale (bool): Display the xaxis in log scale. Defaults to False
            y_max (int): Max height of the y-axis. Defaults to 1
        """

        cumulative_rewards = np.cumsum(self.rewards)
        win_rates = cumulative_rewards / (np.arange(self.n_trials) + 1)
        if y_max:
            plt.ylim([0, y_max])
        plt.plot(win_rates)
        plt.plot(np.ones(self.n_trials)*np.max(self.bandit_probs))
        if log_scale:
            plt.xscale('log')

        plt.show()


class EpsilonGreedy(MultiArmBandit):
    """MultiArmBandit with the Epsilon-greedy algorithm"""

    def __init__(self, *args, eps: float = 0.1, decay: Union[str, EpsilonDecay] = "constant", **kwargs):
        """
        A Multi-arm Bandit simulation with epsilon-greedy algorithm. Epsilon controls the probability that an explore
        action is randomly taken instead of a greedy action. In this implementation, explore could make a greedy choice.

        Args:
            eps (float, optional): Probabiliy of randomly choosing between all bandits. Defaults to 0.1.
            nbandits (int): Total number of bandits (arms) to use in the simulation
            probs (list | None): The theoretical probability distribution of each bandit
            ntrials (int): Total number of simulated steps to run
            dist (Callable, optional): The probability distribution to use when determining rewards. 
                Defaults to np.random.random.
            seed (int, optional): Set the random seed for reproduceability. Defaults to 123.
        """
        super().__init__(*args, **kwargs)
        self.eps = eps
        decay_strats = {"constant": EpsilonDecay, "linear": LinearDecay, "exponential": ExponentialDecay, 
                        "inverse_sqrt": InverseSqrtDecay}
        if decay.lower() in decay_strats:
            self.decay = decay_strats[decay.lower()]()
        else:
            print(f"'{decay}' Decay strategy not know. Defaulting to Constant")
            self.decay = decay_strats["constant"]()

    def algorithm(self) -> int:
        """
        Implements the Epsilon-greedy algorithm choice. Used by the experiment() method to make a choice.

        Returns:
            int: The position in the list of the bandit chosen for this pull
        """
        # Epsilon-greedy, explore if we generate a number lower than the value of epsilon
        if self.dist() < self.eps:
            self.n_explored += 1
            i = np.random.choice(self.i_bandits)  # pick a random bandit
        else:
            self.n_exploited += 1
            i = np.argmax([b.p_estimate for b in self.bandits])  # pick a bandit with the current MLE
        # Decay epsilon according to the choosen deay strategy
        self.eps = self.decay(self.eps)

        return i


class OptimisticInitialValues(MultiArmBandit):
    """Multi-arm Bandit with the Optimistic Initial values algorithm"""

    def __init__(self, nbandits: int, initial_mean: float, probs: list | None, ntrials: int,
                 dist: Callable = np.random.random, seed: int = 123):
        """
        Optimistic Initial Values is an algorithm that starts by over estimating the mean initially, and then selecting
        the bandit with the current highest mean. In this way, the algorithm lowers the probability estimate iteratively
        until the current bandit's probability estimate falls below that of another bandit. The algorithm switches to 
        the new current highest mean estimate bandit and continues the process until discovering the bandit with the 
        highest true probability.

        Args:
            nbandits (int): Total number of bandits (arms) to use in the simulation
            initial_mean (float): The higher the value the greater the exploration
            probs (list | None): The theoretical probability distribution of each bandit
            ntrials (int): Total number of simulated steps to run
            dist (Callable, optional): The probability distribution to use when determining rewards. 
                Defaults to np.random.random.
            seed (int, optional): Set the random seed for reproduceability. Defaults to 123.
        """
        super().__init__(nbandits=nbandits, probs=probs, ntrials=ntrials, dist=dist, seed=seed)
        self.initial_mean = initial_mean
        # We need to modify how the initial values of the Bandits to start with a high estimated mean
        # And set the starting number to 1 so that p_estimate doesn't get overwritten to zero on first iteration
        for bandit in self.bandits:
            bandit.p_estimate = initial_mean + self.dist()
            bandit.n_pulls = 1.
        self.current_bandit = np.argmax([b.p_estimate for b in self.bandits])
        # TODO something isn't right about the plot, I expect the plot of rewards to descend

    def algorithm(self) -> int:
        """
        Implement the I.O.V. loop by selecting the bandit with the highest probability estimate. I keep the explore/exploit
        tracking from the epsilon-greedy implementation so the user can see the impact of the initial mean (high values
        explore more, lower values exploit more)

        Returns:
            int: The index of the currently selected bandit
        """

        i_largest_mean = np.argmax([b.p_estimate for b in self.bandits])
        if i_largest_mean == self.current_bandit:
            self.n_exploited += 1
        else:
            self.n_explored += 1

        self.current_bandit = i_largest_mean
        return i_largest_mean


class UpperConfidenceBound1(MultiArmBandit):
    """Multi-arm Bandit with the Upper Confidence Bound 1 algorithm."""

    def __init__(self, *args, **kwargs):
        """
        The Upper Confidence Bound (UCB) algorithm is a reinforcement learning approach for the multi-armed bandit 
        problem that balances exploration and exploitation. It calculates the upper confidence bound for each arm using
        the formula: UCB_i = Q_i/n_i + c * sqrt(ln(N)/n_i), where Q_i/n_i is the average reward, N is the total number 
        of pulls, n_i is the number of pulls for arm i, c is a constant controlling exploration, and sqrt(ln(N)/n_i) 
        represents uncertainty. The algorithm selects the arm with the highest UCB value to maximize the 
        cumulative reward.

        Args:
            nbandits (int): Total number of bandits (arms) to use in the simulation
            probs (list | None): The theoretical probability distribution of each bandit
            ntrials (int): Total number of simulated steps to run
            dist (Callable, optional): The probability distribution to use when determining rewards. 
                Defaults to np.random.random.
            seed (int, optional): Set the random seed for reproduceability. Defaults to 123.
        """
        super().__init__(*args, **kwargs)
        # Initialize the bandits by pulling each bandit once
        self.total_trials = 0.
        for bandit in self.bandits:
            self.total_trials += 1
            reward = bandit.pull()
            bandit.update(reward)
        self.current_bandit = np.argmax([b.p_estimate for b in self.bandits])
    
    def ucb(self, mean_bandit: float, n_total: int, n_bandit: int):
        """
        Return the upper confidence bound for the given bandit

        Args:
            mean_bandit (float): The current mean estimate for the bandit
            n_total (int): The total number of trials across all bandits
            n_bandit (int): The number of trials for the given bandit

        Returns:
            float: The upper confidence bound for the given bandit
        """
        return mean_bandit + math.sqrt(2 * math.log(n_total) / n_bandit)
    
    def algorithm(self) -> int:
        """
        Implements the UCB1 algorithm. Selects the bandit with the highest upper confidence bound.

        Returns:
            int: The index of the currently selected bandit
        """
        i_largest_ucb = np.argmax([self.ucb(bandit.p_estimate, self.total_trials, bandit.n_pulls) for bandit in self.bandits])
        if i_largest_ucb == self.current_bandit:
            self.n_exploited += 1
        else:
            self.n_explored += 1
        
        self.current_bandit = i_largest_ucb
        return i_largest_ucb
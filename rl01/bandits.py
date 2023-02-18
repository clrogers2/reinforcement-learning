"""Bandits and Multi-arm Bandits"""
from typing import List
import numpy as np
import matplotlib.pyplot as plt


class Bandit(object):
    """
    A simulated Bandit
    """

    def __init__(self, p_true: float):
        """
        Hold the true and estimated probability (p) for a simulated bandit

        Args:
            p_true (float): The true probability for this simulated bandit between 0 and 1.
        """

        self.p_true = p_true
        self.p_estimate = 0
        # Number of times this bandit was chosen. Needed to update the online probability (mean) using the previous mean
        # and the new value being added to the mean
        self.n_trials = 0

    def pull(self) -> bool:
        """
        Generate a win or loss based on the true probability

        Returns:
            bool: The win or loss
        """
        return np.random.random() < self.p_true

    def update(self, x: bool) -> None:
        """
        Update the probability estimate and the number of time chosen tracker.

        Args:
            x (bool): The win or loss generated by this pull of the simulated bandit. Automatically converted to an 
            integer of value 0 for False and 1 for True
        """
        self.n_trials += 1
        self.p_estimate += (x / self.n_trials)


class MultiArmBandit(object):
    """
    Generic Multi-arm Bandit class to be inherited by the specific algorithmic implementation.
    """
    bandits: list = []

    def __init__(self, nbandits: int, probs: list | None, ntrials: int, seed: int = 123):
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
            seed (int, optional): Set the random seed for reproduceability. Defaults to 123.
        """

        self.MAB = MultiArmBandit  # Shortcut the class
        self.nbandits = nbandits
        self.ibandits = list(range(self.nbandits))
        if not probs:
            probs = np.random.random((1, self.nbandits)).tolist()
        self.bandit_probs: List[float] = probs
        self.ntrials = ntrials
        self.bandits = [bandit for bandit in map(Bandit, self.bandit_probs)]
        self.rewards = np.zeros(self.ntrials)
        self.nexplored = 0
        self.nexploited = 0
        self.num_optimal = 0
        self.seed = seed
        np.random.seed(self.seed)
        self.optimal_j = np.argmax([b.p for b in self.bandits])

    def algorithm(self) -> int:
        """
        Populated by each individual algorithm class
        """
        return np.random.choice(self.ibandits)

    def experiment(self):
        """
        Contains the logic necessary to run the experiement and collect the data from the simulation.
        """
        for i in range(self.ntrials):
            #  Run the algorithm
            j = self.algorithm()
            # Since this is a simulation, we can keep track of how often the algorithm choose the optimal solution
            if j == self.optimal_j:
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
        est = [f"{i}: {b.p_estimate}" for i,b in enumerate(self.bandits)]
        rewards = self.rewards.sum()
        win_rate = rewards / self.ntrials

        print(f"Mean Estimate: {est}")
        print(f"Total Rewards Earned: {rewards}")
        print(f"Overall Win Rate: {win_rate}")
        print(f"Times Explored: {self.nexplored}")
        print(f"Times Exploited: {self.nexploited}")
        print(f"Times Selected Optimal Bandit: {self.num_optimal}")

    def plot_results(self):
        """
        Plot the cumulative reward for the simulation against the maximum probable likelihood
        """
        cumulative_rewards = np.cumsum(self.rewards)
        win_rates = cumulative_rewards / (np.arange(self.ntrials) + 1)
        plt.plot(win_rates)
        plt.plot(np.ones(self.ntrials)*np.max(self.bandit_probs))
        plt.show()


class EpsilonGreedy(MultiArmBandit):
    """
    MultiArmBandit with the Epsilon-greedy algorithm"""
    def __init__(self, *args, eps: float = 0.1, **kwargs):
        """
        A Multi-arm Bandit simulation with epsilon-greedy algorithm. Epsilon controls the probability that an explore
        action is randomly taken instead of a greedy action. In this implementation, explore could make a greedy choice.

        Args:
            eps (float, optional): Probabiliy of randomly choosing between all bandits. Defaults to 0.1.
        """
        super().__init__(*args, **kwargs)
        self.eps = eps


    def algorithm(self) -> int:
        """
        Implements the Epsilon-greedy algorithm choice. Used by the experiment() method to make a choice.

        Returns:
            int: The position in the list of the bandit chosen for this pull
        """
        # Epsilon-greedy, explore if we generate a number lower than the value of epsilon
        if np.random.random() < self.eps:
            self.nexplored += 1
            i = np.random.choice(self.ibandits)  # pick a random bandit
        else:
            self.nexploited += 1
            i = np.argmax([b.p_estimate for b in self.bandits])  # pick a bandit with the current MLE

        return i

# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """
        # Store only the features relevant for Q-learning.
        # Using hashable types (tuple, frozenset) so states can be dict keys.
        self.pacman_pos = state.getPacmanPosition()
        self.ghost_positions = tuple(state.getGhostPositions())
        self.food = frozenset(state.getFood().asList())

    def __eq__(self, other):
        if not isinstance(other, GameStateFeatures):
            return False
        return (self.pacman_pos == other.pacman_pos and
                self.ghost_positions == other.ghost_positions and
                self.food == other.food)

    def __hash__(self):
        return hash((self.pacman_pos, self.ghost_positions, self.food))


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.2,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0

        # Q-table: maps (GameStateFeatures, action) -> float
        self.q_values = {}

        # Count table: maps (GameStateFeatures, action) -> int
        self.counts = {}

        # Remember the previous state and action to learn from each transition
        self.prev_state = None
        self.prev_action = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        return self.q_values.get((state, action), 0.0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        all_actions = [Directions.NORTH, Directions.SOUTH,
                       Directions.EAST, Directions.WEST]
        return max(self.getQValue(state, a) for a in all_actions)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        # Q(s,a) <- Q(s,a) + alpha * (reward + gamma * maxQ(s') - Q(s,a))
        current_q = self.getQValue(state, action)
        target = reward + self.gamma * self.maxQValue(nextState)
        self.q_values[(state, action)] = current_q + self.alpha * (target - current_q)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        key = (state, action)
        self.counts[key] = self.counts.get(key, 0) + 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        return self.counts.get((state, action), 0)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        if counts < self.maxAttempts:
            return 1.0
        else:
            return utility

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # logging to help you understand the inputs, feel free to remove
        print("Legal moves: ", legal)
        print("Pacman position: ", state.getPacmanPosition())
        print("Ghost positions:", state.getGhostPositions())
        print("Food locations: ")
        print(state.getFood())
        print("Score: ", state.getScore())

        current_features = GameStateFeatures(state)

        # If there was a previous step, learn from that transition
        if self.prev_state is not None and self.prev_action is not None:
            reward = self.computeReward(self.prev_state, state)
            self.learn(
                GameStateFeatures(self.prev_state),
                self.prev_action,
                reward,
                current_features
            )

        # Pick the action with the highest exploration score
        best_action = max(
            legal,
            key=lambda a: self.explorationFn(
                self.getQValue(current_features, a),
                self.getCount(current_features, a)
            )
        )

        self.updateCount(current_features, best_action)
        self.prev_state = state
        self.prev_action = best_action

        return best_action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Do a final Q-learning update using the terminal reward.
        # Terminal states have no future value so the target is reward only.
        if self.prev_state is not None and self.prev_action is not None:
            reward = self.computeReward(self.prev_state, state)
            prev_features = GameStateFeatures(self.prev_state)
            current_q = self.getQValue(prev_features, self.prev_action)
            self.q_values[(prev_features, self.prev_action)] = current_q + self.alpha * (reward - current_q)

        # Reset for next episode
        self.prev_state = None
        self.prev_action = None

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
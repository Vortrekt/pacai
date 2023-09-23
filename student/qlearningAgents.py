from pacai.agents.learning.reinforcement import ReinforcementAgent
from pacai.util import reflection, probability
import random

class QLearningAgent(ReinforcementAgent):
    """
    A Q-Learning agent.

    Some functions that may be useful:

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getAlpha`:
    Get the learning rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getDiscountRate`:
    Get the discount rate.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`:
    Get the exploration probability.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.getLegalActions`:
    Get the legal actions for a reinforcement agent.

    `pacai.util.probability.flipCoin`:
    Flip a coin (get a binary value) with some probability.

    `random.choice`:
    Pick randomly from a list.

    Additional methods to implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Compute the action to take in the current state.
    With probability `pacai.agents.learning.reinforcement.ReinforcementAgent.getEpsilon`,
    we should take a random action and take the best policy action otherwise.
    Note that if there are no legal actions, which is the case at the terminal state,
    you should choose None as the action.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    The parent class calls this to observe a state transition and reward.
    You should do your Q-Value update here.
    Note that you should never call this function, it will be called on your behalf.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

        # You can initialize Q-values here.
        self.qValues = {}

    def getQValue(self, state, action):
        """
        Get the Q-Value for a `pacai.core.gamestate.AbstractGameState`
        and `pacai.core.directions.Directions`.
        Should return 0.0 if the (state, action) pair has never been seen.
        """

        return self.qValues.get((state, action), 0.0)

    def getValue(self, state):
        """
        Return the value of the best action in a state.
        I.E., the value of the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of 0.0.

        This method pairs with `QLearningAgent.getPolicy`,
        which returns the actual best action.
        Whereas this method returns the value of the best action.
        """

        legalActions = self.getLegalActions(state)
        if not legalActions:  # Terminal state
            return 0.0

        return max(self.getQValue(state, action) for action in legalActions)

    def getPolicy(self, state):
        """
        Return the best action in a state.
        I.E., the action that solves: `max_action Q(state, action)`.
        Where the max is over legal actions.
        Note that if there are no legal actions, which is the case at the terminal state,
        you should return a value of None.

        This method pairs with `QLearningAgent.getValue`,
        which returns the value of the best action.
        Whereas this method returns the best action itself.
        """

        legalActions = self.getLegalActions(state)
        if not legalActions:  # Terminal state
            return None

        bestAction = None
        bestQValue = float('-inf')

        for action in legalActions:
            qValue = self.getQValue(state, action)
            if qValue > bestQValue:
                bestQValue = qValue
                bestAction = action

        return bestAction

    def update(self, state, action, nextState, reward):
        """
        The parent class calls this to observe a state transition and reward.
        Perform the Q-Value update here.
        """
        # Get the learning rate and discount rate.
        alpha = self.getAlpha()
        gamma = self.getDiscountRate()

        # Compute the sample update target.
        sample = reward + gamma * self.getValue(nextState)

        # Get the current Q-Value for the state-action pair.
        currentQValue = self.getQValue(state, action)

        # Update the Q-Value using the learning rate and sample update target.
        updatedQValue = (1 - alpha) * currentQValue + alpha * sample

        # Update the Q-Value in the dictionary.
        self.qValues[(state, action)] = updatedQValue

    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:  # Terminal state
            return None

        explorationProb = self.getEpsilon()
        if probability.flipCoin(explorationProb):  # Random action with epsilon probability
            return random.choice(legalActions)
        else:  # Best policy action otherwise
            return self.getPolicy(state)

class PacmanQAgent(QLearningAgent):
    """
    Exactly the same as `QLearningAgent`, but with different default parameters.
    """

    def __init__(self, index, epsilon = 0.05, gamma = 0.8, alpha = 0.2, numTraining = 0, **kwargs):
        kwargs['epsilon'] = epsilon
        kwargs['gamma'] = gamma
        kwargs['alpha'] = alpha
        kwargs['numTraining'] = numTraining

        super().__init__(index, **kwargs)

    def getAction(self, state):
        """
        Simply calls the super getAction method and then informs the parent of an action for Pacman.
        Do not change or remove this method.
        """

        action = super().getAction(state)
        self.doAction(state, action)

        return action

class ApproximateQAgent(PacmanQAgent):
    """
    An approximate Q-learning agent.

    You should only have to overwrite `QLearningAgent.getQValue`
    and `pacai.agents.learning.reinforcement.ReinforcementAgent.update`.
    All other `QLearningAgent` functions should work as is.

    Additional methods to implement:

    `QLearningAgent.getQValue`:
    Should return `Q(state, action) = w * featureVector`,
    where `*` is the dotProduct operator.

    `pacai.agents.learning.reinforcement.ReinforcementAgent.update`:
    Should update your weights based on transition.

    DESCRIPTION: <Write something here so we know what you did.>
    """

    def __init__(self, index,
            extractor = 'pacai.core.featureExtractors.IdentityExtractor', **kwargs):
        super().__init__(index, **kwargs)
        self.featExtractor = reflection.qualifiedImport(extractor)

        # You might want to initialize weights here.
        self.weights = {}  # Initialize weights with a Counter object

    def getQValue(self, state, action):
        """
        Should return `Q(state, action) = w * featureVector`,
        where `*` is the dotProduct operator.
        """
        features = self.featExtractor().getFeatures(state, action)
        # Dot product of weights and features
        q_value = sum(self.weights.get(feature, 0) * value for feature, value in features.items())
        return q_value

    def update(self, state, action, nextState, reward):
        """
        Should update your weights based on transition.
        """
        features = self.featExtractor().getFeatures(state, action)
        q_value = self.getQValue(state, action)
        difference = (reward + self.discountRate * self.getValue(nextState)) - q_value

        for feature, value in features.items():
            self.weights[feature] = self.weights.get(feature, 0) + self.alpha * difference * value

    def final(self, state):
        """
        Called at the end of each game.
        """

        # Call the super-class final method.
        super().final(state)

        # Did we finish training?
        if self.episodesSoFar == self.numTraining:
            # You might want to print your weights here for debugging.
            # *** Your Code Here ***
            print(self.weights)

from pacai.util import util
from random import choice
from pacai.core.directions import Directions
from pacai.agents.capture.capture import CaptureAgent
from logging import debug
from time import time


def createTeam(firstIndex, secondIndex, isRed, first='Hunter', second='Gatherer'):
    return [eval(first)(firstIndex), eval(second)(secondIndex)]

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that chooses score-maximizing actions.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def chooseAction(self, gameState):
        """
        Picks among the actions with the highest return from `ReflexCaptureAgent.evaluate`.
        """

        actions = gameState.getLegalActions(self.index)

        start = time()
        values = [self.evaluate(gameState, a) for a in actions]
        debug('evaluate() time for agent %d: %.4f' % (self.index, time() - start))

        maxValue = max(values)
        bestActions = [a for a, v in zip(actions, values) if v == maxValue]

        return choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """

        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()

        if (pos != util.nearestPoint(pos)):
            # Only half a grid position was covered.
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
        Computes a linear combination of features and feature weights.
        """

        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        stateEval = sum(features[feature] * weights[feature] for feature in features)

        return stateEval

    def getFeatures(self, gameState, action):
        """
        Returns a dict of features for the state.
        The keys match up with the return from `ReflexCaptureAgent.getWeights`.
        """

        successor = self.getSuccessor(gameState, action)

        return {
            'successorScore': self.getScore(successor)
        }

    def getWeights(self, gameState, action):
        """
        Returns a dict of weights for the state.
        The keys match up with the return from `ReflexCaptureAgent.getFeatures`.
        """

        return {
            'successorScore': 1.0
        }

class Gatherer (ReflexCaptureAgent):

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)

        # Compute distance to the nearest food.
        foodList = self.getFood(successor).asList()

        # This should always be True, but better safe than sorry.
        if (len(foodList) > 0):
            myPos = successor.getAgentState(self.index).getPosition()
            minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
            features['distanceToFood'] = minDistance

        return features

    def getWeights(self, gameState, action):
        return {
            'successorScore': 100,
            'distanceToFood': -1
        }

class Hunter(ReflexCaptureAgent):

    def getFeatures(self, gameState, action):
        features = {}
        successor = self.getSuccessor(gameState, action)
        my_state = successor.getAgentState(self.index)
        my_pos = my_state.getPosition()
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() is not None]

        if len(invaders) > 0:
            dists = [self.getMazeDistance(my_pos, a.getPosition()) for a in invaders]
            features['enemy'] = min(dists)

        if action == Directions.STOP:
            features['STOP'] = 1

        rev = Directions.REVERSE[gameState.getAgentState(self.index)._direction]
        if action == rev:
            features['REVERSE'] = 1

        if my_state._scaredTimer > 0:
            if 'enemy' in features and features['enemy'] <= 2:
                features['enemy'] = 2

        team_nums = self.getTeam(gameState)
        init_pos = gameState.getInitialAgentPosition(team_nums[0])
        features['spawn_dist'] = min(10, abs(my_pos[0] - init_pos[0])) + \
            int(abs(my_pos[0] - init_pos[0]) <= 4) * 2.5
        features['enemy_dist'] = 0
        features['defend'] = 1
        features['food'] = 0

        if my_state.isPacman:
            features['defend'] = -1

        if len(invaders) == 0 and successor.getScore() != 0:
            features['defend'] = -1
            food_list = self.getFood(successor).asList()
            if food_list:
                features['food'] = min([self.getMazeDistance(my_pos, food) for food in food_list])
            features['dots'] = len(food_list)
            features['spawn_dist'] = 0
            features['enemy_dist'] += 2
            features['enemy_dist'] *= features['enemy_dist']
        elif len(invaders) != 0:
            features['spawn_dist'] = 0

        return features

    def getWeights(self, gameState, action):
        return {
            'STOP': -500,
            'REVERSE': -300,
            'dots': -20,
            'food': -1,
            'spawn_dist': 2,
            'defend': 20,
            'enemy_dist': 40,
            'enemy': -1500
        }

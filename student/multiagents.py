import random

from pacai.agents.base import BaseAgent
from pacai.agents.search.multiagent import MultiAgentSearchAgent
from pacai.core.distance import manhattan

class ReflexAgent(BaseAgent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.
    You are welcome to change it in any way you see fit,
    so long as you don't touch the method headers.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        `ReflexAgent.getAction` chooses among the best options according to the evaluation function.

        Just like in the previous project, this method takes a
        `pacai.core.gamestate.AbstractGameState` and returns some value from
        `pacai.core.directions.Directions`.
        """

        # Collect legal moves.
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions.
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best.

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current `pacai.bin.pacman.PacmanGameState`
        and an action, and returns a number, where higher numbers are better.
        Make sure to understand the range of different values before you combine them
        in your evaluation function.
        """

        successorGameState = currentGameState.generatePacmanSuccessor(action)

        # Useful information you can extract.
        newPosition = successorGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.getScaredTimer() for ghostState in newGhostStates]
        ghostPositions = [ghostState.getPosition() for ghostState in newGhostStates]

        # *** Your Code Here ***
        newFood = successorGameState.getFood()
        foodList = newFood.asList()

        foodItems = [manhattan(newPosition, food) for food in foodList]

        minFoodItem = float("inf")
        if len(foodItems):
            minFoodItem = min(foodItems)

        scaredTime = min(newScaredTimes)

        ghosts = [manhattan(newPosition, ghost) for ghost in ghostPositions]

        closestGhost = min(ghosts)
        if closestGhost == 0:
            closestGhost = 1

        ghostScore = 0
        if scaredTime == 0 and closestGhost < 2:
            ghostScore = - 1000 / closestGhost
        elif scaredTime == 0 and closestGhost >= 2:
            ghostScore = - 2.0 / closestGhost
        else:
            ghostScore = 0.5 / closestGhost

        return successorGameState.getScore() + 1.0 / minFoodItem + ghostScore

class MinimaxAgent(MultiAgentSearchAgent):
    """
    A minimax agent.

    Here are some method calls that might be useful when implementing minimax.

    `pacai.core.gamestate.AbstractGameState.getNumAgents()`:
    Get the total number of agents in the game

    `pacai.core.gamestate.AbstractGameState.getLegalActions`:
    Returns a list of legal actions for an agent.
    Pacman is always at index 0, and ghosts are >= 1.

    `pacai.core.gamestate.AbstractGameState.generateSuccessor`:
    Get the successor game state after an agent takes an action.

    `pacai.core.directions.Directions.STOP`:
    The stop direction, which is always legal, but you may not want to include in your search.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        result = self.minimax(gameState, 0, 0)
        return result[1]

    def minimax(self, gameState, agent, depth):
        num_agents = gameState.getNumAgents()
        if self.getTreeDepth() == depth or len(gameState.getLegalActions()) == 0:
            return self.getEvaluationFunction()(gameState), ""

        if agent % num_agents == 0:
            return self.max_value(gameState, agent % num_agents, depth)
        else:
            return self.min_value(gameState, agent % num_agents, depth)

    def max_value(self, gameState, agent, depth):
        best_value = -float("inf")
        best_action = ""

        for action in gameState.getLegalActions(agent):
            next_state = gameState.generateSuccessor(agent, action)
            curr_value = self.minimax(next_state, agent + 1, depth + 1)[0]
            if curr_value > best_value:
                best_value = curr_value
                best_action = action

        return best_value, best_action

    def min_value(self, gameState, agent, depth):
        best_value = float("inf")
        best_action = ""

        for action in gameState.getLegalActions(agent):
            next_state = gameState.generateSuccessor(agent, action)
            curr_value = self.minimax(next_state, agent + 1, depth + 1)[0]
            if curr_value < best_value:
                best_value = curr_value
                best_action = action

        return best_value, best_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    A minimax agent with alpha-beta pruning.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the minimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        result = self.AlphaBeta(gameState, 0, 0, -float("inf"), float("inf"))
        return result[1]

    def AlphaBeta(self, gameState, agent, depth, alpha, beta):
        num_agents = gameState.getNumAgents()
        if self.getTreeDepth() == depth or len(gameState.getLegalActions()) == 0:
            return self.getEvaluationFunction()(gameState), ""

        if agent % num_agents == 0:
            return self.maximize(gameState, agent % num_agents, depth, alpha, beta)
        else:
            return self.minimize(gameState, agent % num_agents, depth, alpha, beta)

    def maximize(self, gameState, agent, depth, alpha, beta):
        best_value = -float("inf")
        best_action = ""

        for action in gameState.getLegalActions(agent):
            next_state = gameState.generateSuccessor(agent, action)
            curr_value = self.AlphaBeta(next_state, agent + 1, depth + 1, alpha, beta)[0]
            if curr_value > best_value:
                best_value = curr_value
                best_action = action
            if best_value > beta:
                return best_value, best_action
            alpha = max(alpha, best_value)

        return best_value, best_action

    def minimize(self, gameState, agent, depth, alpha, beta):
        best_value = float("inf")
        best_action = ""

        for action in gameState.getLegalActions(agent):
            next_state = gameState.generateSuccessor(agent, action)
            curr_value = self.AlphaBeta(next_state, agent + 1, depth + 1, alpha, beta)[0]
            if curr_value < best_value:
                best_value = curr_value
                best_action = action
            if best_value < alpha:
                return best_value, best_action
            beta = min(beta, best_value)

        return best_value, best_action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    An expectimax agent.

    All ghosts should be modeled as choosing uniformly at random from their legal moves.

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`:
    Returns the expectimax action from the current gameState using
    `pacai.agents.search.multiagent.MultiAgentSearchAgent.getTreeDepth`
    and `pacai.agents.search.multiagent.MultiAgentSearchAgent.getEvaluationFunction`.
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

    def getAction(self, gameState):
        result = self.expectimax(gameState, 0, 0)
        return result[1]

    def expectimax(self, gameState, agent, depth):
        num_agents = gameState.getNumAgents()
        if self.getTreeDepth() == depth or len(gameState.getLegalActions()) == 0:
            return self.getEvaluationFunction()(gameState), ""

        if agent % num_agents == 0:
            return self.max_value(gameState, agent % num_agents, depth)
        else:
            return self.exp_value(gameState, agent % num_agents, depth)

    def max_value(self, gameState, agent, depth):
        best_value = -float("inf")
        best_action = ""

        for action in gameState.getLegalActions(agent):
            next_state = gameState.generateSuccessor(agent, action)
            curr_value = self.expectimax(next_state, agent + 1, depth + 1)[0]
            if curr_value > best_value:
                best_value = curr_value
                best_action = action

        return best_value, best_action

    def exp_value(self, gameState, agent, depth):
        best_value = 0
        best_action = ""
        actions = 0

        for action in gameState.getLegalActions(agent):
            next_state = gameState.generateSuccessor(agent, action)
            curr_value = self.expectimax(next_state, agent + 1, depth + 1)[0]
            best_value += curr_value
            actions += 1

        return best_value / float(actions), best_action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable evaluation function.

    DESCRIPTION: <write something here so we know what you did>
    """
    score = currentGameState.getScore()
    position = currentGameState.getPacmanPosition()
    ghostStates = currentGameState.getGhostStates()
    food = currentGameState.getFood()
    foodList = food.asList()
    foodDist = [manhattan(position, food) for food in foodList]

    closestFood = float("inf")
    if len(foodDist):
        closestFood = min(foodDist)

    score += 0.5 / (closestFood + 1)

    for ghost in ghostStates:
        ghostDistance = manhattan(position, ghost.getPosition())
        if ghostDistance < 2 and ghost.getScaredTimer() == 0:
            score -= 1.0 / (ghostDistance + 1)
        elif ghostDistance < 2 and ghost.getScaredTimer() != 0:
            score += 0.5 / (ghostDistance + 1) + 0.5 * ghost.getScaredTimer()
        else:
            score += 0.5 / (closestFood + 1)

    return score

class ContestAgent(MultiAgentSearchAgent):
    """
    Your agent for the mini-contest.

    You can use any method you want and search to any depth you want.
    Just remember that the mini-contest is timed, so you have to trade off speed and computation.

    Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
    just make a beeline straight towards Pacman (or away if they're scared!)

    Method to Implement:

    `pacai.agents.base.BaseAgent.getAction`
    """

    def __init__(self, index, **kwargs):
        super().__init__(index, **kwargs)

from pacai.util.stack import Stack
from pacai.util.queue import Queue
from pacai.util.priorityQueue import PriorityQueue
"""
In this file, you will implement generic search algorithms which are called by Pacman agents.
"""

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first [p 85].

    Your search algorithm needs to return a list of actions that reaches the goal.
    Make sure to implement a graph search algorithm [Fig. 3.7].

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    ```
    print("Start: %s" % (str(problem.startingState())))
    print("Is the start a goal?: %s" % (problem.isGoal(problem.startingState())))
    print("Start's successors: %s" % (problem.successorStates(problem.startingState())))
    ```
    """

    # *** Your Code Here ***
    start = problem.startingState()
    if problem.isGoal(start):
        return []

    visited = []
    stack = Stack()

    stack.push((start, []))

    while not stack.isEmpty():
        successor, actions = stack.pop()
        if problem.isGoal(successor):
            return actions
        if successor not in visited:
            visited.append(successor)
            for new_successor in problem.successorStates(successor):
                stack.push((new_successor[0], actions + [new_successor[1]]))

    return []

    raise NotImplementedError()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first. [p 81]
    """

    # *** Your Code Here ***
    start = problem.startingState()
    if problem.isGoal(start):
        return []

    visited = []
    queue = Queue()

    queue.push((start, []))

    while not queue.isEmpty():
        successor, actions = queue.pop()
        if problem.isGoal(successor):
            return actions
        if successor not in visited:
            visited.append(successor)
            for new_successor in problem.successorStates(successor):
                queue.push((new_successor[0], actions + [new_successor[1]]))

    return []

    raise NotImplementedError()

def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """

    # *** Your Code Here ***
    start = problem.startingState()
    if problem.isGoal(start):
        return []

    visited = []
    priorityQueue = PriorityQueue()

    priorityQueue.push((start, [], 0), 0)

    while not priorityQueue.isEmpty():
        successor, actions, cost = priorityQueue.pop()
        if problem.isGoal(successor):
            return actions
        if successor not in visited:
            visited.append(successor)
            for new_successor in problem.successorStates(successor):
                new_action = actions + [new_successor[1]]
                new_cost = cost + new_successor[2]
                priorityQueue.push((new_successor[0], new_action, new_cost), new_cost)

    return []

    raise NotImplementedError()

def aStarSearch(problem, heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """

    # *** Your Code Here ***
    start = problem.startingState()
    if problem.isGoal(start):
        return []

    visited = []
    priorityQueue = PriorityQueue()
    priorityQueue.push((start, [], 0), 0)

    while not priorityQueue.isEmpty():
        successor, actions, cost = priorityQueue.pop()
        if problem.isGoal(successor):
            return actions
        if successor not in visited:
            visited.append(successor)
            for new_successor in problem.successorStates(successor):
                new_action = actions + [new_successor[1]]
                new_cost = cost + new_successor[2]
                priority = new_cost + heuristic(new_successor[0], problem)
                priorityQueue.push((new_successor[0], new_action, new_cost), priority)

    return []

    raise NotImplementedError()

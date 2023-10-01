# search.py
# ---------
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


"""
In search.py, we implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
import time
import timeit


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """Search the deepest nodes in the search tree first with a depth limit."""
    depth_limit = 20
    # states to be explored (LIFO). holds nodes in the form (state, action, depth)
    frontier = util.Stack()
    # previously explored states (for path checking), holds states
    exploredNodes = []
    # define start node
    startState = problem.getStartState()
    startNode = (startState, [], 0)

    frontier.push(startNode)
    start_time = timeit.default_timer()  # Record start time

    max_fringe_size = 0  # Track max fringe size
    expandedNodes = 0  # Track expanded nodes

    while not frontier.isEmpty():
        max_fringe_size = max(max_fringe_size, len(frontier.list))
        currentState, actions, currentDepth = frontier.pop()

        if currentDepth >= depth_limit:
            continue  # Skip nodes at or beyond the depth limit

        if currentState not in exploredNodes:
            exploredNodes.append(currentState)
            expandedNodes += 1  # Increment expanded nodes count

            if problem.isGoalState(currentState):
                elapsed_time = timeit.default_timer() - start_time
                return actions, len(actions), len(exploredNodes), max_fringe_size, elapsed_time, expandedNodes
            else:
                successors = problem.getSuccessors(currentState)
                #max_fringe_size = max(max_fringe_size, len(frontier.list))

                for succState, succAction, succCost in successors:
                    newAction = actions + [succAction]
                    newNode = (succState, newAction, currentDepth + 1)
                    frontier.push(newNode)

    elapsed_time = timeit.default_timer() - start_time
    return actions, len(actions), len(exploredNodes), max_fringe_size, elapsed_time, expandedNodes

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    #to be explored (FIFO)
    frontier = util.Queue()
    
    #previously expanded states (for cycle checking), holds states
    exploredNodes = []
    
    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)
    
    frontier.push(startNode)
    start_time = timeit.default_timer()  # Record start time

    max_fringe_size = 0  # Track max fringe size
    expandedNodes = 0  # Track expanded nodes

    while not frontier.isEmpty():
        #begin exploring first (earliest-pushed) node on frontier
        max_fringe_size = max(max_fringe_size, len(frontier.list))
        currentState, actions, currentCost = frontier.pop()
        
        if currentState not in exploredNodes:
            #put popped node state into explored list
            exploredNodes.append(currentState)
            expandedNodes += 1  # Increment expanded nodes count
            if problem.isGoalState(currentState):
                elapsed_time = timeit.default_timer() - start_time
                return actions, len(actions), len(exploredNodes), max_fringe_size, elapsed_time, expandedNodes
                
            else:
                #list of (successor, action, stepCost)
                successors = problem.getSuccessors(currentState)
                #max_fringe_size = max(max_fringe_size, len(frontier.list))
                for succState, succAction, succCost in successors:
                    newAction = actions + [succAction]
                    newCost = currentCost + succCost
                    newNode = (succState, newAction, newCost)

                    frontier.push(newNode)
    elapsed_time = timeit.default_timer() - start_time
    return actions, len(actions), len(exploredNodes), max_fringe_size, elapsed_time, expandedNodes
        
def uniformCostSearch(problem):
    """Search the node of least total cost first."""

    #to be explored (FIFO): holds (item, cost)
    frontier = util.PriorityQueue()

    #previously expanded states (for cycle checking), holds state:cost
    exploredNodes = {}
    
    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)
    
    frontier.push(startNode, 0)

    start_time = timeit.default_timer()  # Record start time

    max_fringe_size = 0  # Track max fringe size
    expandedNodes = 0  # Track expanded nodes
    while not frontier.isEmpty():
        #begin exploring first (lowest-cost) node on frontier
        max_fringe_size = max(max_fringe_size, len(frontier.heap))
        currentState, actions, currentCost = frontier.pop()
       
        if (currentState not in exploredNodes) or (currentCost < exploredNodes[currentState]):
            #put popped node's state into explored list
            exploredNodes[currentState] = currentCost
            expandedNodes += 1  # Increment expanded nodes count
            if problem.isGoalState(currentState):
                elapsed_time = timeit.default_timer() - start_time
                return actions, len(actions), len(exploredNodes), max_fringe_size, elapsed_time, expandedNodes

            else:
                #list of (successor, action, stepCost)
                successors = problem.getSuccessors(currentState)
                #max_fringe_size = max(max_fringe_size, len(frontier.heap))
                for succState, succAction, succCost in successors:
                    newAction = actions + [succAction]
                    newCost = currentCost + succCost
                    newNode = (succState, newAction, newCost)

                    frontier.update(newNode, newCost)
                    
    elapsed_time = timeit.default_timer() - start_time
    return actions, len(actions), len(exploredNodes), max_fringe_size, elapsed_time, expandedNodes


def aStarSearch(problem, heuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    #to be explored (FIFO): takes in item, cost+heuristic
    frontier = util.PriorityQueue()

    exploredNodes = [] #holds (state, cost)
   

    startState = problem.getStartState()
    startNode = (startState, [], 0) #(state, action, cost)

    frontier.push(startNode, 0)

    start_time = timeit.default_timer()  # Record start time

    max_fringe_size = 0 
    expandedNodes=0
    while not frontier.isEmpty():
        max_fringe_size = max(max_fringe_size, len(frontier.heap))
        #begin exploring first (lowest-combined (cost+heuristic) ) node on frontier
        currentState, actions, currentCost = frontier.pop()

        #put popped node into explored list

        currentNode = (currentState, currentCost)

        exploredNodes.append((currentState, currentCost))

        if problem.isGoalState(currentState):
            elapsed_time = timeit.default_timer() - start_time # Calculate elapsed time
            return actions, len(actions), len(exploredNodes), max_fringe_size, elapsed_time, expandedNodes

        else:
            #list of (successor, action, stepCost)
            successors = problem.getSuccessors(currentState)
            expandedNodes+=1

            #examine each successor
            for succState, succAction, succCost in successors:
                newAction = actions + [succAction]
                newCost = problem.getCostOfActions(newAction)
                newNode = (succState, newAction, newCost)

                #check if this successor has been explored
                already_explored = False
                for explored in exploredNodes:
                    #examine each explored node tuple
                    exploredState, exploredCost = explored

                    if (succState == exploredState) and (newCost >= exploredCost):
                        already_explored = True

                #if this successor not explored, put on frontier and explored list
                if not already_explored:
                    frontier.push(newNode, newCost + heuristic(succState, problem))
                    exploredNodes.append((succState, newCost))
    
                
    elapsed_time = timeit.default_timer() - start_time
    return actions, len(actions), len(exploredNodes), max_fringe_size, elapsed_time, expandedNodes
    


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
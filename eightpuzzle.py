# eightpuzzle.py
# --------------
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


import search
import random
import math

#heuristic 1
def misplaced_tiles(current_state, goal_state):
    
    # Initialize the count to zero
    count = 0
    
    # Check the dimensions of the provided matrices
    if len(current_state) != len(goal_state) or len(current_state[0]) != len(goal_state[0]):
        raise ValueError("Both states must have the same dimensions")
    
    # Compare each tile in the current state to the goal state
    for i in range(len(current_state)):
        for j in range(len(current_state[0])):
            # Exclude the blank tile (represented as ' ')
            if current_state[i][j] != goal_state[i][j] and current_state[i][j] != ' ':
                count += 1
                
    return count



def get_position(state, value):
        for i in range(len(state)):
            for j in range(len(state[i])):
                if state[i][j] == value:
                    return (i, j)
        return None  # Return None if the value isn't found

def euclidean_distance(current_state, goal_state):
    total_distance = 0
    for i in range(len(current_state)):
        for j in range(len(current_state[i])):
            if current_state[i][j] != ' ':  # Exclude the blank tile
                goal_position = get_position(goal_state, current_state[i][j])
                if goal_position:  # Check if the position is not None
                    goal_i, goal_j = goal_position
                    distance = math.sqrt((i - goal_i)**2 + (j - goal_j)**2)
                    total_distance += distance
    return total_distance


def manhattan_distance(current_state, goal_state):
    total_distance = 0
    for i in range(len(current_state)):
        for j in range(len(current_state[i])):
            if current_state[i][j] != ' ':  # Exclude the blank tile
                goal_position = get_position(goal_state, current_state[i][j])
                if goal_position:  # Check if the position is not None
                    goal_i, goal_j = goal_position
                    distance = abs(i - goal_i) + abs(j - goal_j)
                    total_distance += distance
    #print(str("current cost: ")+str(total_distance))
    return total_distance

def tiles_out_of_position(current_state, goal_state):
    total_out_of_position = 0
    for i in range(len(current_state)):
        for j in range(len(current_state[i])):
            if current_state[i][j] != ' ':  # Exclude the blank tile
                goal_position = get_position(goal_state, current_state[i][j])
                if goal_position:  # Check if the position is not None
                    goal_i, goal_j = goal_position
                    if i != goal_i:  # Tile is out of its correct row
                        total_out_of_position += 1
                    if j != goal_j:  # Tile is out of its correct column
                        total_out_of_position += 1
    return total_out_of_position

# Module Classes

class EightPuzzleState:
    """
    The Eight Puzzle is described in the course textbook on
    page 64.

    This class defines the mechanics of the puzzle itself.  The
    task of recasting this puzzle as a search problem is left to
    the EightPuzzleSearchProblem class.
    """

    def __init__( self, numbers ):
        """
          Constructs a new eight puzzle from an ordering of numbers.

        numbers: a list of integers from 0 to 8 representing an
          instance of the eight puzzle.  0 represents the blank
          space.  Thus, the list

            [1, 0, 2, 3, 4, 5, 6, 7, 8]

          represents the eight puzzle:
            -------------
            | 1 |   | 2 |
            -------------
            | 3 | 4 | 5 |
            -------------
            | 6 | 7 | 8 |
            ------------

        The configuration of the puzzle is stored in a 2-dimensional
        list (a list of lists) 'cells'.
        """
        self.cells = []
        numbers = numbers[:] # Make a copy so as not to cause side-effects.
        numbers.reverse()
        for row in range( 3 ):
            self.cells.append( [] )
            for col in range( 3 ):
                self.cells[row].append( numbers.pop() )
                if self.cells[row][col] == 0:
                    self.blankLocation = row, col

    def isGoal( self ):
        """
          Checks to see if the puzzle is in its goal state.

            -------------
            |   | 1 | 2 |
            -------------
            | 3 | 4 | 5 |
            -------------
            | 6 | 7 | 8 |
            -------------

        >>> EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8]).isGoal()
        True

        >>> EightPuzzleState([1, 0, 2, 3, 4, 5, 6, 7, 8]).isGoal()
        False
        """
        current = 0
        for row in range( 3 ):
            for col in range( 3 ):
                if current != self.cells[row][col]:
                    return False
                current += 1
        return True

    def legalMoves( self ):
        """
          Returns a list of legal moves from the current state.

        Moves consist of moving the blank space up, down, left or right.
        These are encoded as 'up', 'down', 'left' and 'right' respectively.

        >>> EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8]).legalMoves()
        ['down', 'right']
        """
        moves = []
        row, col = self.blankLocation
        if(row != 0):
            moves.append('up')
        if(row != 2):
            moves.append('down')
        if(col != 0):
            moves.append('left')
        if(col != 2):
            moves.append('right')
        return moves

    def result(self, move):
        """
          Returns a new eightPuzzle with the current state and blankLocation
        updated based on the provided move.

        The move should be a string drawn from a list returned by legalMoves.
        Illegal moves will raise an exception, which may be an array bounds
        exception.

        NOTE: This function *does not* change the current object.  Instead,
        it returns a new object.
        """
        row, col = self.blankLocation
        if(move == 'up'):
            newrow = row - 1
            newcol = col
        elif(move == 'down'):
            newrow = row + 1
            newcol = col
        elif(move == 'left'):
            newrow = row
            newcol = col - 1
        elif(move == 'right'):
            newrow = row
            newcol = col + 1
        else:
            raise "Illegal Move"

        # Create a copy of the current eightPuzzle
        newPuzzle = EightPuzzleState([0, 0, 0, 0, 0, 0, 0, 0, 0])
        newPuzzle.cells = [values[:] for values in self.cells]
        # And update it to reflect the move
        newPuzzle.cells[row][col] = self.cells[newrow][newcol]
        newPuzzle.cells[newrow][newcol] = self.cells[row][col]
        newPuzzle.blankLocation = newrow, newcol

        return newPuzzle

    # Utilities for comparison and display
    def __eq__(self, other):
        """
            Overloads '==' such that two eightPuzzles with the same configuration
          are equal.

          >>> EightPuzzleState([0, 1, 2, 3, 4, 5, 6, 7, 8]) == \
              EightPuzzleState([1, 0, 2, 3, 4, 5, 6, 7, 8]).result('left')
          True
        """
        for row in range( 3 ):
            if self.cells[row] != other.cells[row]:
                return False
        return True

    def __hash__(self):
        return hash(str(self.cells))

    def __getAsciiString(self):
        """
          Returns a display string for the maze
        """
        lines = []
        horizontalLine = ('-' * (13))
        lines.append(horizontalLine)
        for row in self.cells:
            rowLine = '|'
            for col in row:
                if col == 0:
                    col = ' '
                rowLine = rowLine + ' ' + col.__str__() + ' |'
            lines.append(rowLine)
            lines.append(horizontalLine)
        return '\n'.join(lines)

    def __str__(self):
        return self.__getAsciiString()

# TODO: Implement The methods in this class

class EightPuzzleSearchProblem(search.SearchProblem):
    """
      Implementation of a SearchProblem for the  Eight Puzzle domain

      Each state is represented by an instance of an eightPuzzle.
    """
    def __init__(self,puzzle):
        "Creates a new EightPuzzleSearchProblem which stores search information."
        self.puzzle = puzzle
    
    def getStartState(self):
        return self.puzzle

    def isGoalState(self,state):
        return state.isGoal()

    def getSuccessors(self,state):
        """
          Returns list of (successor, action, stepCost) pairs where
          each succesor is either left, right, up, or down
          from the original state and the cost is 1.0 for each
        """
        succ = []
        for a in state.legalMoves():
            succ.append((state.result(a), a, 1))
        return succ

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)
    
    def getHeuristicValue(self, state, heuristic_type):
        # Convert the EightPuzzleState object to a 2D list format
        current_state = [row for row in state.cells]
        goal_state = [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8]
        ]
        
        if heuristic_type == "misplaced":
            return misplaced_tiles(current_state, goal_state)
        elif heuristic_type == "euclidean":
            return euclidean_distance(current_state, goal_state)
        elif heuristic_type == "manhattan":
            return manhattan_distance(current_state, goal_state)
        elif heuristic_type == "out_of_position":
            return tiles_out_of_position(current_state, goal_state)
        else:
            raise ValueError("Invalid heuristic type")


    

EIGHT_PUZZLE_DATA = [[1, 0, 2, 3, 4, 5, 6, 7, 8],
                     [1, 7, 8, 2, 3, 4, 5, 6, 0],
                     [4, 3, 2, 7, 0, 5, 1, 6, 8],
                     [5, 1, 3, 4, 0, 2, 6, 7, 8],
                     [1, 2, 5, 7, 6, 8, 0, 4, 3],
                     [0, 3, 1, 6, 8, 2, 7, 5, 4]]

def loadEightPuzzle(puzzleNumber):
    """
      puzzleNumber: The number of the eight puzzle to load.

      Returns an eight puzzle object generated from one of the
      provided puzzles in EIGHT_PUZZLE_DATA.

      puzzleNumber can range from 0 to 5.

      >>> print loadEightPuzzle(0)
      -------------
      | 1 |   | 2 |
      -------------
      | 3 | 4 | 5 |
      -------------
      | 6 | 7 | 8 |
      -------------
    """
    return EightPuzzleState(EIGHT_PUZZLE_DATA[puzzleNumber])

def createRandomEightPuzzle(moves=100):
    """
      moves: number of random moves to apply

      Creates a random eight puzzle by applying
      a series of 'moves' random moves to a solved
      puzzle.
    """
    puzzle = EightPuzzleState([0,1,2,3,4,5,6,7,8])
    for i in range(moves):
        # Execute a random legal move
        puzzle = puzzle.result(random.sample(puzzle.legalMoves(), 1)[0])
    return puzzle
# choosing the heuristic
def chooseHeuristic():
    print("Choose the heuristic you want to use:")
    print("1. Misplaced tiles")
    print("2. Euclidean distance")
    print("3. Manhattan distance")
    print("4. Tiles out of position")
    print("5. All")
    choice = int(input("Enter your choice: "))
    if choice == 1:
        return "misplaced"
    elif choice == 2:
        return "euclidean"
    elif choice == 3:
        return "manhattan"
    elif choice == 4:
        return "out_of_position"
    elif choice == 5:
        return "all"
    else:
        print("Invalid choice. Please try again.")
        return chooseHeuristic()

def run_all_heuristics(puzzle):
    heuristics = ["misplaced", "euclidean", "manhattan", "out_of_position"]
    
    for heuristic_type in heuristics:
        print(f"Using {heuristic_type} heuristic:")
        problem = EightPuzzleSearchProblem(puzzle)
        path, depth, explored_nodes, max_fringe_size, elapsed_time, expanded_nodes = search.aStarSearch(problem, lambda state, problem=None: problem.getHeuristicValue(state, heuristic_type=heuristic_type))
        
        if path:
            print('A* found a path of %d moves: %s' % (len(path), str(path)))
            print('Depth:', depth)
            print('Explored Nodes:', explored_nodes)
            print('Expanded Nodes:', expanded_nodes)
            print('Max Fringe Size:', max_fringe_size)
            print('Elapsed Time (ms):', elapsed_time)
            curr = puzzle
            i = 1
            for a in path:
                curr = curr.result(a)
                print(f'After {i} move{"s" if i > 1 else ""}: {a}')
                print(curr)
                i += 1
        else:
            print("No solution found using this heuristic.")
        print("-------------------------------------------------")
# RUNNING ALL HEURISTICS: 
# def using_all_heuristics():
    
if __name__ == '__main__':
    heuristic_type = chooseHeuristic()
    if(heuristic_type == "all"):
        puzzle = createRandomEightPuzzle(25)
            
        print("Initial Puzzle:")
        print(puzzle)
        run_all_heuristics(puzzle)
    else:
        puzzle = createRandomEightPuzzle(25)
        print('A random puzzle:')
        print(puzzle)
        problem = EightPuzzleSearchProblem(puzzle)
    
        path, depth, explored_nodes, max_fringe_size, elapsed_time, expanded_nodes  = search.aStarSearch(problem, lambda state, problem=None: problem.getHeuristicValue(state, heuristic_type=heuristic_type))


        """
        path = search.breadthFirstSearch(problem)
        """   
        print('A* found a path of %d moves: %s' % (len(path), str(path)))
        print('Depth:', depth)
        print('Explored Nodes:', explored_nodes)
        print('Expanded Nodes:', expanded_nodes)
        print('Max Fringe Size:', max_fringe_size)
        print('Elapsed Time (ms):', elapsed_time)
        curr = puzzle
        i = 1
        for a in path:
            curr = curr.result(a)
            print('After %d move%s: %s' % (i, ("", "s")[i>1], a))
            print(curr)

            input("Press return for the next state...")   # wait for key stroke
            i += 1


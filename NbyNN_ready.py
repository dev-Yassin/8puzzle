import csv
import search
import pandas as pd
import csv
import random
import math

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

class NByNPuzzleSearchProblem(search.SearchProblem):
    def __init__(self, puzzle, heuristic_type):
        self.puzzle = puzzle
        self.heuristic_type = heuristic_type

    def getStartState(self):
        return self.puzzle

    def isGoalState(self, state):
        return state.is_goal()

    def getSuccessors(self, state):
        successors = []
        for action in state.legal_moves():
            successor = state.result(action)
            cost = 1  # The cost of each action is 1
            successors.append((successor, action, cost))
        return successors

    def getCostOfActions(self, actions):
        return len(actions)
    
def getHeuristicValue(self, current_state, heuristic_type):
    n = len(current_state)  # Get the dimensions of the current state
    goal_state = [[j + n * i for j in range(n)] for i in range(n)]  # Create a goal state based on dimensions

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
         

class NByNPuzzleState:
    def __init__(self, numbers):
        self.cells = []
        self.n = int(len(numbers) ** 0.5)
        numbers = numbers[:]  # Make a copy so as not to cause side-effects.
        numbers.reverse()
        
        for _ in range(self.n):
            self.cells.append([])
            for _ in range(self.n):
                self.cells[-1].append(numbers.pop())
                if self.cells[-1][-1] == 0:
                    self.blankLocation = (len(self.cells) - 1, len(self.cells[-1]) - 1)

    def get_size(self):
        return self.n

    def is_goal(self):
        current = 0
        for row in self.cells:
            for cell in row:
                if current != cell:
                    return False
                current += 1
        return True

    def legal_moves(self):
        moves = []
        row, col = self.blankLocation
        if row != 0:
            moves.append('up')
        if row != self.n - 1:
            moves.append('down')
        if col != 0:
            moves.append('left')
        if col != self.n - 1:
            moves.append('right')
        return moves

    def result(self, move):
        row, col = self.blankLocation
        if move == 'up':
            new_row = row - 1
            new_col = col
        elif move == 'down':
            new_row = row + 1
            new_col = col
        elif move == 'left':
            new_row = row
            new_col = col - 1
        elif move == 'right':
            new_row = row
            new_col = col + 1
        else:
            raise ValueError("Illegal Move")

        new_puzzle = NByNPuzzleState([0] * (self.n * self.n))
        new_puzzle.cells = [values[:] for values in self.cells]
        new_puzzle.cells[row][col] = self.cells[new_row][new_col]
        new_puzzle.cells[new_row][new_col] = self.cells[row][col]
        new_puzzle.blankLocation = new_row, new_col

        return new_puzzle

    def __eq__(self, other):
        for i in range(self.n):
            if self.cells[i] != other.cells[i]:
                return False
        return True

    def __hash__(self):
        return hash(str(self.cells))

    def __str__(self):
        lines = []
        horizontal_line = '-' * (4 * self.n + 1)
        lines.append(horizontal_line)
        for row in self.cells:
            row_line = '|'
            for col in row:
                if col == 0:
                    col = ' '
                row_line = row_line + f' {col} |'
            lines.append(row_line)
            lines.append(horizontal_line)
        return '\n'.join(lines)

def generate_and_write_states_to_csv(filename, num_states, n):
    initial_states = []
    for _ in range(num_states):
        solved_puzzle = list(range(n * n))
        random.shuffle(solved_puzzle)
        initial_states.append(solved_puzzle)

    with open(filename, mode='w', newline='') as scenarios_file:
        writer = csv.writer(scenarios_file)
        for state in initial_states:
            writer.writerow(state)

if __name__ == '__main__':
    # Define the number of initial states to generate, the puzzle size (N), and the filename
    NUM_INITIAL_STATES = 5
    N = 4
    FILENAME = 'scenarios.csv'

    # Call the method to generate and write initial states to CSV
    generate_and_write_states_to_csv(FILENAME, NUM_INITIAL_STATES, N)

def load_puzzles_from_csv(filename):
    puzzle_states = []

    try:
        with open(filename, mode='r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                # Convert the row (list of integers) into an NByNPuzzleState object
                nbyn_puzzle_state = NByNPuzzleState(list(map(int, row)))
                puzzle_states.append(nbyn_puzzle_state)
    except FileNotFoundError:
        print(f"CSV file '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")

    return puzzle_states

if __name__ == '__main__':
    # Specify the CSV filename where the initial states are stored
    csv_filename = 'scenarios.csv'
    heuristics = ["misplaced", "euclidean", "manhattan", "out_of_position"]
    results_df = pd.DataFrame(columns=["Heuristic", "Puzzle", "Depth", "Explored Nodes", "Max Fringe Size", "Elapsed Time", "Expanded Nodes"])
    
    # Load the NByNPuzzleState objects from the CSV file
    puzzle_states = load_puzzles_from_csv(csv_filename)

    # Print the loaded NByNPuzzleState objects
    for i, puzzle in enumerate(puzzle_states, start=1):
        print(f"Puzzle {i}:")
        print(puzzle)

    # Define the puzzle size (N) based on the loaded puzzle
    N = puzzle_states[0].get_size()  # Assuming all puzzles have the same size
    
    # Iterate over each puzzle and run A* search with different heuristics
    for puzzle in puzzle_states:
        for heuristic_type in heuristics:
            print(f"Running A* search for puzzle:")
            print(puzzle)
            print(f"Using {heuristic_type} heuristic:")
            problem = NByNPuzzleSearchProblem(puzzle, heuristic_type)
            path, depth, explored_nodes, max_fringe_size, elapsed_time, expanded_nodes = search.aStarSearch(problem, lambda state, problem=None: problem.getHeuristicValue(state, heuristic_type=heuristic_type))
        
        
            if path:
                print('A* found a path of %d moves.' % (len(path)))
                print('Depth:', depth)
                print('Explored Nodes:', explored_nodes)
                print('Max Fringe Size:', max_fringe_size)
                print('Elapsed Time (ms):', elapsed_time)
                print('Expanded Nodes:', expanded_nodes)
            else:
                print("No solution found using this heuristic.")
            print("-------------------------------------------------")

            # Add results to the DataFrame
            results_df = pd.concat([results_df, pd.DataFrame({
                "Heuristic": [heuristic_type],
                "Puzzle": [str(puzzle)],
                "Depth": [depth],
                "Explored Nodes": [explored_nodes],
                "Max Fringe Size": [max_fringe_size],
                "Elapsed Time": [elapsed_time],
                "Expanded Nodes": [expanded_nodes]
            })], ignore_index=True)

    # Save the results to a CSV file
    results_df.to_csv("_results.csv", index=False)

    # Create a dictionary to store average metrics for each heuristic
    avg_metrics = {heuristic: {} for heuristic in heuristics}

    # Calculate average metrics for each heuristic
    for heuristic_type in heuristics:
        avg_metrics[heuristic_type]["Average Depth"] = results_df[results_df["Heuristic"] == heuristic_type]["Depth"].mean()
        avg_metrics[heuristic_type]["Average Explored Nodes"] = results_df[results_df["Heuristic"] == heuristic_type]["Explored Nodes"].mean()
        avg_metrics[heuristic_type]["Average Max Fringe Size"] = results_df[results_df["Heuristic"] == heuristic_type]["Max Fringe Size"].mean()
        avg_metrics[heuristic_type]["Average Expanded Nodes"] = results_df[results_df["Heuristic"] == heuristic_type]["Expanded Nodes"].mean()
        avg_metrics[heuristic_type]["Average Time"] = results_df[results_df["Heuristic"] == heuristic_type]["Elapsed Time"].mean()

    # Create a DataFrame with columns for each heuristic's average metrics
    avg_metrics_df = pd.DataFrame.from_dict(avg_metrics, orient='index')

    # Reset the index to add heuristic as a column
    avg_metrics_df.reset_index(level=0, inplace=True)
    avg_metrics_df.rename(columns={'index': 'Heuristic'}, inplace=True)

    # Save the average metrics to a CSV file
    avg_metrics_df.to_csv("average_metrics.csv", index=False)

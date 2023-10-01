import csv
import random
import math
import pandas as pd
import timeit
from enum import Enum

# Define an Enum for tile movement directions
class MoveDirection(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

class PuzzleState:
    def __init__(self, size):
        self.size = size
        self.tiles = self.generate_random_puzzle()

    def generate_random_puzzle(self):
        tiles = list(range(self.size**2))
        random.shuffle(tiles)
        return tiles

    def get_tile(self, row, col):
        return self.tiles[row * self.size + col]

    def set_tile(self, row, col, value):
        self.tiles[row * self.size + col] = value

    def find_blank(self):
        for row in range(self.size):
            for col in range(self.size):
                if self.get_tile(row, col) == 0:
                    return row, col

    def is_goal_state(self):
        for i in range(self.size ** 2 - 1):
            if self.tiles[i] != i + 1:
                return False
        return True

    def is_valid_move(self, move):
        blank_row, blank_col = self.find_blank()

        if move == MoveDirection.UP:
            return blank_row > 0
        elif move == MoveDirection.DOWN:
            return blank_row < self.size - 1
        elif move == MoveDirection.LEFT:
            return blank_col > 0
        elif move == MoveDirection.RIGHT:
            return blank_col < self.size - 1

    def apply_move(self, move):
        if not self.is_valid_move(move):
            raise ValueError("Invalid move")

        blank_row, blank_col = self.find_blank()

        if move == MoveDirection.UP:
            target_row = blank_row - 1
            target_col = blank_col
        elif move == MoveDirection.DOWN:
            target_row = blank_row + 1
            target_col = blank_col
        elif move == MoveDirection.LEFT:
            target_row = blank_row
            target_col = blank_col - 1
        elif move == MoveDirection.RIGHT:
            target_row = blank_row
            target_col = blank_col + 1

        # Swap the blank tile with the target tile
        target_value = self.get_tile(target_row, target_col)
        self.set_tile(target_row, target_col, 0)
        self.set_tile(blank_row, blank_col, target_value)

    def copy(self):
        new_state = PuzzleState(self.size)
        new_state.tiles = self.tiles.copy()
        return new_state

    def __str__(self):
        result = ""
        for row in range(self.size):
            result += " ".join(str(self.get_tile(row, col)) for col in range(self.size)) + "\n"
        return result

# Define heuristics
def misplaced_tiles(state):
    count = 0
    for i in range(state.size):
        for j in range(state.size):
            if state.get_tile(i, j) != 0 and state.get_tile(i, j) != i * state.size + j + 1:
                count += 1
    return count

def manhattan_distance(state):
    distance = 0
    for i in range(state.size):
        for j in range(state.size):
            if state.get_tile(i, j) != 0:
                target_row = (state.get_tile(i, j) - 1) // state.size
                target_col = (state.get_tile(i, j) - 1) % state.size
                distance += abs(i - target_row) + abs(j - target_col)
    return distance

# Define search algorithms
def bfs(initial_state):
    frontier = [initial_state]
    explored = set()
    depth = 0
    max_fringe_size = 0
    expanded_nodes = 0

    while frontier:
        node = frontier.pop(0)
        explored.add(node)

        if node.is_goal_state():
            return depth, expanded_nodes, max_fringe_size

        expanded_nodes += 1

        for move in MoveDirection:
            if node.is_valid_move(move):
                child = node.copy()
                child.apply_move(move)
                if child not in explored and child not in frontier:
                    frontier.append(child)
        max_fringe_size = max(max_fringe_size, len(frontier))
        depth += 1

    return None

def dfs(initial_state):
    frontier = [initial_state]
    explored = set()
    depth = 0
    max_fringe_size = 0
    expanded_nodes = 0

    while frontier:
        node = frontier.pop()
        explored.add(node)

        if node.is_goal_state():
            return depth, expanded_nodes, max_fringe_size

        expanded_nodes += 1

        for move in MoveDirection:
            if node.is_valid_move(move):
                child = node.copy()
                child.apply_move(move)
                if child not in explored and child not in frontier:
                    frontier.append(child)
        max_fringe_size = max(max_fringe_size, len(frontier))
        depth += 1

    return None

def uniform_cost_search(initial_state):
    frontier = [(initial_state, 0)]
    explored = set()
    depth = 0
    max_fringe_size = 0
    expanded_nodes = 0

    while frontier:
        node, cost = frontier.pop(0)
        explored.add(node)

        if node.is_goal_state():
            return depth, expanded_nodes, max_fringe_size

        expanded_nodes += 1

        for move in MoveDirection:
            if node.is_valid_move(move):
                child = node.copy()
                child.apply_move(move)
                if child not in explored and (child, cost + 1) not in frontier:
                    frontier.append((child, cost + 1))
                    frontier.sort(key=lambda x: x[1])
        max_fringe_size = max(max_fringe_size, len(frontier))
        depth += 1

    return None 

def a_star_search(initial_state, heuristic):
    frontier = [(initial_state, 0 + heuristic(initial_state))]
    explored = set()
    depth = 0
    max_fringe_size = 0
    expanded_nodes = 0

    while frontier:
        node, cost = frontier.pop(0)
        explored.add(node)

        if node.is_goal_state():
            return depth, expanded_nodes, max_fringe_size

        expanded_nodes += 1

        for move in MoveDirection:
            if node.is_valid_move(move):
                child = node.copy()
                child.apply_move(move)
                child_cost = depth + 1 + heuristic(child)
                if child not in explored and (child, child_cost) not in frontier:
                    frontier.append((child, child_cost))
                    frontier.sort(key=lambda x: x[1])
        max_fringe_size = max(max_fringe_size, len(frontier))
        depth += 1

    return None

def generate_and_write_states_to_csv(filename, num_states, size):
    initial_states = []

    while len(initial_states) < num_states:
        # Create a random initial state
        initial_state = PuzzleState(size)
        initial_states.append(initial_state)

    # Write the initial states to a CSV file
    with open(filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        for state in initial_states:
            writer.writerow(state.tiles)

def load_states_from_csv(filename, size):
    states = []

    try:
        with open(filename, mode='r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                # Convert the row (list of integers) into a PuzzleState object
                state = PuzzleState(size)
                state.tiles = list(map(int, row))
                states.append(state)
    except FileNotFoundError:
        print(f"CSV file '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")

    return states

def run_search_algorithm(problem, algorithm, heuristic=None):
    if algorithm == "BFS":
        return bfs(problem)
    elif algorithm == "DFS":
        return dfs(problem)
    elif algorithm == "UniformCost":
        return uniform_cost_search(problem)
    elif algorithm == "AStar":
        if heuristic is None:
            raise ValueError("Heuristic function is required for A* Search")
        return a_star_search(problem, heuristic)

def main():
    # Get user input for puzzle size (N)
    while True:
        try:
            puzzle_size = int(input("Enter the puzzle size (N): "))
            if puzzle_size < 2:
                print("Puzzle size must be at least 2.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    # Generate and write initial states to CSV
    csv_filename = f'scenarios_{puzzle_size}x{puzzle_size}.csv'
    generate_and_write_states_to_csv(csv_filename, num_states=5, size=puzzle_size)

    # Load puzzle states from CSV
    states = load_states_from_csv(csv_filename, size=puzzle_size)

    algorithms = ["BFS", "DFS", "UniformCost", "AStar"]
    heuristics = [misplaced_tiles, manhattan_distance]

    results_df = pd.DataFrame(columns=["Algorithm", "Heuristic", "Puzzle", "Depth", "Explored Nodes", "Max Fringe Size", "Elapsed Time"])

    for i, state in enumerate(states, start=1):
        print(f"Puzzle {i}:")
        print(state)

        for algorithm in algorithms:
            for heuristic in heuristics:
                if algorithm == "AStar" and heuristic is None:
                    continue

                print(f"Running {algorithm} search for puzzle:")
                print(state)
                print(f"Using {heuristic.__name__ if heuristic else 'None'} heuristic:")

                start_time = timeit.default_timer()
                depth, expanded_nodes, max_fringe_size = run_search_algorithm(state, algorithm, heuristic)
                elapsed_time = timeit.default_timer() - start_time

                if depth is not None:
                    print(f"{algorithm} found a solution:")
                    print(f"Depth: {depth}")
                    print(f"Explored Nodes: {expanded_nodes}")
                    print(f"Max Fringe Size: {max_fringe_size}")
                    print(f"Elapsed Time (s): {elapsed_time:.4f}")
                else:
                    print(f"No solution found using {algorithm} with {heuristic.__name__ if heuristic else 'None'} heuristic.")

                print("-------------------------------------------------")
                
                results_df = pd.concat([results_df, pd.DataFrame({
                    "Algorithm": [algorithm],
                    "Heuristic": [heuristic.__name__ if heuristic else 'None'],
                    "Puzzle": [f"Puzzle {i}"],
                    "Depth": [depth],
                    "Explored Nodes": [expanded_nodes],
                    "Max Fringe Size": [max_fringe_size],
                    "Elapsed Time": [elapsed_time]
                })], ignore_index=True)

    results_df.to_csv(f"search_results_{puzzle_size}x{puzzle_size}.csv")

if __name__ == '__main__':
    main()
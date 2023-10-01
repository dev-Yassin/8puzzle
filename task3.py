from eightpuzzle import EightPuzzleState  # Import the EightPuzzleState class
import csv
import search
from eightpuzzle import EightPuzzleSearchProblem
import pandas as pd
import util
import timeit

def load_eight_puzzles_from_csv(filename):
    eight_puzzles = []

    try:
        with open(filename, mode='r') as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                # Convert the row (list of integers) into an EightPuzzleState object
                eight_puzzle_state = EightPuzzleState(list(map(int, row)))
                eight_puzzles.append(eight_puzzle_state)
    except FileNotFoundError:
        print(f"CSV file '{filename}' not found.")
    except Exception as e:
        print(f"An error occurred while reading the CSV file: {e}")

    return eight_puzzles

def run_search_algorithm(problem, algorithm, heuristic_type=None):
    if algorithm == "BFS":
        return search.breadthFirstSearch(problem)
    elif algorithm == "DFS":
        return search.depthFirstSearch(problem)
    elif algorithm == "uninformedCost":
        return search.uniformCostSearch(problem)
    elif algorithm == "AStar":
        if heuristic_type is None:
            raise ValueError("Heuristic type is required for A* Search")
        return search.aStarSearch(problem, lambda state, problem=None: problem.getHeuristicValue(state, heuristic_type=heuristic_type))

def main():
    # Specify the CSV filename where the initial states are stored
    csv_filename = 'scenarios.csv'
    algorithms = ["BFS", "DFS", "uninformedCost", "AStar"]
    heuristics = ["manhattan"]
    results_df = pd.DataFrame(columns=["Algorithm", "Heuristic", "Puzzle", "Depth", "Explored Nodes", "Max Fringe Size", "Elapsed Time", "Expanded Nodes"])
    # Load the EightPuzzleState objects from the CSV file
    eight_puzzles = load_eight_puzzles_from_csv(csv_filename)

    # Print the loaded EightPuzzleState objects
    for i, puzzle in enumerate(eight_puzzles, start=1):
        print(f"Puzzle {i}:")
        print(puzzle)
        
        for algorithm in algorithms:
            if algorithm == "AStar":
                for heuristic_type in heuristics:
                    print(f"Running {algorithm} search for puzzle:")
                    print(puzzle)
                    print(f"Using {heuristic_type} heuristic:")
                    problem = EightPuzzleSearchProblem(puzzle)
                    
                    start_time = timeit.default_timer()  # Record start time
                    actions, depth, explored_nodes, max_fringe_size, elapsed_time, expanded_nodes = run_search_algorithm(problem, algorithm, heuristic_type=heuristic_type)
                    elapsed_time = timeit.default_timer() - start_time  # Calculate elapsed time
                    
                    if actions:
                        print(f'{algorithm} found a path of {len(actions)} moves: {str(actions)}')
                        print('Depth:', depth)
                        print('Explored Nodes:', explored_nodes)
                        print('Max Fringe Size:', max_fringe_size)
                        print('Elapsed Time (ms):', elapsed_time)
                        print('Expanded Nodes:', expanded_nodes)
                    else:
                        print(f"No solution found using {algorithm} with {heuristic_type} heuristic.")
                    print("-------------------------------------------------")

                    # Add results to the DataFrame
                    results_df = pd.concat([results_df, pd.DataFrame({
                        "Algorithm": [algorithm],
                        "Heuristic": [heuristic_type],
                        "Puzzle": [str(puzzle)],
                        "Depth": [depth],
                        "Explored Nodes": [explored_nodes],
                        "Max Fringe Size": [max_fringe_size],
                        "Elapsed Time": [elapsed_time],
                        "Expanded Nodes": [expanded_nodes]
                    })], ignore_index=True)
            else:
                print(f"Running {algorithm} search for puzzle:")
                print(puzzle)
                problem = EightPuzzleSearchProblem(puzzle)

                start_time = timeit.default_timer()  # Record start time
                actions, depth, explored_nodes, max_fringe_size, elapsed_time, expanded_nodes = run_search_algorithm(problem, algorithm)
                elapsed_time = timeit.default_timer() - start_time  # Calculate elapsed time
                
                if actions:
                    print(f'{algorithm} found a path of {len(actions)} moves: {str(actions)}')
                    print('Depth:', depth)
                    print('Explored Nodes:', explored_nodes)
                    print('Max Fringe Size:', max_fringe_size)
                    print('Elapsed Time (ms):', elapsed_time)
                    print('Expanded Nodes:', expanded_nodes)
                else:
                    print(f"No solution found using {algorithm}.")
                print("-------------------------------------------------")

                # Add results to the DataFrame
                results_df = pd.concat([results_df, pd.DataFrame({
                    "Algorithm": [algorithm],
                    "Heuristic": [None],
                    "Puzzle": [str(puzzle)],
                    "Depth": [depth],
                    "Explored Nodes": [explored_nodes],
                    "Max Fringe Size": [max_fringe_size],
                    "Elapsed Time": [elapsed_time],
                    "Expanded Nodes": [expanded_nodes]
                })], ignore_index=True)

    # Save the results to a CSV file
    results_df.to_csv("__search_results.csv", index=False)

    # Create a dictionary to store average metrics for each algorithm
    avg_metrics = {algorithm: {} for algorithm in algorithms}

    # Calculate average metrics for each algorithm
    for algorithm in algorithms:
        avg_metrics[algorithm]["Average Depth"] = results_df[results_df["Algorithm"] == algorithm]["Depth"].mean()
        avg_metrics[algorithm]["Average Explored Nodes"] = results_df[results_df["Algorithm"] == algorithm]["Explored Nodes"].mean()
        avg_metrics[algorithm]["Average Max Fringe Size"] = results_df[results_df["Algorithm"] == algorithm]["Max Fringe Size"].mean()
        avg_metrics[algorithm]["Average Expanded Nodes"] = results_df[results_df["Algorithm"] == algorithm]["Expanded Nodes"].mean()
        avg_metrics[algorithm]["Average Time"] = results_df[results_df["Algorithm"] == algorithm]["Elapsed Time"].mean()

    # Create a DataFrame with columns for each algorithm's average metrics
    avg_metrics_df = pd.DataFrame.from_dict(avg_metrics, orient='index')

    # Reset the index to add algorithm as a column
    avg_metrics_df.reset_index(level=0, inplace=True)
    avg_metrics_df.rename(columns={'index': 'Algorithm'}, inplace=True)

    # Save the average metrics to a CSV file
    avg_metrics_df.to_csv("__average_metrics.csv", index=False)

if __name__ == '__main__':
    main()

import csv
from eightpuzzle import EightPuzzleState, EightPuzzleSearchProblem, search, chooseHeuristic
import pandas as pd
# Function to read initial states from CSV
def read_initial_states_from_csv(filename):
    initial_states = []
    with open(filename, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            # Convert row to a list of integers, replacing '0' with None for empty cells
            state = [int(cell) if cell != '0' else None for cell in row]
            initial_states.append(state)
    return initial_states

# Function to apply A* search with a chosen heuristic on an initial state
def run_a_star_on_initial_state(initial_state, heuristic_type):
    puzzle = EightPuzzleState(initial_state)
    problem = EightPuzzleSearchProblem(puzzle)
    path, depth, explored_nodes, max_fringe_size, expanded_nodes, elapsed_time = search.aStarSearch(
        problem, lambda state, problem=None: problem.getHeuristicValue(state, heuristic_type=heuristic_type)
    )
    return path, depth, explored_nodes, max_fringe_size, expanded_nodes, elapsed_time

if __name__ == '__main__':
    # Read initial states from CSV file
    initial_states = read_initial_states_from_csv('scenarios.csv')

    # Choose a heuristic
    heuristic_type = chooseHeuristic()

    # Apply A* search on each initial state and print results
    for i, initial_state in enumerate(initial_states, start=1):
        print(f"Running A* search for initial state {i}...")
        path, depth, explored_nodes, max_fringe_size, expanded_nodes, elapsed_time = run_a_star_on_initial_state(
            initial_state, heuristic_type
        )

        if path:
            print('A* found a path of %d moves: %s' % (len(path), str(path)))
            print('Depth:', depth)
            print('Explored Nodes:', explored_nodes)
            print('Expanded Nodes:', expanded_nodes)
            print('Max Fringe Size:', max_fringe_size)
            print('Elapsed Time (ms):', elapsed_time)
        else:
            print("No solution found using this heuristic.")
        print("-------------------------------------------------")
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
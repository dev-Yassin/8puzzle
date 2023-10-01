from eightpuzzle import EightPuzzleState  # Import the EightPuzzleState class
import csv
import search
from eightpuzzle import EightPuzzleSearchProblem 
import pandas as pd
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

if __name__ == '__main__':
    # Specify the CSV filename where the initial states are stored
    csv_filename = 'scenarios.csv'
    heuristics = ["misplaced", "euclidean", "manhattan", "out_of_position"]
    results_df = pd.DataFrame(columns=["Heuristic", "Puzzle", "Depth", "Explored Nodes", "Max Fringe Size", "Elapsed Time", "Expanded Nodes"])
    # Load the EightPuzzleState objects from the CSV file
    eight_puzzles = load_eight_puzzles_from_csv(csv_filename)

    #Print the loaded EightPuzzleState objects
    for i, puzzle in enumerate(eight_puzzles, start=1):
        print(f"Puzzle {i}:")
        print(puzzle)
        # Iterate over each puzzle and run A* search with different heuristics
    for puzzle in eight_puzzles:
        for heuristic_type in heuristics:
            print(f"Running A* search for puzzle:")
            print(puzzle)
            print(f"Using {heuristic_type} heuristic:")
            problem = EightPuzzleSearchProblem(puzzle)
            path, depth, explored_nodes, max_fringe_size, elapsed_time, expanded_nodes = search.aStarSearch(problem, lambda state, problem=None: problem.getHeuristicValue(state, heuristic_type=heuristic_type))
        
            if path:
                print('A* found a path of %d moves: %s' % (len(path), str(path)))
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

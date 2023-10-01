import csv
import random

def generate_and_write_easy_states_to_csv(filename, num_states, n, num_moves=25):
    initial_states = []

    while len(initial_states) < num_states:
        # Create a solved puzzle
        solved_puzzle = list(range(n * n))

        # Apply a small number of random moves
        for _ in range(num_moves):
            legal_moves = get_legal_moves(solved_puzzle, n)
            move = random.choice(legal_moves)
            swap_tiles(solved_puzzle, move, n)

        initial_states.append(solved_puzzle)

    # Write the initial states to a CSV file
    with open(filename, mode='w', newline='') as scenarios_file:
        writer = csv.writer(scenarios_file)
        for state in initial_states:
            writer.writerow(state)

def get_legal_moves(puzzle, n):
    # Determine legal moves based on the location of the empty tile (0)
    empty_tile_index = puzzle.index(0)
    legal_moves = []

    if empty_tile_index % n > 0:
        legal_moves.append('left')
    if empty_tile_index % n < n - 1:
        legal_moves.append('right')
    if empty_tile_index >= n:
        legal_moves.append('up')
    if empty_tile_index < n * (n - 1):
        legal_moves.append('down')

    return legal_moves

def swap_tiles(puzzle, move, n):
    empty_tile_index = puzzle.index(0)

    if move == 'left':
        puzzle[empty_tile_index], puzzle[empty_tile_index - 1] = puzzle[empty_tile_index - 1], puzzle[empty_tile_index]
    elif move == 'right':
        puzzle[empty_tile_index], puzzle[empty_tile_index + 1] = puzzle[empty_tile_index + 1], puzzle[empty_tile_index]
    elif move == 'up':
        puzzle[empty_tile_index], puzzle[empty_tile_index - n] = puzzle[empty_tile_index - n], puzzle[empty_tile_index]
    elif move == 'down':
        puzzle[empty_tile_index], puzzle[empty_tile_index + n] = puzzle[empty_tile_index + n], puzzle[empty_tile_index]

if __name__ == '__main__':
    # Define the number of initial states to generate, the puzzle size (N), and the filename
    NUM_INITIAL_STATES = 5
    N = 4  # Change N to the desired puzzle size
    FILENAME = f'scenarios_{N}x{N}.csv'

    # Call the method to generate and write initial states to CSV
    generate_and_write_easy_states_to_csv(FILENAME, NUM_INITIAL_STATES, N)

























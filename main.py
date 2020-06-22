import numpy as np
from typing import Optional, Callable
from agents.common_arrays import top_row
from agents.common_arrays import PlayerAction, Board, BoardPiece, SavedState, GenMove
# from agents.agent_minimax import generate_move
# from agents.agent_alpha_beta import generate_move, generate_move2
from agents.agent_alpha_beta import generate_move
import cProfile


def user_move(board: Board, _player: BoardPiece,
              saved_state: Optional[SavedState]):
    """ Prompts the human player to select a column to play in

    :param board: 2d array representing current state of the game
    :param _player: the player making the current move (active player)
    :param saved_state: ???

    :return: returns the chosen action and the saved state
    """

    # Initialize the action as a negative and cast as type PlayerAction
    action = PlayerAction(-1)

    # While action is not a valid move
    while not (0 <= action < board.shape[1]):
        try:
            # Human player input action
            action = input("Select column to play (0-6): ")
            # If no input is entered, raise IndexError
            if not action:
                raise IndexError
            # Cast as type PlayerAction
            action = PlayerAction(action)
            # Test whether the column is full. If it is, top_row will throw
            # an IndexError
            top_row(board, action)
        except IndexError:
            print('This is not a valid action. Please choose again.')
            action = -1
            continue

    return action, saved_state


def human_vs_agent(generate_move_1: GenMove,
                   generate_move_2: GenMove = user_move,
                   player_1: str = "Player 1",
                   player_2: str = "Player 2",
                   args_1: tuple = (),
                   args_2: tuple = (),
                   init_1: Callable = lambda board, player: None,
                   init_2: Callable = lambda board, player: None):

    import time
    from agents.common_arrays import PLAYER1, PLAYER2, GameState
    from agents.common_arrays import initialize_game_state, pretty_print_board,\
        apply_player_action, check_end_state

    players = (PLAYER1, PLAYER2)

    # Play two games, where each player gets a chance to go first
    for play_first in (1, -1):
        # Initialize a string to store actions
        game_moves_out = ''
        game_moves = ''
        # This loop initializes the variables to speed up computation when
        # using the numba compiler
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                    players, player_names, gen_moves, gen_args):

                # Time how long a move takes
                t0 = time.time()

                # Inform the player which letter represents them
                print(pretty_print_board(board))
                print(f'{player_name} you are playing with '
                      f'{"X" if player == PLAYER1 else "O"}')

                # Generate an action, either through user input or by an
                # agent function
                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], *args)

                print(f"Move time: {time.time() - t0:.3f}s")

                # Save the move
                game_moves_out += str(action)
                game_moves += str(action)
                game_moves += ', '
                # Update the board with the action
                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)

                # Check to see whether the game is a win or draw
                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))

                    if end_state == GameState.IS_DRAW:
                        print("Game ended in draw")
                        print(game_moves)
                    else:
                        print(f'{player_name} won playing '
                              f'{"X" if player == PLAYER1 else "O"}')
                        print(game_moves)
                        text_file = open("tests/Test_cases_wins", "a")
                        text_file.write(game_moves_out+'\n')
                        text_file.close()

                    playing = False
                    break


# cProfile.run("human_vs_agent(generate_move, generate_move2)", "tests/mmab_compete")
# cProfile.run("human_vs_agent(generate_move, generate_move)", "tests/mmab_all_bits")

if __name__ == "__main__":
    # human_vs_agent(user_move)
    human_vs_agent(generate_move)

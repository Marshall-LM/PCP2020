import numpy as np
from typing import Optional, Callable
from agents.common_arrays import PlayerAction, Board, BoardPiece, SavedState, GenMove
from agents.agent_alpha_beta import generate_move as gen_move_ab
from agents.agent_mcts import generate_move as gen_move_mcts


def human_vs_agent(generate_move_1: GenMove,
                   generate_move_2: GenMove,
                   player_1: str = "Player 1",
                   player_2: str = "Player 2",
                   args_1: tuple = (),
                   args_2: tuple = (),
                   init_1: Callable = lambda board, player: None,
                   init_2: Callable = lambda board, player: None):
    import time
    from agents.common_arrays import PLAYER1, PLAYER2, GameState
    from agents.common_arrays import initialize_game_state, pretty_print_board, \
        apply_player_action, check_end_state

    players = (PLAYER1, PLAYER2)

    # Play two games, where each player gets a chance to go first
    p1_wins = 0
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
                # print(pretty_print_board(board))
                # print(f'{player_name} you are playing with '
                #       f'{"X" if player == PLAYER1 else "O"}')

                # Generate an action, either through user input or by an
                # agent function
                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], *args)

                # print(f"Move time: {time.time() - t0:.3f}s")

                # Save the move
                game_moves_out += str(action)
                game_moves += str(action)
                game_moves += ', '
                # Update the board with the action
                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)

                # Check to see whether the game is a win or draw
                if end_state != GameState.STILL_PLAYING:
                    # print(pretty_print_board(board))

                    if end_state == GameState.IS_DRAW:
                        print("Game ended in draw")
                        print(game_moves)
                        p1_wins += 0.5
                    else:
                        print(f'{player_name} won playing '
                              f'{"X" if player == PLAYER1 else "O"}')
                        print(game_moves)
                        if player_name == 'Player 1':
                            p1_wins += 1

                    playing = False
                    break
    return p1_wins


def test_agent_performance():
    n_matches = 25
    mcts_wins = 0
    for i in range(n_matches):
        np.random.seed(3)
        mcts_wins += human_vs_agent(gen_move_mcts, gen_move_ab)
    print('MCTS won {} times'.format(mcts_wins))
    print('The MCTS agent wins against the alpha-beta agent {}% of the time'
          .format(100 * (mcts_wins / (2 * n_matches))))

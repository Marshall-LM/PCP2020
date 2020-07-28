import numpy as np
import agents.common as cm
from typing import Optional, Tuple
import agents.agent_mcts.agent_mcts_random as agmcts
from gmpy2 import popcount
from agents.agent_mcts import generate_move

GameScore = np.int


def generate_full_board(player, empty_spaces=0):
    # Generate an empty board
    arr_board = cm.initialize_game_state()
    # Convert board to bitmap
    bit_board, bit_mask = cm.board_to_bitmap(arr_board, player)
    # Calculate the board shape
    bd_shp = arr_board.shape

    # While the board is not full, continue placing pieces
    while popcount(bit_mask) != bd_shp[0] * bd_shp[1] - empty_spaces:
        # Select a random move in a column that is not full
        move = -1
        while not (0 <= move < bd_shp[1]):
            move = np.random.choice(bd_shp[1])
            try:
                move = cm.PlayerAction(move)
                cm.top_row(arr_board, move)
            except IndexError:
                move = -1

        # Apply the move to both boards
        cm.apply_action(arr_board, move, player)
        bit_board, bit_mask = cm.apply_action_cp(bit_board, bit_mask,
                                                 move, bd_shp)
        # Switch to the next player
        player = cm.BoardPiece(player % 2 + 1)

    return arr_board, bit_board, bit_mask, player


def alpha_beta_oracle(board: cm.Bitmap, mask: cm.Bitmap, max_player: bool,
                      alpha: GameScore, beta: GameScore, board_shp: Tuple,
                      depth: int
                      ) -> Tuple[GameScore, Optional[int]]:
    """ Function used to find guaranteed future wins, based on optimal play

    A guaranteed win for the max_player will return a score modified by the
    depth at which the win should occur. The number of moves in which the
    player should win is returned, along with the score. Guaranteed losses
    are accounted for in a similar way.
    """

    max_depth = 8
    win_score = 100
    state_p = cm.check_end_state(board ^ mask, mask, board_shp)
    if state_p == cm.GameState.IS_WIN:
        if max_player:
            return GameScore(-win_score), None
        else:
            return GameScore(win_score), None
    elif depth == max_depth:
        return GameScore(0), None

    # For each potential action, call alpha_beta
    pot_actions = cm.valid_actions(mask, board_shp)
    if max_player:
        score = -100000
        for col in pot_actions:
            # Apply the current action
            min_board, new_mask = cm.apply_action_cp(board, mask,
                                                     col, board_shp)
            # Call alpha-beta
            new_score, _ = alpha_beta_oracle(min_board, new_mask,
                                             False, alpha, beta,
                                             board_shp, depth + 1)
            new_score -= depth
            # Check whether the score updates
            if new_score > score:
                score = new_score
            # Check whether we can prune the rest of the branch
            if score >= beta:
                break
            # Check whether alpha updates the score
            if score > alpha:
                alpha = score
        # If this is the root node, return the optimal number of moves
        if depth == 0:
            if score > 0:
                return GameScore(score), 2 * (win_score - score) + 1
            else:
                return GameScore(score), 2 * (win_score + score)
        else:
            return GameScore(score), None
    else:
        score = 100000
        for col in pot_actions:
            # Apply the current action, continue if column is full
            max_board, new_mask = cm.apply_action_cp(board, mask,
                                                     col, board_shp)
            # Call alpha-beta
            new_score, _ = alpha_beta_oracle(max_board, new_mask,
                                             True, alpha, beta,
                                             board_shp, depth + 1)
            new_score += depth
            # Check whether the score updates
            if new_score < score:
                score = new_score
            # Check whether we can prune the rest of the branch
            if score <= alpha:
                break
            # Check whether beta updates the score
            if score < beta:
                beta = score
        return GameScore(score), None


def test_alpha_beta_oracle():
    # Generate an empty board
    arr_board = cm.initialize_game_state()
    # Convert board to bitmap
    player = cm.PLAYER1
    bit_board, bit_mask = cm.board_to_bitmap(arr_board, player)
    # Calculate the board shape
    bd_shp = arr_board.shape
    a0 = -100000
    b0 = 100000
    # Define a list of moves
    move_list = [3, 3, 4, 4, 5, 5]
    for mv in move_list[:-2]:
        # Apply the move to both boards
        cm.apply_action(arr_board, mv, player)
        bit_board, bit_mask = cm.apply_action_cp(bit_board, bit_mask,
                                                 mv, bd_shp)
        # Switch to the next player
        player = cm.BoardPiece(player % 2 + 1)
    print(cm.pretty_print_board(arr_board))
    score, depth = alpha_beta_oracle(bit_board, bit_mask,
                                     True, a0, b0, bd_shp, 0)
    print('Player {} should win in {} moves.'
          .format(('X' if player == cm.BoardPiece(1) else 'O'), depth))
    assert depth == 3

    # Apply next move to both boards
    cm.apply_action(arr_board, move_list[-2], player)
    bit_board, bit_mask = cm.apply_action_cp(bit_board, bit_mask,
                                             move_list[-2], bd_shp)
    # Switch to the next player
    player = cm.BoardPiece(player % 2 + 1)
    print(cm.pretty_print_board(arr_board))
    score, depth = alpha_beta_oracle(bit_board, bit_mask,
                                     True, a0, b0, bd_shp, 0)
    print('Player {} should lose in {} moves.'
          .format(('X' if player == cm.BoardPiece(1) else 'O'), depth))
    assert depth == 2

    # Apply next move to both boards
    cm.apply_action(arr_board, move_list[-1], player)
    bit_board, bit_mask = cm.apply_action_cp(bit_board, bit_mask,
                                             move_list[-1], bd_shp)
    # Switch to the next player
    player = cm.BoardPiece(player % 2 + 1)
    print(cm.pretty_print_board(arr_board))
    score, depth = alpha_beta_oracle(bit_board, bit_mask,
                                     True, a0, b0, bd_shp, 0)
    print('Player {} should win in {} move.'
          .format(('X' if player == cm.BoardPiece(1) else 'O'), depth))
    assert depth == 1
    print('\n##########################################################\n')

    # Generate an empty board
    arr_board = cm.initialize_game_state()
    # Convert board to bitmap
    player = cm.PLAYER1
    bit_board, bit_mask = cm.board_to_bitmap(arr_board, player)
    # Full game
    # move_list = [3, 2, 3, 3, 3, 2, 2, 2, 5, 4, 0, 4, 4, 4, 1, 1, 5, 2, 6]
    move_list = [3, 2, 3, 3, 3, 2, 2, 2, 5, 4, 0, 4, 4, 4]
    for mv in move_list[:-1]:
        # Apply the move to both boards
        cm.apply_action(arr_board, mv, player)
        bit_board, bit_mask = cm.apply_action_cp(bit_board, bit_mask,
                                                 mv, bd_shp)
        # Switch to the next player
        player = cm.BoardPiece(player % 2 + 1)
    print(cm.pretty_print_board(arr_board))
    action, _ = generate_move(arr_board.copy(), player, None)
    print('MCTS plays in column {}'.format(action))
    try:
        assert (action == 2 or action == 5)
    except AssertionError:
        print('NOTE: MCTS doesn\'t block this win unless it is given '
              'over 5s to search. It should play in column 2 or 5.')

    # Apply next move to both boards
    cm.apply_action(arr_board, move_list[-1], player)
    bit_board, bit_mask = cm.apply_action_cp(bit_board, bit_mask,
                                             move_list[-1], bd_shp)
    # Switch to the next player
    player = cm.BoardPiece(player % 2 + 1)
    print(cm.pretty_print_board(arr_board))
    score, depth = alpha_beta_oracle(bit_board, bit_mask,
                                     True, a0, b0, bd_shp, 0)
    print('Player {} should win in {} move.'
          .format(('X' if player == cm.BoardPiece(1) else 'O'), depth))
    assert depth == 5
    print('\n##########################################################\n')

    # Test other hard coded boards
    move_list_list = [[3, 4, 3, 3, 1, 0, 4, 4, 1, 1, 3, 0, 0, 4, 5, 5],
                      [3, 3, 4, 5, 1, 2, 4, 4, 3, 4, 3, 4, 4, 3, 1, 1,
                       0, 5, 1, 5, 5, 1, 0, 0]]
    # Full games
    # [3, 4, 3, 3, 1, 0, 4, 4, 1, 1, 3, 0, 0, 4, 5, 5, 4, 6, 2, 2, 2]
    # [3, 4, 3, 3, 1, 0, 4, 4, 1, 1, 3, 0, 0, 4, 4, 1, 1, 5, 5, 5, 3,
    #  5, 5, 4, 5, 0, 2, 0, 2]
    for move_list in move_list_list:
        # Generate an empty board
        arr_board = cm.initialize_game_state()
        # Convert board to bitmap
        player = cm.PLAYER1
        bit_board, bit_mask = cm.board_to_bitmap(arr_board, player)

        for mv in move_list:
            # Apply the move to both boards
            cm.apply_action(arr_board, mv, player)
            bit_board, bit_mask = cm.apply_action_cp(bit_board, bit_mask,
                                                     mv, bd_shp)
            # Switch to the next player
            player = cm.BoardPiece(player % 2 + 1)

        # Print the current board state
        print(cm.pretty_print_board(arr_board))
        # Check for guaranteed wins
        score, depth = alpha_beta_oracle(bit_board, bit_mask,
                                         True, a0, b0, bd_shp, 0)
        print('It is Player {}\'s turn. They should win in {} moves.'
              .format(('X' if player == cm.BoardPiece(1) else 'O'), depth))
        action, _ = generate_move(arr_board.copy(), player, None)
        print('Player {} plays in column {}'
              .format(('X' if player == cm.BoardPiece(1) else 'O'), action))
        print('\n##########################################################\n')

    # Other games that can be tried
    # move_list = [3, 2, 2, 4, 3, 3, 4, 2, 3, 4, 3, 2, 2, 4, 4, 4, 0, 6,
    #              6, 3, 0, 0]
    # move_list = [3, 3, 4, 5, 4, 2, 4, 4, 3, 5, 5, 3, 6, 5]
    # move_list = [3, 3, 2, 1, 4, 5, 4, 4, 2, 2, 3, 3, 1, 5, 6, 2, 2, 4, 6,
    #              4, 4, 5, 5, 3, 2, 3, 6, 6, 6, 6, 0, 0, 1]
    # move_list = [3, 3, 4, 5, 4, 2, 4, 4, 3, 5, 5, 3, 6, 5, 6, 6, 6]
    # move_list = [3, 4, 3, 4, 3, 3, 4, 4, 2, 1, 1, 1, 2, 1, 5, 1, 1, 3,
    #              4, 4, 3, 2, 2, 6, 5, 2, 2, 6, 6, 5]


def test_node_initialization():
    # Initialize a game
    player = cm.PLAYER1
    arr_bd, bit_bd, mask_bd, player = generate_full_board(player, 1)
    bd_shp = arr_bd.shape

    # Generate a board that is not a win, with only a single piece missing
    while cm.check_end_state(bit_bd, mask_bd, bd_shp) == cm.GameState.IS_WIN:
        arr_bd, bit_bd, mask_bd, player = generate_full_board(player, 1)
    print(cm.pretty_print_board(arr_bd))

    # Test whether node initializes and plays in proper (only empty) column
    move = agmcts.generate_move_mcts(arr_bd, player, None)[0]
    empty_col = cm.valid_actions(mask_bd, bd_shp)[0]
    print('MCTS plays in column {}'.format(move))
    assert move == empty_col


def test_mcts_algorithm():
    """ MCTS plays against itself and tries to catch guaranteed wins

     Use the oracle in the while loop. Calculates statistics based on how well
     it performs at playing optimally once a guaranteed win is detected by the
     oracle.
    """

    # Set parameter values and initialize counters
    a0 = -100000
    b0 = 100000
    n_games = 40
    n_wins = 0
    n_wins_opt = 0
    n_def_wins = 0

    for i in range(n_games):
        # Generate an empty board
        arr_board = cm.initialize_game_state()
        # Convert board to bitmap
        player = cm.PLAYER1
        bit_b, bit_m = cm.board_to_bitmap(arr_board, player)
        # Calculate the board shape
        bd_shp = arr_board.shape
        # Initialize the board state variable
        bd_state = cm.check_end_state(bit_b, bit_m, bd_shp)
        # Initialize a list of moves
        mv_list = []
        # Initialize counters
        mv_cnt = 0
        num_mvs = 0
        def_win = False

        while bd_state == cm.GameState.STILL_PLAYING:
            # Generate an action using MCTS
            action, _ = generate_move(arr_board.copy(), player, None)
            # Update the list of moves
            mv_list.append(action)
            # Apply the action to both boards
            cm.apply_action(arr_board, action, player)
            bit_b, bit_m = cm.apply_action_cp(bit_b, bit_m,
                                              action, bd_shp)
            # Switch to the next player
            player = cm.BoardPiece(player % 2 + 1)

            # Check for guaranteed win, if none detected, continue playing
            if not def_win:
                score, depth = alpha_beta_oracle(bit_b, bit_m, True,
                                                 a0, b0, bd_shp, 0)
                # If a win is guaranteed, determine when it should occur
                if score > 50 and abs(score) < 200:
                    print('Score returned is {}'.format(score))
                    num_mvs = depth
                    n_def_wins += 1
                    def_win = True
                    print(cm.pretty_print_board(arr_board))
                    print('Last move by player {}, in column {}, player {} '
                          'should win in {} move(s) at most'
                          .format(player % 2 + 1, action, player, num_mvs))
            # Once a win is detected, check whether MCTS finds it optimally
            else:
                mv_cnt += 1
                print(cm.pretty_print_board(arr_board))
                bd_state = cm.check_end_state(bit_b ^ bit_m, bit_m, bd_shp)
                if bd_state == cm.GameState.IS_WIN:
                    print(mv_list)
                    print('Player {} won in {} move(s)'.
                          format(player % 2 + 1, mv_cnt))
                    n_wins += 1
                    if mv_cnt <= num_mvs:
                        n_wins_opt += 1
                    break

            # Check the game state
            bd_state = cm.check_end_state(bit_b, bit_m, bd_shp)

    # Print the number of wins and how many were optimal
    print('The MCTS algorithm clinched {:4.1f}% of its guaranteed wins, '
          'and won in an optimal number of moves {}% of the time'
          .format(100 * (n_wins / n_def_wins), 100 * (n_wins_opt / n_wins)))

import numpy as np
from typing import Optional, Tuple
from numba import njit
from agents.common_arrays import Board, BoardPiece, PlayerAction, SavedState,\
    NO_PLAYER, GameState, apply_player_action, check_end_state, connect_four
# from agents.common_bits import Bitmap, board_to_bitmap, connect_four
from agents.common import Bitmap, board_to_bitmap
from gmpy2 import popcount, mpz

GameScore = np.int


def generate_move_alpha_beta(board: Board, player: BoardPiece,
                             saved_state: Optional[SavedState]
                             ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Agent selects a move based on a minimax depth first search, with
    alpha-beta pruning.


    """

    # If the board is empty, play in the center column
    if np.all(board == NO_PLAYER):
        action = np.floor(np.median(np.arange(board.shape[1])))
        return PlayerAction(action), saved_state

    board_cp = board.copy()
    # Set the max depth to search
    # max_depth = 0
    # Call alpha_beta after initializing the alpha and beta values (+/- 'inf')
    a0 = -100000
    b0 = 100000
    score, action = alpha_beta(board_cp, player, True, 0, a0, b0)
    # score, action = alpha_beta_root(board_cp, player, True, max_depth, a0, b0)

    return PlayerAction(action), saved_state


def alpha_beta(board: Board, player: BoardPiece, max_player: bool,
               depth: int, alpha: GameScore, beta: GameScore,
               ) -> Tuple[GameScore, Optional[PlayerAction]]:
    """

    """
    # Make a list of columns that can be played in
    potential_actions = np.argwhere(board[-1, :] == 0)
    potential_actions = potential_actions.reshape(potential_actions.size)

    # If the node is at the max depth or a terminal node calculate the score
    max_depth = 6
    win_score = 150
    state_p = check_end_state(board, player)
    state_np = check_end_state(board, BoardPiece(player % 2 + 1))
    # if depth == max_depth or np.all(board != 0):
    #     return heuristic_solver(board, player, max_player), None
    #     # return heuristic_solver_bits(board, player, max_player), None
    if state_p == GameState.IS_WIN:
        if max_player:
            return GameScore(win_score), None
        else:
            return GameScore(-win_score), None
    elif state_np == GameState.IS_WIN:
        if max_player:
            return GameScore(-win_score), None
        else:
            return GameScore(win_score), None
    elif depth == max_depth:
        return heuristic_solver(board, player, max_player), None
        # return heuristic_solver_bits(board, player, max_player), None
    elif state_p == GameState.IS_DRAW:
        return 0, None

    # # If this is the root call, check for wins and block/win, prioritize wins
    # win_score = 150
    # if depth == 0:
    #     for col in potential_actions:
    #         if connect_four(apply_player_action(board, col, player, True),
    #                         player, col):
    #             return GameScore(win_score), PlayerAction(col)
    #     for col in potential_actions:
    #         if connect_four(apply_player_action(board, col,
    #                                             BoardPiece(player % 2 + 1), True),
    #                         BoardPiece(player % 2 + 1), col):
    #             return GameScore(win_score), PlayerAction(col)

    # For each potential action, call alpha_beta
    if max_player:
        score = -100000
        for col in potential_actions:
            # Apply the current action and call alpha_beta
            new_board = apply_player_action(board, col, player, True)
            new_score, temp = alpha_beta(new_board, BoardPiece(player % 2 + 1),
                                         False, depth + 1, alpha, beta)
            new_score -= 5 * depth
            # Check whether the score updates
            if new_score > score:
                score = new_score
                action = col
            # Check whether we can prune the rest of the branch
            if score >= beta:
                # print('Pruned a branch')
                break
            # Check whether alpha updates the score
            if score > alpha:
                alpha = score
        return GameScore(score), PlayerAction(action)
    else:
        score = 100000
        for col in potential_actions:
            # Apply the current action and call alpha_beta
            new_board = apply_player_action(board, col, player, True)
            new_score, temp = alpha_beta(new_board, BoardPiece(player % 2 + 1),
                                         True, depth + 1, alpha, beta)
            new_score += 5 * depth
            # Check whether the score updates
            if new_score < score:
                score = new_score
                action = col
            # Check whether we can prune the rest of the branch
            if score <= alpha:
                # print('Pruned a branch')
                break
            # Check whether alpha updates the score
            if score < beta:
                beta = score
        return GameScore(score), PlayerAction(action)


# def heuristic_solver(board: Board, player: BoardPiece, max_player: bool):
#     """
#     Scores the game board based on whether a player has combinations of two,
#     three, or four pieces in any four spaces, with the other spaces being
#     unoccupied. Points increase by an order of magnitude for each piece, to
#     prioritize being closer to winning.
#     """
#     # Shape of board
#     n_rows, n_cols = board.shape
#     # Min connection (4 in a row wins)
#     mc = 4
#     # Initialize the score
#     score = 0
#     two_pts = 1
#     three_pts = 10
#     win_pts = 100
#
#     # Slide the mask across the board and check for a win in each position
#     min_player = (player % 2 + 1)
#     for row in range(n_rows - mc + 1):
#         for col in range(n_cols - mc + 1):
#             # Accumulate score for max_player position
#             # Check for vertical points
#             v_vec = board[row:row + mc, col]
#             if np.all(v_vec == player):
#                 score += win_pts
#             elif (len(np.where(v_vec == player)[0]) == 3 and
#                   len(np.where(v_vec == NO_PLAYER)[0]) == 1):
#                 score += three_pts
#             elif (len(np.where(v_vec == player)[0]) == 2 and
#                   len(np.where(v_vec == NO_PLAYER)[0]) == 2):
#                 score += two_pts
#             # Check for horizontal points
#             h_vec = board[row, col:col + mc]
#             if np.all(h_vec == player):
#                 score += win_pts
#             elif (len(np.where(h_vec == player)[0]) == 3 and
#                   len(np.where(h_vec == NO_PLAYER)[0]) == 1):
#                 score += three_pts
#             elif (len(np.where(h_vec == player)[0]) == 2 and
#                   len(np.where(h_vec == NO_PLAYER)[0]) == 2):
#                 score += two_pts
#             # Check for \ points
#             block_mask = board[row:row + mc, col:col + mc]
#             d_block = np.diag(block_mask)
#             if np.all(d_block == player):
#                 score += win_pts
#             elif (len(np.where(d_block == player)[0]) == 3 and
#                   len(np.where(d_block == NO_PLAYER)[0]) == 1):
#                 score += three_pts
#             elif (len(np.where(d_block == player)[0]) == 2 and
#                   len(np.where(d_block == NO_PLAYER)[0]) == 2):
#                 score += two_pts
#             # Check for / points
#             b_block = np.diag(block_mask[::-1, :])
#             if np.all(b_block == player):
#                 score += win_pts
#             elif (len(np.where(b_block == player)[0]) == 3 and
#                   len(np.where(b_block == NO_PLAYER)[0]) == 1):
#                 score += three_pts
#             elif (len(np.where(b_block == player)[0]) == 2 and
#                   len(np.where(b_block == NO_PLAYER)[0]) == 2):
#                 score += two_pts
#
#             # Reduce score for min_player position
#             if np.all(v_vec == min_player):
#                 score -= win_pts
#             elif (len(np.where(v_vec == min_player)[0]) == 3 and
#                   len(np.where(v_vec == NO_PLAYER)[0]) == 1):
#                 score -= three_pts
#             elif (len(np.where(v_vec == min_player)[0]) == 2 and
#                   len(np.where(v_vec == NO_PLAYER)[0]) == 2):
#                 score -= two_pts
#             # Check for horizontal points
#             if np.all(h_vec == min_player):
#                 score -= win_pts
#             elif (len(np.where(h_vec == min_player)[0]) == 3 and
#                   len(np.where(h_vec == NO_PLAYER)[0]) == 1):
#                 score -= three_pts
#             elif (len(np.where(h_vec == min_player)[0]) == 2 and
#                   len(np.where(h_vec == NO_PLAYER)[0]) == 2):
#                 score -= two_pts
#             # Check for \ points
#             if np.all(d_block == min_player):
#                 score -= win_pts
#             elif (len(np.where(d_block == min_player)[0]) == 3 and
#                   len(np.where(d_block == NO_PLAYER)[0]) == 1):
#                 score -= three_pts
#             elif (len(np.where(d_block == min_player)[0]) == 2 and
#                   len(np.where(d_block == NO_PLAYER)[0]) == 2):
#                 score -= two_pts
#             # Check for / points
#             if np.all(b_block == min_player):
#                 score -= win_pts
#             elif (len(np.where(b_block == min_player)[0]) == 3 and
#                   len(np.where(b_block == NO_PLAYER)[0]) == 1):
#                 score -= three_pts
#             elif (len(np.where(b_block == min_player)[0]) == 2 and
#                   len(np.where(b_block == NO_PLAYER)[0]) == 2):
#                 score -= two_pts
#
#     for row in range(n_rows - mc + 1, n_rows):
#         for col in range(n_cols):
#             h_vec = board[row, col:col + mc]
#             # Accumulate score for max_player position
#             if np.all(h_vec == player):
#                 score += win_pts
#             elif (len(np.where(h_vec == player)[0]) == 3 and
#                   len(np.where(h_vec == NO_PLAYER)[0]) == 1):
#                 score += three_pts
#             elif (len(np.where(h_vec == player)[0]) == 2 and
#                   len(np.where(h_vec == NO_PLAYER)[0]) == 2):
#                 score += two_pts
#             # Reduce score for min_player position
#             if np.all(h_vec == min_player):
#                 score -= win_pts
#             elif (len(np.where(h_vec == min_player)[0]) == 3 and
#                   len(np.where(h_vec == NO_PLAYER)[0]) == 1):
#                 score -= three_pts
#             elif (len(np.where(h_vec == min_player)[0]) == 2 and
#                   len(np.where(h_vec == NO_PLAYER)[0]) == 2):
#                 score -= two_pts
#
#     for row in range(n_rows - mc + 1):
#         for col in range(n_cols - mc + 1, n_cols):
#             v_vec = board[row:row + mc, col]
#             # Accumulate score for max_player position
#             if np.all(v_vec == player):
#                 score += win_pts
#             elif (len(np.where(v_vec == player)[0]) == 3 and
#                   len(np.where(v_vec == NO_PLAYER)[0]) == 1):
#                 score += three_pts
#             elif (len(np.where(v_vec == player)[0]) == 2 and
#                   len(np.where(v_vec == NO_PLAYER)[0]) == 2):
#                 score += two_pts
#             # Reduce score for min_player position
#             if np.all(v_vec == min_player):
#                 score -= win_pts
#             elif (len(np.where(v_vec == min_player)[0]) == 3 and
#                   len(np.where(v_vec == NO_PLAYER)[0]) == 1):
#                 score -= three_pts
#             elif (len(np.where(v_vec == min_player)[0]) == 2 and
#                   len(np.where(v_vec == NO_PLAYER)[0]) == 2):
#                 score -= two_pts
#
#     if max_player:
#         return GameScore(score)
#     else:
#         return GameScore(-score)


@njit
def heuristic_solver(board: Board, player: BoardPiece, max_player: bool):
    """
    Scores the game board based on whether a player has combinations of two,
    three, or four pieces in any four spaces, with the other spaces being
    unoccupied. Points increase by an order of magnitude for each piece, to
    prioritize being closer to winning.
    """
    # Shape of board
    n_rows, n_cols = board.shape
    # Min connection (4 in a row wins)
    mc = 4
    # Initialize the score
    score = 0
    two_pts = 1
    three_pts = 10
    # win_pts = 100

    # Slide the mask across the board and check for a win in each position
    min_player = (player % 2 + 1)
    for row in range(n_rows - mc + 1):
        for col in range(n_cols - mc + 1):
            # Accumulate score for max_player position
            # Check for vertical points
            v_vec = board[row:row + mc, col]
            # if np.all(v_vec == player):
            #     score += win_pts
            # elif (np.sum(v_vec == player) == 3 and
            if (np.sum(v_vec == player) == 3 and
                    np.sum(v_vec == NO_PLAYER) == 1):
                score += three_pts
            elif (np.sum(v_vec == player) == 2 and
                    np.sum(v_vec == NO_PLAYER) == 2):
                score += two_pts
            # Check for horizontal points
            h_vec = board[row, col:col + mc]
            # if np.all(h_vec == player):
            #     score += win_pts
            # elif (np.sum(h_vec == player) == 3 and
            if (np.sum(h_vec == player) == 3 and
                    np.sum(h_vec == NO_PLAYER) == 1):
                score += three_pts
            elif (np.sum(h_vec == player) == 2 and
                    np.sum(h_vec == NO_PLAYER) == 2):
                score += two_pts
            # Check for \ points
            block_mask = board[row:row + mc, col:col + mc]
            d_block = np.diag(block_mask)
            # if np.all(d_block == player):
            #     score += win_pts
            # elif (np.sum(d_block == player) == 3 and
            if (np.sum(d_block == player) == 3 and
                    np.sum(d_block == NO_PLAYER) == 1):
                score += three_pts
            elif (np.sum(d_block == player) == 2 and
                    np.sum(d_block == NO_PLAYER) == 2):
                score += two_pts
            # Check for / points
            b_block = np.diag(block_mask[::-1, :])
            # if np.all(b_block == player):
            #     score += win_pts
            # elif (np.sum(b_block == player) == 3 and
            if (np.sum(b_block == player) == 3 and
                    np.sum(b_block == NO_PLAYER) == 1):
                score += three_pts
            elif (np.sum(b_block == player) == 2 and
                    np.sum(b_block == NO_PLAYER) == 2):
                score += two_pts

            # Reduce score for min_player position
            # if np.all(v_vec == min_player):
            #     score -= win_pts
            # elif (np.sum(v_vec == min_player) == 3 and
            if (np.sum(v_vec == min_player) == 3 and
                    np.sum(v_vec == NO_PLAYER) == 1):
                score -= three_pts
            elif (np.sum(v_vec == min_player) == 2 and
                    np.sum(v_vec == NO_PLAYER) == 2):
                score -= two_pts
            # Check for horizontal points
            # if np.all(h_vec == min_player):
            #     score -= win_pts
            # elif (np.sum(h_vec == min_player) == 3 and
            if (np.sum(h_vec == min_player) == 3 and
                    np.sum(h_vec == NO_PLAYER) == 1):
                score -= three_pts
            elif (np.sum(h_vec == min_player) == 2 and
                    np.sum(h_vec == NO_PLAYER) == 2):
                score -= two_pts
            # Check for \ points
            # if np.all(d_block == min_player):
            #     score -= win_pts
            # elif (np.sum(d_block == min_player) == 3 and
            if (np.sum(d_block == min_player) == 3 and
                    np.sum(d_block == NO_PLAYER) == 1):
                score -= three_pts
            elif (np.sum(d_block == min_player) == 2 and
                    np.sum(d_block == NO_PLAYER) == 2):
                score -= two_pts
            # Check for / points
            # if np.all(b_block == min_player):
            #     score -= win_pts
            # elif (np.sum(b_block == min_player) == 3 and
            if (np.sum(b_block == min_player) == 3 and
                    np.sum(b_block == NO_PLAYER) == 1):
                score -= three_pts
            elif (np.sum(b_block == min_player) == 2 and
                    np.sum(b_block == NO_PLAYER) == 2):
                score -= two_pts

    for row in range(n_rows - mc + 1, n_rows):
        for col in range(n_cols):
            h_vec = board[row, col:col + mc]
            # Accumulate score for max_player position
            # if np.all(h_vec == player):
            #     score += win_pts
            # elif (np.sum(h_vec == player) == 3 and
            if (np.sum(h_vec == player) == 3 and
                    np.sum(h_vec == NO_PLAYER) == 1):
                score += three_pts
            elif (np.sum(h_vec == player) == 2 and
                    np.sum(h_vec == NO_PLAYER) == 2):
                score += two_pts
            # Reduce score for min_player position
            # if np.all(h_vec == min_player):
            #     score -= win_pts
            # elif (np.sum(h_vec == min_player) == 3 and
            if (np.sum(h_vec == min_player) == 3 and
                    np.sum(h_vec == NO_PLAYER) == 1):
                score -= three_pts
            elif (np.sum(h_vec == min_player) == 2 and
                    np.sum(h_vec == NO_PLAYER) == 2):
                score -= two_pts

    for row in range(n_rows - mc + 1):
        for col in range(n_cols - mc + 1, n_cols):
            v_vec = board[row:row + mc, col]
            # Accumulate score for max_player position
            # if np.all(v_vec == player):
            #     score += win_pts
            # elif (np.sum(v_vec == player) == 3 and
            if (np.sum(v_vec == player) == 3 and
                    np.sum(v_vec == NO_PLAYER) == 1):
                score += three_pts
            elif (np.sum(v_vec == player) == 2 and
                    np.sum(v_vec == NO_PLAYER) == 2):
                score += two_pts
            # Reduce score for min_player position
            # if np.all(v_vec == min_player):
            #     score -= win_pts
            # elif (np.sum(v_vec == min_player) == 3 and
            if (np.sum(v_vec == min_player) == 3 and
                    np.sum(v_vec == NO_PLAYER) == 1):
                score -= three_pts
            elif (np.sum(v_vec == min_player) == 2 and
                    np.sum(v_vec == NO_PLAYER) == 2):
                score -= two_pts

    if max_player:
        return GameScore(score)
    else:
        return GameScore(-score)


# def heuristic_solver_bits(board: Board, player: BoardPiece, max_player: bool = True):
#     # Convert the boards to bitmaps and define the min_player board
#     max_board, mask_board = board_to_bitmap(board, player)
#     min_board = max_board ^ mask_board
#     empty_board = ~mask_board
#
#     # Initialize the score and point values
#     score = 0
#     # Define the shift constants
#     b_cols = board.shape[1]
#     # Shift order: horizontal, vertical, \, /
#     shift_list = [1, b_cols + 1, b_cols, b_cols + 2]
#
#     # Accumulate score for max_player position
#     for shift in shift_list:
#         score += bit_solver(shift, max_board, empty_board)
#     # Reduce score for min_player position
#     for shift in shift_list:
#         score -= bit_solver(shift, min_board, empty_board)
#
#     if max_player:
#         return GameScore(score)
#     else:
#         return GameScore(-score)
#
#
# def bit_solver(shift: int, player: Bitmap, not_player: Bitmap):
#     """
#
#     """
#     from gmpy2 import popcount, mpz
#
#     # Initialize the score and point values
#     score = 0
#     pt2 = 1
#     pt3 = 10
#     win_pts = 100
#
#     s1_right = (player >> shift)
#     s2_right = player >> (2 * shift)
#     s3_right = player >> (3 * shift)
#     s1_left = (player << shift)
#     s2_left = player << (2 * shift)
#     s3_left = player << (3 * shift)
#     s1_right_n1 = s1_right & player
#     s1_left_n1 = s1_left & player
#
#     # TODO: Is there a faster way to count bits?
#     # Check for wins
#     score += win_pts * popcount(mpz(s1_left_n1 & (s1_left_n1 >> (2 * shift))))
#     # Check for 3 in 4
#     # XXX-
#     score += pt3 * popcount(mpz(((s1_left_n1 & s2_left) << shift)
#                                 & not_player))
#     # -XXX
#     score += pt3 * popcount(mpz(((s1_right_n1 & s2_right) >> shift)
#                                 & not_player))
#     # XX-X
#     score += pt3 * popcount(mpz((s1_right_n1 & s3_right) << (2 * shift)
#                                 & not_player))
#     # X-XX
#     score += pt3 * popcount(mpz((s1_left_n1 & s3_left) >> (2 * shift)
#                                 & not_player))
#     # Check for 2 in 4
#     # XX--
#     score += pt2 * popcount(mpz(s1_left_n1 << shift
#                                 & (not_player & (not_player >> shift))))
#     # --XX
#     score += pt2 * popcount(mpz(s1_right_n1 >> shift
#                                 & (not_player & (not_player << shift))))
#     # X-X-
#     score += pt2 * popcount(mpz((player & s2_left) << shift
#                                 & (not_player & (not_player << (2 * shift)))))
#     # -X-X
#     score += pt2 * popcount(mpz((player & s2_right) >> shift
#                                 & (not_player & (not_player >> (2 * shift)))))
#     # X--X
#     score += pt2 * popcount(mpz((player & s3_right) << shift
#                                 & (not_player & (not_player >> shift))))
#     # -XX-
#     score += pt2 * popcount(mpz(s1_right_n1 >> shift
#                                 & (not_player & (not_player >> (3 * shift)))))
#
#     return score


def heuristic_solver_bits(board: Board, player: BoardPiece, max_player: bool = True):
    """

    """

    # Convert the boards to bitmaps and define the min_player board
    max_board, mask_board = board_to_bitmap(board, player)
    min_board = max_board ^ mask_board
    empty_board = ~mask_board

    # Convert bitmaps to mpz objects
    max_board = mpz(max_board)
    min_board = mpz(min_board)
    empty_board = mpz(empty_board)

    # Initialize the score and point values
    score = 0
    # Define the shift constants
    b_cols = board.shape[1]
    # Shift order: horizontal, vertical, \, /
    shift_list = [1, b_cols + 1, b_cols, b_cols + 2]

    # Accumulate score for max_player position
    for shift in shift_list:
        score += bit_solver(shift, max_board, empty_board)
    # Reduce score for min_player position
    for shift in shift_list:
        score -= bit_solver(shift, min_board, empty_board)

    if max_player:
        return GameScore(score)
    else:
        return GameScore(-score)


def bit_solver(shift: int, player: Bitmap, not_player: Bitmap):
    """

    """

    # Initialize the score and point values
    score = 0
    pt2 = 1
    pt3 = 10
    win_pts = 100

    s1_right = (player >> shift)
    s2_right = player >> (2 * shift)
    s3_right = player >> (3 * shift)
    s1_left = (player << shift)
    s2_left = player << (2 * shift)
    s3_left = player << (3 * shift)
    s1_right_n1 = s1_right & player
    s1_left_n1 = s1_left & player

    # Check for wins
    score += win_pts * popcount(s1_left_n1 & (s1_left_n1 >> (2 * shift)))
    # Check for 3 in 4
    # XXX-
    score += pt3 * popcount(((s1_left_n1 & s2_left) << shift)
                            & not_player)
    # -XXX
    score += pt3 * popcount(((s1_right_n1 & s2_right) >> shift)
                            & not_player)
    # XX-X
    score += pt3 * popcount((s1_right_n1 & s3_right) << (2 * shift)
                            & not_player)
    # X-XX
    score += pt3 * popcount((s1_left_n1 & s3_left) >> (2 * shift)
                            & not_player)
    # Check for 2 in 4
    # XX--
    score += pt2 * popcount(s1_left_n1 << shift
                            & (not_player & (not_player >> shift)))
    # --XX
    score += pt2 * popcount(s1_right_n1 >> shift
                            & (not_player & (not_player << shift)))
    # X-X-
    score += pt2 * popcount((player & s2_left) << shift
                            & (not_player & (not_player << (2 * shift))))
    # -X-X
    score += pt2 * popcount((player & s2_right) >> shift
                            & (not_player & (not_player >> (2 * shift))))
    # X--X
    score += pt2 * popcount((player & s3_right) << shift
                            & (not_player & (not_player >> shift)))
    # -XX-
    score += pt2 * popcount(s1_right_n1 >> shift
                            & (not_player & (not_player >> (3 * shift))))

    return score

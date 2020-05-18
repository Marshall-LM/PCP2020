import numpy as np
from typing import Optional, Tuple
from agents.common import Board, BoardPiece, PlayerAction, SavedState,\
    NO_PLAYER, apply_player_action, connect_four
# from agents.common_bits import Bitmap, board_to_bitmap, connect_four
from agents.common_bits import Bitmap, board_to_bitmap

GameScore = np.int


def generate_move_alpha_beta(board: Board, player: BoardPiece,
                             saved_state: Optional[SavedState]
                             ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Agent selects a move based on a minimax depth first search


    """

    # If the board is empty, play in the center column
    if np.all(board == NO_PLAYER):
        action = np.floor(np.median(np.arange(board.shape[1])))
        return PlayerAction(action), saved_state

    board_cp = board.copy()
    # Call minimax
    # score, action = minimax(board_cp, player, True, 0)
    # Call alpha_beta
    alpha0 = -100000
    beta0 = 100000
    score, action = alpha_beta(board_cp, player, True, 0, alpha0, beta0)

    return PlayerAction(action), saved_state


def alpha_beta(board: Board, player: BoardPiece, max_player: bool,
               depth: int, alpha: GameScore, beta: GameScore
               ) -> Tuple[GameScore, Optional[PlayerAction]]:
    """

    """
    # Make a list of columns that can be played in
    potential_actions = np.argwhere(board[-1, :] == 0)
    potential_actions = potential_actions.reshape(potential_actions.size)

    # If the node is at the max depth, a terminal node, or is the root node
    # return the heursitic score of the node
    max_depth = 4
    # if depth == 0 or np.all(board != 0):
    if depth == max_depth or np.all(board != 0):
        return heuristic_solver(board, player, max_player), None
        # return heuristic_solver_bits(board, player, max_player), None

    # For each potential action, call alpha_beta
    if max_player:
        score = -np.inf
        for col in potential_actions:
            # Add a shortcut for blocking wins
            if (depth == 0 and connect_four(apply_player_action(board, col,
                                            BoardPiece(player % 2 + 1), True),
                                            BoardPiece(player % 2 + 1), col)):
                return GameScore(100), PlayerAction(col)

            # Apply the current action and call alpha_beta
            new_board = apply_player_action(board, col, player, True)
            new_score, temp = alpha_beta(new_board, BoardPiece(player % 2 + 1),
                                         False, depth + 1, alpha, beta)
            new_score -= 5 * depth
            # if depth == 0:
            #     print(new_score)
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
        score = np.inf
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


def heuristic_solver(board: Board, player: BoardPiece, max_player: bool):
    """

    """
    # Shape of board
    n_rows, n_cols = board.shape
    # Min connection (4 in a row wins)
    mc = 4
    # Initialize the score
    score = 0
    two_pts = 1
    three_pts = 10
    win_pts = 100

    # Slide the mask across the board and check for a win in each position
    min_player = (player % 2 + 1)
    for row in range(n_rows - mc + 1):
        for col in range(n_cols - mc + 1):
            # Accumulate score for max_player position
            # Check for vertical points
            v_vec = board[row:row + mc, col]
            if np.all(v_vec == player):
                score += win_pts
            elif (len(np.argwhere(v_vec == player)) == 3 and
                  len(np.argwhere(v_vec == NO_PLAYER)) == 1):
                score += three_pts
            elif (len(np.argwhere(v_vec == player)) == 2 and
                  len(np.argwhere(v_vec == NO_PLAYER)) == 2):
                score += two_pts
            # Check for horizontal points
            h_vec = board[row, col:col + mc]
            if np.all(h_vec == player):
                score += win_pts
            elif (len(np.argwhere(h_vec == player)) == 3 and
                  len(np.argwhere(h_vec == NO_PLAYER)) == 1):
                score += three_pts
            elif (len(np.argwhere(h_vec == player)) == 2 and
                  len(np.argwhere(h_vec == NO_PLAYER)) == 2):
                score += two_pts
            # Check for \ points
            block_mask = board[row:row + mc, col:col + mc]
            d_block = np.diag(block_mask)
            if np.all(d_block == player):
                score += win_pts
            elif (len(np.argwhere(d_block == player)) == 3 and
                  len(np.argwhere(d_block == NO_PLAYER)) == 1):
                score += three_pts
            elif (len(np.argwhere(d_block == player)) == 2 and
                  len(np.argwhere(d_block == NO_PLAYER)) == 2):
                score += two_pts
            # Check for / points
            b_block = np.diag(block_mask[::-1, :])
            if np.all(b_block == player):
                score += win_pts
            elif (len(np.argwhere(b_block == player)) == 3 and
                  len(np.argwhere(b_block == NO_PLAYER)) == 1):
                score += three_pts
            elif (len(np.argwhere(b_block == player)) == 2 and
                  len(np.argwhere(b_block == NO_PLAYER)) == 2):
                score += two_pts

            # Reduce score for min_player position
            if np.all(v_vec == min_player):
                score -= win_pts
            elif (len(np.argwhere(v_vec == min_player)) == 3 and
                  len(np.argwhere(v_vec == NO_PLAYER)) == 1):
                score -= three_pts
            elif (len(np.argwhere(v_vec == min_player)) == 2 and
                  len(np.argwhere(v_vec == NO_PLAYER)) == 2):
                score -= two_pts
            # Check for horizontal points
            if np.all(h_vec == min_player):
                score -= win_pts
            elif (len(np.argwhere(h_vec == min_player)) == 3 and
                  len(np.argwhere(h_vec == NO_PLAYER)) == 1):
                score -= three_pts
            elif (len(np.argwhere(h_vec == min_player)) == 2 and
                  len(np.argwhere(h_vec == NO_PLAYER)) == 2):
                score -= two_pts
            # Check for \ points
            if np.all(d_block == min_player):
                score -= win_pts
            elif (len(np.argwhere(d_block == min_player)) == 3 and
                  len(np.argwhere(d_block == NO_PLAYER)) == 1):
                score -= three_pts
            elif (len(np.argwhere(d_block == min_player)) == 2 and
                  len(np.argwhere(d_block == NO_PLAYER)) == 2):
                score -= two_pts
            # Check for / points
            if np.all(b_block == min_player):
                score -= win_pts
            elif (len(np.argwhere(b_block == min_player)) == 3 and
                  len(np.argwhere(b_block == NO_PLAYER)) == 1):
                score -= three_pts
            elif (len(np.argwhere(b_block == min_player)) == 2 and
                  len(np.argwhere(b_block == NO_PLAYER)) == 2):
                score -= two_pts

    for row in range(n_rows - mc + 1, n_rows):
        for col in range(n_cols):
            h_vec = board[row, col:col + mc]
            # Accumulate score for max_player position
            if np.all(h_vec == player):
                score += win_pts
            elif (len(np.argwhere(h_vec == player)) == 3 and
                  len(np.argwhere(h_vec == NO_PLAYER)) == 1):
                score += three_pts
            elif (len(np.argwhere(h_vec == player)) == 2 and
                  len(np.argwhere(h_vec == NO_PLAYER)) == 2):
                score += two_pts
            # Reduce score for min_player position
            if np.all(h_vec == min_player):
                score -= win_pts
            elif (len(np.argwhere(h_vec == min_player)) == 3 and
                  len(np.argwhere(h_vec == NO_PLAYER)) == 1):
                score -= three_pts
            elif (len(np.argwhere(h_vec == min_player)) == 2 and
                  len(np.argwhere(h_vec == NO_PLAYER)) == 2):
                score -= two_pts

    for row in range(n_rows - mc + 1):
        for col in range(n_cols - mc + 1, n_cols):
            v_vec = board[row:row + mc, col]
            # Accumulate score for max_player position
            if np.all(v_vec == player):
                score += win_pts
            elif (len(np.argwhere(v_vec == player)) == 3 and
                  len(np.argwhere(v_vec == NO_PLAYER)) == 1):
                score += three_pts
            elif (len(np.argwhere(v_vec == player)) == 2 and
                  len(np.argwhere(v_vec == NO_PLAYER)) == 2):
                score += two_pts
            # Reduce score for min_player position
            if np.all(v_vec == min_player):
                score -= win_pts
            elif (len(np.argwhere(v_vec == min_player)) == 3 and
                  len(np.argwhere(v_vec == NO_PLAYER)) == 1):
                score -= three_pts
            elif (len(np.argwhere(v_vec == min_player)) == 2 and
                  len(np.argwhere(v_vec == NO_PLAYER)) == 2):
                score -= two_pts

    if max_player:
        return GameScore(score)
    else:
        return GameScore(-score)


# def heuristic_solver(board: Board, player: BoardPiece, max_player: bool = True):
#
#     # Convert the boards to bitmaps and define the min_player board
#     max_board, mask_board = board_to_bitmap(board, player)
#     min_board = max_board ^ mask_board
#     not_max = ~max_board
#     not_min = ~min_board
#
#     # Initialize the score and point values
#     score = 0
#     pt2 = 1
#     pt3 = 10
#     win_pts = 100
#     # Define the shift constants
#     board_cols = board.shape[1]
#     h_shift = 1
#     v_shift = board_cols + 1
#     d_shift = board_cols
#     b_shift = board_cols + 2
#
#     # Accumulate score for max_player position
#     # Horizontal shifts
#     sh1_right = (max_board >> h_shift)
#     sh1_left = (max_board << h_shift)
#     sh1_right_n1 = sh1_right & max_board
#     sh1_left_n1 = sh1_left & max_board
#     sh2_left = max_board >> (2 * h_shift)
#     sh3_left = max_board >> (3 * h_shift)
#     sh2_right = max_board >> (2 * h_shift)
#     # Check for wins
#     # if sh1_left_n1 & (sh1_left_n1 >> (2 * h_shift)):
#     #     score += win_pts
#     score += win_pts * sh1_left_n1 & (sh1_left_n1 >> (2 * h_shift))
#     # Check for 3 in 4
#     # elif (
#     #     ((sh1_left_n1 << h_shift) & not_min) &
#     #     ((sh1_left_n1 << (2 * h_shift)) & max_board) or # XX-X
#     #     (((sh1_left_n1 << h_shift) & max_board) << h_shift) & not_min or # XXX-
#     #     ((sh1_right_n1 >> h_shift) & not_min) &
#     #     ((sh1_right_n1 >> (2 * h_shift)) & max_board) or  # X-XX
#     #     (((sh1_right_n1 >> h_shift) & max_board) >> h_shift) & not_min # -XXX
#     # ):
#     #     score += three_pts
#     score += pt3 * (((sh1_left_n1 << h_shift) & not_min)
#                     & ((sh1_left_n1 << (2 * h_shift)) & max_board))  # XX-X
#     score += pt3 * ((((sh1_left_n1 << h_shift) & max_board) << h_shift)
#                     & not_min)  # XXX-
#     score += pt3 * (((sh1_right_n1 >> h_shift) & not_min)
#                     & ((sh1_right_n1 >> (2 * h_shift)) & max_board))  # X-XX
#     score += pt3 * ((((sh1_right_n1 >> h_shift) & max_board) >> h_shift)
#                     & not_min)  # -XXX
#     # Check for 2 in 4
#     # elif(
#     #     (((sh1_left_n1 << h_shift) & not_min)
#     #      & ((sh1_left_n1 << (2 * h_shift)) & not_min)) or  # XX--
#     #     (((sh2_left & max_board) << h_shift) & not_min) or  # X-X-
#     #     (((sh1_right_n1 >> h_shift) & not_min)
#     #      & ((sh1_right_n1 >> (2 * h_shift)) & not_min)) or  # --XX
#     #     (((sh2_right & max_board) >> h_shift) & not_min) or  # -X-X
#     #     ((sh1_left & not_min) & (sh2_left & not_min) & (sh3_left & not_min))  # X--X
#     # ):
#     #     score += two_pts
#     score += pt2 * (((sh1_left_n1 << h_shift) & not_min)
#                     & ((sh1_left_n1 << (2 * h_shift)) & not_min))  # XX--
#     score += pt2 * (((sh2_left & max_board) << h_shift) & not_min)  # X-X-
#     score += pt2 * (((sh1_right_n1 >> h_shift) & not_min)
#                     & ((sh1_right_n1 >> (2 * h_shift)) & not_min))  # --XX
#     score += pt2 * (((sh2_right & max_board) >> h_shift) & not_min)  # -X-X
#     score += pt2 * ((sh1_left & not_min) & (sh2_left & not_min)
#                     & (sh3_left & not_min))  # X--X
#     # Diagonal \ check
#     m = max_board & (max_board >> d_shift)
#     if m & (m >> (2 * d_shift)):
#         return True
#     # Diagonal / check
#     m = max_board & (max_board >> b_shift)
#     if m & (m >> (2 * b_shift)):
#         return True
#     # Vertical check
#     m = max_board & (max_board >> v_shift)
#     if m & (m >> (2 * v_shift)):
#         return True
#     # Nothing found
#     return False
#
#     if max_player:
#         return GameScore(score)
#     else:
#         return GameScore(-score)

# def heuristic_solver(board: Board, player: BoardPiece, max_player: bool = True):
#     # Convert the boards to bitmaps and define the min_player board
#     max_board, mask_board = board_to_bitmap(board, player)
#     min_board = max_board ^ mask_board
#     not_max = ~max_board
#     not_min = ~min_board
#
#     # Initialize the score and point values
#     score = 0
#     pt2 = 1
#     pt3 = 10
#     win_pts = 100
#     # Define the shift constants
#     board_cols = board.shape[1]
#     h_shift = 1
#     v_shift = board_cols + 1
#     d_shift = board_cols
#     b_shift = board_cols + 2
#
#     # Accumulate score for max_player position
#     # Horizontal shifts
#     s1_right = (max_board >> h_shift)
#     s1_left = (max_board << h_shift)
#     s1_right_n1 = s1_right & max_board
#     s1_left_n1 = s1_left & max_board
#     s2_left = max_board >> (2 * h_shift)
#     s3_left = max_board >> (3 * h_shift)
#     s2_right = max_board >> (2 * h_shift)
#     # Check for wins
#     score += win_pts * s1_left_n1 & (s1_left_n1 >> (2 * h_shift))
#     # Check for 3 in 4
#     score += pt3 * (((s1_left_n1 << h_shift) & not_min)
#                     & ((s1_left_n1 << (2 * h_shift)) & max_board))  # XX-X
#     score += pt3 * ((((s1_left_n1 << h_shift) & max_board) << h_shift)
#                     & not_min)  # XXX-
#     score += pt3 * (((s1_right_n1 >> h_shift) & not_min)
#                     & ((s1_right_n1 >> (2 * h_shift)) & max_board))  # X-XX
#     score += pt3 * ((((s1_right_n1 >> h_shift) & max_board) >> h_shift)
#                     & not_min)  # -XXX
#     # Check for 2 in 4
#     score += pt2 * (((s1_left_n1 << h_shift) & not_min)
#                     & ((s1_left_n1 << (2 * h_shift)) & not_min))  # XX--
#     score += pt2 * (((s2_left & max_board) << h_shift) & not_min)  # X-X-
#     score += pt2 * (((s1_right_n1 >> h_shift) & not_min)
#                     & ((s1_right_n1 >> (2 * h_shift)) & not_min))  # --XX
#     score += pt2 * (((s2_right & max_board) >> h_shift) & not_min)  # -X-X
#     score += pt2 * ((s1_left & not_min) & (s2_left & not_min)
#                     & (s3_left & not_min))  # X--X
#
#     # Diagonal \ shifts
#     s1_right = (max_board >> d_shift)
#     s1_left = (max_board << d_shift)
#     s1_right_n1 = s1_right & max_board
#     s1_left_n1 = s1_left & max_board
#     s2_left = max_board >> (2 * d_shift)
#     s3_left = max_board >> (3 * d_shift)
#     s2_right = max_board >> (2 * d_shift)
#     # Check for wins
#     score += win_pts * s1_left_n1 & (s1_left_n1 >> (2 * d_shift))
#     # Check for 3 in 4
#     score += pt3 * (((s1_left_n1 << d_shift) & not_min)
#                     & ((s1_left_n1 << (2 * d_shift)) & max_board))  # XX-X
#     score += pt3 * ((((s1_left_n1 << d_shift) & max_board) << d_shift)
#                     & not_min)  # XXX-
#     score += pt3 * (((s1_right_n1 >> d_shift) & not_min)
#                     & ((s1_right_n1 >> (2 * d_shift)) & max_board))  # X-XX
#     score += pt3 * ((((s1_right_n1 >> d_shift) & max_board) >> d_shift)
#                     & not_min)  # -XXX
#     # Check for 2 in 4
#     score += pt2 * (((s1_left_n1 << d_shift) & not_min)
#                     & ((s1_left_n1 << (2 * d_shift)) & not_min))  # XX--
#     score += pt2 * (((s2_left & max_board) << d_shift) & not_min)  # X-X-
#     score += pt2 * (((s1_right_n1 >> d_shift) & not_min)
#                     & ((s1_right_n1 >> (2 * d_shift)) & not_min))  # --XX
#     score += pt2 * (((s2_right & max_board) >> d_shift) & not_min)  # -X-X
#     score += pt2 * ((s1_left & not_min) & (s2_left & not_min)
#                     & (s3_left & not_min))  # X--X
#
#     # Diagonal / shifts
#     s1_right = (max_board >> b_shift)
#     s1_left = (max_board << b_shift)
#     s1_right_n1 = s1_right & max_board
#     s1_left_n1 = s1_left & max_board
#     s2_left = max_board >> (2 * b_shift)
#     s3_left = max_board >> (3 * b_shift)
#     s2_right = max_board >> (2 * b_shift)
#     # Check for wins
#     score += win_pts * s1_left_n1 & (s1_left_n1 >> (2 * b_shift))
#     # Check for 3 in 4
#     score += pt3 * (((s1_left_n1 << b_shift) & not_min)
#                     & ((s1_left_n1 << (2 * b_shift)) & max_board))  # XX-X
#     score += pt3 * ((((s1_left_n1 << b_shift) & max_board) << b_shift)
#                     & not_min)  # XXX-
#     score += pt3 * (((s1_right_n1 >> b_shift) & not_min)
#                     & ((s1_right_n1 >> (2 * b_shift)) & max_board))  # X-XX
#     score += pt3 * ((((s1_right_n1 >> b_shift) & max_board) >> b_shift)
#                     & not_min)  # -XXX
#     # Check for 2 in 4
#     score += pt2 * (((s1_left_n1 << b_shift) & not_min)
#                     & ((s1_left_n1 << (2 * b_shift)) & not_min))  # XX--
#     score += pt2 * (((s2_left & max_board) << b_shift) & not_min)  # X-X-
#     score += pt2 * (((s1_right_n1 >> b_shift) & not_min)
#                     & ((s1_right_n1 >> (2 * b_shift)) & not_min))  # --XX
#     score += pt2 * (((s2_right & max_board) >> b_shift) & not_min)  # -X-X
#     score += pt2 * ((s1_left & not_min) & (s2_left & not_min)
#                     & (s3_left & not_min))  # X--X
#
#     # Vertical check
#     s1_right = (max_board >> v_shift)
#     s1_left = (max_board << v_shift)
#     s1_right_n1 = s1_right & max_board
#     s1_left_n1 = s1_left & max_board
#     s2_left = max_board >> (2 * v_shift)
#     s3_left = max_board >> (3 * v_shift)
#     s2_right = max_board >> (2 * v_shift)
#     # Check for wins
#     score += win_pts * s1_left_n1 & (s1_left_n1 >> (2 * v_shift))
#     # Check for 3 in 4
#     score += pt3 * (((s1_left_n1 << v_shift) & not_min)
#                     & ((s1_left_n1 << (2 * v_shift)) & max_board))  # XX-X
#     score += pt3 * ((((s1_left_n1 << v_shift) & max_board) << v_shift)
#                     & not_min)  # XXX-
#     score += pt3 * (((s1_right_n1 >> v_shift) & not_min)
#                     & ((s1_right_n1 >> (2 * v_shift)) & max_board))  # X-XX
#     score += pt3 * ((((s1_right_n1 >> v_shift) & max_board) >> v_shift)
#                     & not_min)  # -XXX
#     # Check for 2 in 4
#     score += pt2 * (((s1_left_n1 << v_shift) & not_min)
#                     & ((s1_left_n1 << (2 * v_shift)) & not_min))  # XX--
#     score += pt2 * (((s2_left & max_board) << v_shift) & not_min)  # X-X-
#     score += pt2 * (((s1_right_n1 >> v_shift) & not_min)
#                     & ((s1_right_n1 >> (2 * v_shift)) & not_min))  # --XX
#     score += pt2 * (((s2_right & max_board) >> v_shift) & not_min)  # -X-X
#     score += pt2 * ((s1_left & not_min) & (s2_left & not_min)
#                     & (s3_left & not_min))  # X--X
#
#     # Reduce score for min_player position
#     # Horizontal shifts
#     s1_right = (min_board >> h_shift)
#     s1_left = (min_board << h_shift)
#     s1_right_n1 = s1_right & min_board
#     s1_left_n1 = s1_left & min_board
#     s2_left = min_board >> (2 * h_shift)
#     s3_left = min_board >> (3 * h_shift)
#     s2_right = min_board >> (2 * h_shift)
#     # Check for wins
#     score -= win_pts * s1_left_n1 & (s1_left_n1 >> (2 * h_shift))
#     # Check for 3 in 4
#     score -= pt3 * (((s1_left_n1 << h_shift) & not_max)
#                     & ((s1_left_n1 << (2 * h_shift)) & min_board))  # XX-X
#     score -= pt3 * ((((s1_left_n1 << h_shift) & min_board) << h_shift)
#                     & not_max)  # XXX-
#     score -= pt3 * (((s1_right_n1 >> h_shift) & not_max)
#                     & ((s1_right_n1 >> (2 * h_shift)) & min_board))  # X-XX
#     score -= pt3 * ((((s1_right_n1 >> h_shift) & min_board) >> h_shift)
#                     & not_max)  # -XXX
#     # Check for 2 in 4
#     score -= pt2 * (((s1_left_n1 << h_shift) & not_max)
#                     & ((s1_left_n1 << (2 * h_shift)) & not_max))  # XX--
#     score -= pt2 * (((s2_left & min_board) << h_shift) & not_max)  # X-X-
#     score -= pt2 * (((s1_right_n1 >> h_shift) & not_max)
#                     & ((s1_right_n1 >> (2 * h_shift)) & not_max))  # --XX
#     score -= pt2 * (((s2_right & min_board) >> h_shift) & not_max)  # -X-X
#     score -= pt2 * ((s1_left & not_max) & (s2_left & not_max)
#                     & (s3_left & not_max))  # X--X
#
#     # Diagonal \ shifts
#     s1_right = (min_board >> d_shift)
#     s1_left = (min_board << d_shift)
#     s1_right_n1 = s1_right & min_board
#     s1_left_n1 = s1_left & min_board
#     s2_left = min_board >> (2 * d_shift)
#     s3_left = min_board >> (3 * d_shift)
#     s2_right = min_board >> (2 * d_shift)
#     # Check for wins
#     score -= win_pts * s1_left_n1 & (s1_left_n1 >> (2 * d_shift))
#     # Check for 3 in 4
#     score -= pt3 * (((s1_left_n1 << d_shift) & not_max)
#                     & ((s1_left_n1 << (2 * d_shift)) & min_board))  # XX-X
#     score -= pt3 * ((((s1_left_n1 << d_shift) & min_board) << d_shift)
#                     & not_max)  # XXX-
#     score -= pt3 * (((s1_right_n1 >> d_shift) & not_max)
#                     & ((s1_right_n1 >> (2 * d_shift)) & min_board))  # X-XX
#     score -= pt3 * ((((s1_right_n1 >> d_shift) & min_board) >> d_shift)
#                     & not_max)  # -XXX
#     # Check for 2 in 4
#     score -= pt2 * (((s1_left_n1 << d_shift) & not_max)
#                     & ((s1_left_n1 << (2 * d_shift)) & not_max))  # XX--
#     score -= pt2 * (((s2_left & min_board) << d_shift) & not_max)  # X-X-
#     score -= pt2 * (((s1_right_n1 >> d_shift) & not_max)
#                     & ((s1_right_n1 >> (2 * d_shift)) & not_max))  # --XX
#     score -= pt2 * (((s2_right & min_board) >> d_shift) & not_max)  # -X-X
#     score -= pt2 * ((s1_left & not_max) & (s2_left & not_max)
#                     & (s3_left & not_max))  # X--X
#
#     # Diagonal / shifts
#     s1_right = (min_board >> b_shift)
#     s1_left = (min_board << b_shift)
#     s1_right_n1 = s1_right & min_board
#     s1_left_n1 = s1_left & min_board
#     s2_left = min_board >> (2 * b_shift)
#     s3_left = min_board >> (3 * b_shift)
#     s2_right = min_board >> (2 * b_shift)
#     # Check for wins
#     score -= win_pts * s1_left_n1 & (s1_left_n1 >> (2 * b_shift))
#     # Check for 3 in 4
#     score -= pt3 * (((s1_left_n1 << b_shift) & not_max)
#                     & ((s1_left_n1 << (2 * b_shift)) & min_board))  # XX-X
#     score -= pt3 * ((((s1_left_n1 << b_shift) & min_board) << b_shift)
#                     & not_max)  # XXX-
#     score -= pt3 * (((s1_right_n1 >> b_shift) & not_max)
#                     & ((s1_right_n1 >> (2 * b_shift)) & min_board))  # X-XX
#     score -= pt3 * ((((s1_right_n1 >> b_shift) & min_board) >> b_shift)
#                     & not_max)  # -XXX
#     # Check for 2 in 4
#     score -= pt2 * (((s1_left_n1 << b_shift) & not_max)
#                     & ((s1_left_n1 << (2 * b_shift)) & not_max))  # XX--
#     score -= pt2 * (((s2_left & min_board) << b_shift) & not_max)  # X-X-
#     score -= pt2 * (((s1_right_n1 >> b_shift) & not_max)
#                     & ((s1_right_n1 >> (2 * b_shift)) & not_max))  # --XX
#     score -= pt2 * (((s2_right & min_board) >> b_shift) & not_max)  # -X-X
#     score -= pt2 * ((s1_left & not_max) & (s2_left & not_max)
#                     & (s3_left & not_max))  # X--X
#
#     # Vertical check
#     s1_right = (min_board >> v_shift)
#     s1_left = (min_board << v_shift)
#     s1_right_n1 = s1_right & min_board
#     s1_left_n1 = s1_left & min_board
#     s2_left = min_board >> (2 * v_shift)
#     s3_left = min_board >> (3 * v_shift)
#     s2_right = min_board >> (2 * v_shift)
#     # Check for wins
#     score -= win_pts * s1_left_n1 & (s1_left_n1 >> (2 * v_shift))
#     # Check for 3 in 4
#     score -= pt3 * (((s1_left_n1 << v_shift) & not_max)
#                     & ((s1_left_n1 << (2 * v_shift)) & min_board))  # XX-X
#     score -= pt3 * ((((s1_left_n1 << v_shift) & min_board) << v_shift)
#                     & not_max)  # XXX-
#     score -= pt3 * (((s1_right_n1 >> v_shift) & not_max)
#                     & ((s1_right_n1 >> (2 * v_shift)) & min_board))  # X-XX
#     score -= pt3 * ((((s1_right_n1 >> v_shift) & min_board) >> v_shift)
#                     & not_max)  # -XXX
#     # Check for 2 in 4
#     score -= pt2 * (((s1_left_n1 << v_shift) & not_max)
#                     & ((s1_left_n1 << (2 * v_shift)) & not_max))  # XX--
#     score -= pt2 * (((s2_left & min_board) << v_shift) & not_max)  # X-X-
#     score -= pt2 * (((s1_right_n1 >> v_shift) & not_max)
#                     & ((s1_right_n1 >> (2 * v_shift)) & not_max))  # --XX
#     score -= pt2 * (((s2_right & min_board) >> v_shift) & not_max)  # -X-X
#     score -= pt2 * ((s1_left & not_max) & (s2_left & not_max)
#                     & (s3_left & not_max))  # X--X
#
#     if max_player:
#         return GameScore(score)
#     else:
#         return GameScore(-score)


def heuristic_solver_bits(board: Board, player: BoardPiece, max_player: bool = True):
    # Convert the boards to bitmaps and define the min_player board
    max_board, mask_board = board_to_bitmap(board, player)
    min_board = max_board ^ mask_board
    not_max = ~max_board
    not_min = ~min_board

    # Initialize the score and point values
    score = 0
    # Define the shift constants
    b_cols = board.shape[1]
    shift_list = [1, b_cols + 1, b_cols, b_cols + 2]

    # Accumulate score for max_player position
    for shift in shift_list:
        score += bit_solver(shift, max_board, not_min)
    # Reduce score for min_player position
    for shift in shift_list:
        score -= bit_solver(shift, min_board, not_max)

    if max_player:
        return GameScore(score)
    else:
        return GameScore(-score)


def bit_solver(shift: int, board: Bitmap, not_board: Bitmap):
    """

    """

    # Initialize the score and point values
    score = 0
    pt2 = 1
    pt3 = 10
    win_pts = 100

    s1_right = (board >> shift)
    s1_left = (board << shift)
    s1_right_n1 = s1_right & board
    s1_left_n1 = s1_left & board
    s2_left = board >> (2 * shift)
    s3_left = board >> (3 * shift)
    s2_right = board >> (2 * shift)
    # Check for wins
    score += win_pts * s1_left_n1 & (s1_left_n1 >> (2 * shift))
    # Check for 3 in 4
    score += pt3 * (((s1_left_n1 << shift) & not_board)
                    & ((s1_left_n1 << (2 * shift)) & board))  # XX-X
    score += pt3 * ((((s1_left_n1 << shift) & board) << shift)
                    & not_board)  # XXX-
    score += pt3 * (((s1_right_n1 >> shift) & not_board)
                    & ((s1_right_n1 >> (2 * shift)) & board))  # X-XX
    score += pt3 * ((((s1_right_n1 >> shift) & board) >> shift)
                    & not_board)  # -XXX
    # Check for 2 in 4
    score += pt2 * (((s1_left_n1 << shift) & not_board)
                    & ((s1_left_n1 << (2 * shift)) & not_board))  # XX--
    score += pt2 * (((s2_left & board) << shift) & not_board)  # X-X-
    score += pt2 * (((s1_right_n1 >> shift) & not_board)
                    & ((s1_right_n1 >> (2 * shift)) & not_board))  # --XX
    score += pt2 * (((s2_right & board) >> shift) & not_board)  # -X-X
    score += pt2 * ((s1_left & not_board) & (s2_left & not_board)
                    & (s3_left & not_board))  # X--X

    return score

# TODO: Test bitmap calculations and tweak so that it's calculating the proper values.
#  It seems like the shortcut to block wins is not activating for some reason.
#  The scores being calculated are way too large.

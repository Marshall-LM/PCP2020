import numpy as np
from typing import Optional, Tuple
from agents.common import Board, BoardPiece, PlayerAction, SavedState,\
    NO_PLAYER, apply_player_action, connect_four
# from agents.common_bits import board_to_bitmap, connect_four

GameScore = np.int8


def generate_move_minimax(board: Board, player: BoardPiece,
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
    alpha0 = -1000
    beta0 = 1000
    score, action = alpha_beta(board_cp, player, True, 0, alpha0, beta0)

    return PlayerAction(action), saved_state


def minimax(board: Board, player: BoardPiece, max_player: bool,
            depth: int) -> Tuple[GameScore, Optional[PlayerAction]]:
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

    # For each potential action, call minimax
    if max_player:
        score = -np.inf
        for col in potential_actions:
            # Add a shortcut for blocking wins
            if (depth == 0 and connect_four(apply_player_action(board, col,
                                                                BoardPiece(player % 2 + 1), True),
                                            BoardPiece(player % 2 + 1), col)):
                return GameScore(200), PlayerAction(col)

            # Apply the current action and call alpha_beta
            new_board = apply_player_action(board, col, player, True)
            new_score, temp = minimax(new_board, BoardPiece(player % 2 + 1),
                                      False, depth + 1)
            new_score -= 5 * depth
            # Check whether the score updates
            if new_score > score:
                score = new_score
                action = col
        return GameScore(score), PlayerAction(action)
    else:
        score = np.inf
        for col in potential_actions:
            # Apply the current action and call alpha_beta
            new_board = apply_player_action(board, col, player, True)
            new_score, temp = minimax(new_board, BoardPiece(player % 2 + 1),
                                      True, depth + 1)
            new_score += 5 * depth
            # Check whether the score updates
            if new_score < score:
                score = new_score
                action = col
        return GameScore(score), PlayerAction(action)


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

    # For each potential action, call alpha_beta
    if max_player:
        score = -np.inf
        for col in potential_actions:
            # Add a shortcut for blocking wins
            if (depth == 0 and connect_four(apply_player_action(board, col,
                                            BoardPiece(player % 2 + 1), True),
                                            BoardPiece(player % 2 + 1), col)):
                return GameScore(200), PlayerAction(col)

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
    # # Set multiplier for min or max level
    # if max_player:
    #     multiplier = 1
    # else:
    #     multiplier = -1

    # Slide the mask across the board and check for a win in each position
    min_player = (player % 2 + 1)
    for row in range(n_rows - mc + 1):
        for col in range(n_cols - mc + 1):
            # Accumulate score for max_player position
            # Check for vertical points
            v_vec = board[row:row + mc, col]
            if np.all(v_vec == player):
                score += win_pts
                # return win_pts * multiplier
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
                # return win_pts * multiplier
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
                # return win_pts * multiplier
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
                # return win_pts * multiplier
            elif (len(np.argwhere(b_block == player)) == 3 and
                  len(np.argwhere(b_block == NO_PLAYER)) == 1):
                score += three_pts
            elif (len(np.argwhere(b_block == player)) == 2 and
                  len(np.argwhere(b_block == NO_PLAYER)) == 2):
                score += two_pts

            # Reduce score for min_player position
            if np.all(v_vec == min_player):
                score -= win_pts
                # return -win_pts * multiplier
            elif (len(np.argwhere(v_vec == min_player)) == 3 and
                  len(np.argwhere(v_vec == NO_PLAYER)) == 1):
                score -= three_pts
            elif (len(np.argwhere(v_vec == min_player)) == 2 and
                  len(np.argwhere(v_vec == NO_PLAYER)) == 2):
                score -= two_pts
            # Check for horizontal points
            if np.all(h_vec == min_player):
                score -= win_pts
                # return -win_pts * multiplier
            elif (len(np.argwhere(h_vec == min_player)) == 3 and
                  len(np.argwhere(h_vec == NO_PLAYER)) == 1):
                score -= three_pts
            elif (len(np.argwhere(h_vec == min_player)) == 2 and
                  len(np.argwhere(h_vec == NO_PLAYER)) == 2):
                score -= two_pts
            # Check for \ points
            if np.all(d_block == min_player):
                score -= win_pts
                # return -win_pts * multiplier
            elif (len(np.argwhere(d_block == min_player)) == 3 and
                  len(np.argwhere(d_block == NO_PLAYER)) == 1):
                score -= three_pts
            elif (len(np.argwhere(d_block == min_player)) == 2 and
                  len(np.argwhere(d_block == NO_PLAYER)) == 2):
                score -= two_pts
            # Check for / points
            if np.all(b_block == min_player):
                score -= win_pts
                # return -win_pts * multiplier
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
                # return win_pts * multiplier
            elif (len(np.argwhere(h_vec == player)) == 3 and
                  len(np.argwhere(h_vec == NO_PLAYER)) == 1):
                score += three_pts
            elif (len(np.argwhere(h_vec == player)) == 2 and
                  len(np.argwhere(h_vec == NO_PLAYER)) == 2):
                score += two_pts
            # Reduce score for min_player position
            if np.all(h_vec == min_player):
                score -= win_pts
                # return -win_pts * multiplier
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
                # return win_pts * multiplier
            elif (len(np.argwhere(v_vec == player)) == 3 and
                  len(np.argwhere(v_vec == NO_PLAYER)) == 1):
                score += three_pts
            elif (len(np.argwhere(v_vec == player)) == 2 and
                  len(np.argwhere(v_vec == NO_PLAYER)) == 2):
                score += two_pts
            # Reduce score for min_player position
            if np.all(v_vec == min_player):
                score -= win_pts
                # return -win_pts * multiplier
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
    # return score * multiplier


# def heuristic_solver(board: Board, player: BoardPiece, max_player: bool = True):
#     max_player_board, mask_board = board_to_bitmap(board, player)
#     min_player_board = max_player_board ^ mask_board
#     score = 0
#     if connect_four(max_player_board):
#         score += 10
#     if connect_four(min_player_board):
#         score -= 10
#
#     return score

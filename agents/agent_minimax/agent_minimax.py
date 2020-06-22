import numpy as np
from typing import Optional, Tuple
from agents.common_arrays import Board, BoardPiece, PlayerAction, SavedState,\
    NO_PLAYER, apply_player_action, connect_four

GameScore = np.int


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
    score, action = minimax(board_cp, player, True, 0)

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
        # return heuristic_solver_bits(board, player, max_player), None

    # For each potential action, call minimax
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

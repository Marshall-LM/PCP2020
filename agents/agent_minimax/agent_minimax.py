import numpy as np
from typing import Optional, Tuple
from agents.common import Board, BoardPiece, PlayerAction, SavedState,\
    NO_PLAYER, apply_player_action

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
    score, action = minimax(board_cp, player, True, 0)

    return PlayerAction(action), saved_state


def minimax(board: Board, player: BoardPiece, max_player: bool,
            depth: int) -> Tuple[GameScore, Optional[PlayerAction]]:
    # if depth = 0 or node is a terminal node then
    # return the heuristic value of node
    #
    # if maximizingPlayer then
    #     value := −∞
    #     for each child of node do
    #         value := max(value, minimax(child, depth − 1, FALSE))
    #     return value
    # else (*minimizing player *)
    #     value := +∞
    #     for each child of node do
    #         value := min(value, minimax(child, depth − 1, TRUE))
    #     return value

    # Make a list of columns that can be played in
    potential_actions = np.argwhere(board[-1, :] == 0)
    # As a default, choose the action closest to the center
    action = PlayerAction(np.floor(np.median(potential_actions)))

    # If the node is at the max depth, a terminal node, or is the root node
    # return the heursitic score of the node
    max_depth = 4
    if depth == max_depth or np.all(board != 0):
    # if depth == 0 or np.all(board != 0):
        return heuristic_solver(board, player, max_player), None

    # For each potential action, call minimax
    if max_player:
        score = -np.inf
        for col in potential_actions:
            new_board = apply_player_action(board, col, player, True)
            new_score, new_action = minimax(new_board, BoardPiece(player % 2 + 1),
                                            False, depth + 1)
            new_score -= depth
            if new_score > score:
                score = new_score
                action = col
        return GameScore(score), PlayerAction(action)
    else:
        score = np.inf
        for col in potential_actions:
            new_board = apply_player_action(board, col, player, True)
            new_score, new_action = minimax(new_board, BoardPiece(player % 2 + 1),
                                            True, depth + 1)
            new_score += depth
            if new_score < score:
                score = new_score
                action = col
        return GameScore(score), PlayerAction(action)

# (*Initial call *)
# minimax(origin, depth, TRUE)


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
    three_pts = 2
    win_pts = 5

    # Slide the mask across the board and check for a win in each position
    for row in range(n_rows):
        for col in range(n_cols):
            # Check for vertical wins
            if (row + mc) <= n_rows:
                v_vec = board[row:row + mc, col]
                if np.all(v_vec == player):
                    score += win_pts
                elif (len(np.argwhere(v_vec == player)) == 3 and
                      len(np.argwhere(v_vec == NO_PLAYER)) == 1):
                    score += three_pts
                elif (len(np.argwhere(v_vec == player)) == 2 and
                      len(np.argwhere(v_vec == NO_PLAYER)) == 2):
                    score += two_pts
            # Check for horizontal wins
            if (col + mc) <= n_cols:
                h_vec = board[row, col:col + mc]
                if np.all(h_vec == player):
                    score += win_pts
                elif (len(np.argwhere(h_vec == player)) == 3 and
                      len(np.argwhere(h_vec == NO_PLAYER)) == 1):
                    score += three_pts
                elif (len(np.argwhere(h_vec == player)) == 2 and
                      len(np.argwhere(h_vec == NO_PLAYER)) == 2):
                    score += two_pts
            if ((col + mc) <= n_cols) and ((row + mc) <= n_rows):
                # Check for \ wins
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
                # Check for / wins
                b_block = np.diag(block_mask[::-1, :])
                if np.all(b_block == player):
                    score += win_pts
                elif (len(np.argwhere(b_block == player)) == 3 and
                      len(np.argwhere(b_block == NO_PLAYER)) == 1):
                    score += three_pts
                elif (len(np.argwhere(b_block == player)) == 2 and
                      len(np.argwhere(b_block == NO_PLAYER)) == 2):
                    score += two_pts

    # score = np.random.randint(-20, 20)
    if max_player:
        return GameScore(score)
    else:
        return GameScore(-score)

import numpy as np
from typing import Optional, Tuple
from agents.common import Board, BoardPiece, PlayerAction, SavedState,\
    NO_PLAYER, apply_player_action, connect_four

GameScore = np.int


def generate_move_alpha_beta(board: Board, player: BoardPiece,
                             saved_state: Optional[SavedState]
                             ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Agent selects a move based on a minimax depth first search, with
    alpha-beta pruning.

    :param board: 2d array representing current state of the game
    :param player: the player who made the last move (active player)

    :return: the agent's selected move
    """

    # If the board is empty, play in the center column
    if np.all(board == NO_PLAYER):
        action = np.floor(np.median(np.arange(board.shape[1])))
        return PlayerAction(action), saved_state

    board_cp = board.copy()
    # Call alpha_beta
    alpha0 = -100000
    beta0 = 100000
    score, action = alpha_beta(board_cp, player, True, 0, alpha0, beta0)

    return PlayerAction(action), saved_state


def alpha_beta(board: Board, player: BoardPiece, max_player: bool,
               depth: int, alpha: GameScore, beta: GameScore
               ) -> Tuple[GameScore, Optional[PlayerAction]]:
    """
    Recursively call alpha_beta to build a game tree to a pre-determined
    max depth. Once at the max depth, or at a terminal node, calculate and
    return the heuristic score. Scores farther down the tree are penalized.

    Shortcuts are built in to:
    1. Automatically take a win
    2. Automatically block a loss
    3. Return a large score for a win at any depth

    :param board: 2d array representing current state of the game
    :param player: the player who made the last move (active player)
    :param max_player: boolean indicating whether the depth at which alpha_beta
                       is called from is a maximizing or minimizing player
    :param depth: the current depth in the game tree
    :param alpha: the currently best score for the maximizing player along the
                  path to root
    :param beta: the currently best score for the minimizing player along the
                  path to root

    :return: the best action and the associated score
    """
    # Make a list of columns that can be played in
    potential_actions = np.argwhere(board[-1, :] == 0)
    potential_actions = potential_actions.reshape(potential_actions.size)

    # If the node is at the max depth, a terminal node calculate the score
    max_depth = 8
    if depth == max_depth or np.all(board != 0):
        return GameScore(0), None
        # num_pcs = len(np.where(board != NO_PLAYER)[0])
        # if connect_four(board, player):
        #     return GameScore(22 - num_pcs), None

    # If this is the root call, check for wins and block/win, prioritize wins
    win_score = 150
    if depth == 0:
        for col in potential_actions:
            if connect_four(apply_player_action(board, col, player, True),
                            player, col):
                return GameScore(win_score), PlayerAction(col)
        for col in potential_actions:
            if connect_four(apply_player_action(board, col,
                            BoardPiece(player % 2 + 1), True),
                            BoardPiece(player % 2 + 1), col):
                return GameScore(win_score), PlayerAction(col)

    # For each potential action, call alpha_beta
    num_pcs = len(np.where(board != NO_PLAYER)[0])
    if max_player:
        score = -np.inf
        for col in potential_actions:
            # Apply the current action and call alpha_beta
            new_board = apply_player_action(board, col, player, True)
            if connect_four(new_board, player):
                return GameScore(22 - num_pcs), col

            new_score, temp = alpha_beta(new_board, BoardPiece(player % 2 + 1),
                                         False, depth + 1, alpha, beta)
            # new_score -= 5 * depth
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
            if connect_four(new_board, player):
                return GameScore(num_pcs - 22), col

            new_score, temp = alpha_beta(new_board, BoardPiece(player % 2 + 1),
                                         True, depth + 1, alpha, beta)
            # new_score += 5 * depth
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

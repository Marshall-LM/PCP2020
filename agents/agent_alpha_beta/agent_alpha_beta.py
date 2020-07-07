import numpy as np
from typing import Optional, Tuple
from gmpy2 import popcount
from agents.common import Board, BoardPiece, Bitmap, PlayerAction, \
    SavedState, NO_PLAYER, GameState, board_to_bitmap, check_end_state, \
    apply_action_cp

GameScore = np.int


def generate_move_alpha_beta(board: Board, player: BoardPiece,
                             saved_state: Optional[SavedState]
                             ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Agent selects a move based on a minimax depth first search, with
    alpha-beta pruning.

    :param board: 2d array representing current state of the game
    :param player: the player who made the last move (active player)
    :param saved_state: ???

    :return: the agent's selected move
    """

    # If the board is empty, play in the center column
    if np.all(board == NO_PLAYER):
        action = np.floor(np.median(np.arange(board.shape[1])))
        return PlayerAction(action), saved_state

    # Convert the board to bitmaps and define the min_player board
    max_board, mask_board = board_to_bitmap(board, player)

    # Call alpha_beta
    alpha0 = -100000
    beta0 = 100000
    score, action = alpha_beta(max_board, mask_board, True, 0, alpha0, beta0,
                               board.shape)

    return PlayerAction(action), saved_state


def alpha_beta(board: Bitmap, mask: Bitmap, max_player: bool, depth: int,
               alpha: GameScore, beta: GameScore, board_shp: Tuple
               ) -> Tuple[GameScore, Optional[PlayerAction]]:
    """
    Recursively call alpha_beta to build a game tree to a pre-determined
    max depth. Once at the max depth, or at a terminal node, calculate and
    return the heuristic score. Scores farther down the tree are penalized.

    :param board: bitmap representing positions of current player
    :param mask: bitmap representing positions of both players
    :param max_player: boolean indicating whether the depth at which alpha_beta
                       is called from is a maximizing or minimizing player
    :param depth: the current depth in the game tree
    :param alpha: the currently best score for the maximizing player along the
                  path to root
    :param beta: the currently best score for the minimizing player along the
                  path to root
    :param board_shp: the shape of the game board

    :return: the best action and the associated score
    """

    # If the node is at the max depth or a terminal node calculate the score
    max_depth = 7
    win_score = 150
    state_p = check_end_state(board, mask, board_shp)
    if state_p == GameState.IS_WIN:
        if max_player:
            return GameScore(win_score), None
        else:
            return GameScore(-win_score), None
    elif state_p == GameState.IS_DRAW:
        return 0, None
    elif depth == max_depth:
        return heuristic_solver_bits(board, mask, board_shp[0], max_player), None

    # For each potential action, call alpha_beta
    if max_player:
        score = -100000
        action = -1
        for col in range(board_shp[1]):
            # Apply the current action, continue if column is full
            try:
                min_board, new_mask = apply_action_cp(board, mask,
                                                      col, board_shp)
            except IndexError:
                continue
            # Call alpha-beta
            new_score, temp = alpha_beta(min_board, new_mask, False, depth + 1,
                                         alpha, beta, board_shp)
            new_score -= depth
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
        action = -1
        for col in range(board_shp[1]):
            # Apply the current action, continue if column is full
            try:
                max_board, new_mask = apply_action_cp(board, mask,
                                                      col, board_shp)
            except IndexError:
                continue
            # Call alpha-beta
            new_score, temp = alpha_beta(max_board, new_mask, True, depth + 1,
                                         alpha, beta, board_shp)
            new_score += depth
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


def heuristic_solver_bits(player_board: Bitmap, mask_board: Bitmap,
                          board_rows: int, max_player: bool = True
                          ) -> GameScore:
    """
    Calculates the heuristic score based on whether a player has combinations
    of two or three in any four connected spaces, with the other spaces being
    unoccupied. Points increase by an order of magnitude for each piece, to
    prioritize being closer to winning.

    :param player_board: bitmap representing positions of current player
    :param mask_board: bitmap representing positions of both players
    :param board_rows: the number of rows in the game board
    :param max_player: boolean indicating whether the depth at which alpha_beta
                       is called from is a maximizing or minimizing player

    :return: the score associated with this particular board configuration
    """

    # Initialize the score and point values
    score = 0
    # Define the shift constants
    # Shift order: vertical, horizontal, /, \
    shift_list = [1, board_rows + 1, board_rows + 2, board_rows]

    # Accumulate score for max_player position
    for shift in shift_list:
        score += bit_solver(shift, player_board, ~mask_board)
    # Reduce score for min_player position
    for shift in shift_list:
        score -= bit_solver(shift, (player_board ^ mask_board), ~mask_board)

    if max_player:
        return GameScore(score)
    else:
        return GameScore(-score)


def bit_solver(shift: int, player: Bitmap, not_player: Bitmap):
    """
    Calculates the heuristic score for a particular orientation of pieces
    (i.e. vertically aligned, horizontally aligned, right diagonal or left
    diagonal). Each direction has a unique shift value in the bitmap.

    :param shift: bit shift of the current direction being calculated
    :param player: bitmap representing positions of the current player
    :param not_player: bitmap representing positions of the other player

    :return: the score associated with this particular direction
    """

    # Initialize the score and point values
    score = 0
    pt2 = 1
    pt3 = 10

    s1_right = (player >> shift)
    s2_right = player >> (2 * shift)
    s3_right = player >> (3 * shift)
    s1_left = (player << shift)
    s2_left = player << (2 * shift)
    s3_left = player << (3 * shift)
    s1_right_n1 = s1_right & player
    s1_left_n1 = s1_left & player

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

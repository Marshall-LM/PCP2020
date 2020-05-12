import numpy as np
from typing import Optional, Tuple
from agents.common import Board, BoardPiece, PlayerAction, SavedState


def generate_move_random(board: Board, player: BoardPiece,
                         saved_state: Optional[SavedState]
                         ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """ Choose a valid, non-full column randomly and return it as action

    :param board: 2d array representing current state of the game
    :param player: the player making the current move (active player)
    :param saved_state: ???
    :return: returns a tuple containing the randomly generated move, and the
             saved state
    """

    free_cols = np.arange(board.shape[1])[np.argwhere(board[-1, :] == 0)]
    free_cols = free_cols.reshape(len(free_cols))
    action = PlayerAction(np.random.choice(free_cols))

    return action, saved_state

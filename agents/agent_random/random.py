import numpy as np
from typing import Optional, Tuple
from agents.common import top_row
from agents.common import BoardPiece, PlayerAction, SavedState


def generate_move_random(board: np.ndarray, player: BoardPiece,
                         saved_state: Optional[SavedState]
                         ) -> Tuple[PlayerAction, Optional[SavedState]]:
    """ Choose a valid, non-full column randomly and return it as `action

    :param board: 2d array representing current state of the game
    :param player: the player making the current move (active player)
    :param saved_state: ???
    :return: returns a tuple containing the randomly generated move, and the
             saved state
    """

    invalid_action = True
    action = PlayerAction(0)

    while invalid_action:
        action = PlayerAction(np.random.choice(board.shape[1]))
        try:
            top_row(board, action)
            invalid_action = False
        except IndexError:
            continue

    return action, saved_state

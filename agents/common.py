import numpy as np
from enum import Enum
from typing import Optional, Callable, Tuple


# Initialize data types
Board = np.ndarray
BoardPiece = np.int8
PlayerAction = np.int8

# Initialize constant variables
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece


class GameState(Enum):
    """
    Define the possible game states for reference. Make GameState an Enum so
    that it can be iterated through, reducing the chance of errors.
    """

    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


class SavedState:
    """ Can be used to store any extra info I want stored """
    pass


# This provides the type hints for the generate_move function family
GenMove = Callable[[Board, BoardPiece, Optional[SavedState]],
                   Tuple[PlayerAction, Optional[SavedState]]]


def initialize_game_state() -> Board:
    """ Initializes a connect four game board

    :return: ndarray, shape (6, 7) and data type (dtype) BoardPiece,
             initialized to 0 (NO_PLAYER).
    """

    return np.ones((6, 7), dtype=BoardPiece) * NO_PLAYER


def pretty_print_board(board: Board) -> str:
    """
    Converts the game board into a string that can be printed to
    neatly display the current game state. When printed, the piece
    at board[0, 0] should be in the lower-left position.

    :param board: 2d array representing current state of the game
    :return: string representing the current state of the game
    """

    # Define board shape
    bd_shp = board.shape

    # Top is full row of underscores
    length_row = 3 * bd_shp[0] + 3
    board_str = '_' * length_row + '\n'

    # For each row in the board
    for row in range(bd_shp[0] - 1, -1, -1):
        board_str += '|'

        # For each column in the board
        # Iterate in reverse order so that the bottom left corner is (0,0)
        for col in range(bd_shp[1]):
            curr_pos = board[row, col]
            # If the value is zero, position is empty
            if curr_pos == 0:
                board_str += '   '
            # If the value is one, position belongs to player 1 (X)
            elif curr_pos == 1:
                board_str += 'X  '
            # If the value is two, position belongs to player 2 (O)
            elif curr_pos == 2:
                board_str += 'O  '
            else:
                print('The board contains an invalid entry')

        board_str = board_str[:-2] + '|\n'

    # Add a separating row
    board_str += '|' + (length_row-2)*'=' + '|\n'
    # Add a row that shows the column numbers
    columns = np.arange(bd_shp[1])
    board_str += '|'
    for col in columns[:-1]:
        board_str += str(col) + '  '

    # Join
    # '  '.join(list(range()))
    board_str += str(columns[-1]) + '|\n'

    return board_str


def string_to_board(pp_board: str) -> Board:
    """
    Converts a string representing the game state to an ndarray game board

    :param pp_board: string representing the current state of the game
    :return: 2d array representing current state of the game
    """

    # Initialize an empty board
    board_arr = initialize_game_state()
    # Initialize the column and row positions
    pos = 0
    col = 0
    row = board_arr.shape[0]

    row_entries = False
    for char in pp_board:
        if char == '_' or char == '=':
            continue

        if char == '|':
            row_entries = not row_entries
            col = 0
            pos = 0
        elif char == '\n':
            row -= 1

        # Break after all rows have been checked
        if row < 0:
            break

        if row_entries:
            if char == 'X':
                board_arr[row, col] = 1
            elif char == 'O':
                board_arr[row, col] = 2

            pos += 1
            col = pos // 3

    return board_arr


def apply_player_action(board: Board, action: PlayerAction,
                        player: BoardPiece, copy_: bool = False) -> Board:
    """
    Sets board[i, action] = player, where i is the lowest open row. The
    modified board is returned. If copy is True, makes a copy of the board
    before modifying it. If the player's move is invalid column, throw an
    error.

    :param board: 2d array representing current state of the game
    :param action: the column the current player played their piece in
    :param player: the player making the current move (active player)
    :param copy_: boolean indicating whether to copy board before modifying
                  if copy_ is false, modify board in place

    :return: returns a 2d array representing the updated board game state,
             if the player makes an invalid selection, raises an IndexError
    """

    # Either copy the board
    if copy_:
        board_copy = board.copy()
    # Or set board_copy as a pointer to board (to modify in place)
    else:
        board_copy = board

    board_copy[top_row(board_copy, action), action] = player

    return board_copy


def connect_four(board: Board, player: BoardPiece,
                 last_action: Optional[PlayerAction] = None) -> bool:
    """
    Identify whether the current state of the board results in a win
    for the given player

    :param board: 2d array representing current state of the game
    :param player: the player who made the last move (active player)
    :param last_action: the column the last piece was played in

    :return: True if the player who just played has four adjacent pieces,
             False otherwise
    """

    # Shape of board
    n_rows, n_cols = board.shape
    # Min connection (4 in a row wins)
    mc = 4

    # Improve computation speed by using last_action
    if last_action is None:
        col_upper = n_cols
        col_lower = 0
    else:
        col_upper = min(n_cols, last_action + mc - 1)
        col_lower = max(0, last_action - mc + 1)
    col_upper = (mc if col_upper < mc else col_upper)

    # Slide the mask across the board and check for a win in each position
    for row in range(n_rows - mc + 1):
        for col in range(col_lower, col_upper - mc + 1):
            # Check for vertical wins
            v_vec = board[row:row + mc, col]
            if np.all(v_vec == player):
                return True
            # Check for horizontal wins
            h_vec = board[row, col:col + mc]
            if np.all(h_vec == player):
                return True
            # Check for \ wins
            block = board[row:row + mc, col:col + mc]
            if np.all(np.diag(block) == player):
                return True
            # Check for / wins
            if np.all(np.diag(block[::-1, :]) == player):
                return True

    for row in range(n_rows - mc + 1, n_rows):
        for col in range(col_lower, col_upper - mc + 1):
            h_vec = board[row, col:col + mc]
            if np.all(h_vec == player):
                return True

    for row in range(n_rows - mc + 1):
        for col in range(col_upper - mc + 1, col_upper):
            v_vec = board[row:row + mc, col]
            if np.all(v_vec == player):
                return True


def check_end_state(board: Board, player: BoardPiece,
                    last_action: Optional[PlayerAction] = None) -> GameState:
    """
    Returns the current game state for the active player, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?

    :param board: 2d array representing current state of the game
    :param player: the player who made the last move (active player)
    :param last_action: the column the last piece was played in

    :return: GameState class constant indicating new state of game
    """

    # If connect_four returns True, the active player won
    if connect_four(board, player, last_action):
        return GameState.IS_WIN
    # If the game is not won, and there are no empty spots, the game is a draw
    elif np.all(board != 0):
        return GameState.IS_DRAW
    # If the game is neither won, nor drawn, continue playing
    else:
        return GameState.STILL_PLAYING


def top_row(board: Board, col: PlayerAction):
    """
    Returns the highest row containing a board piece for the given column.
    Used to update the game state and determine the top row that must be
    checked to determine a win.

    :param board: 2d array representing current state of the game
    :param col: the column the last piece was played in

    :return: returns the index of the highest row containing a board piece
    """

    play_col = board[:, col]
    play_col = play_col.reshape(len(play_col))
    if play_col[-1] != 0:
        raise IndexError('This column is full')
    else:
        return np.min(np.argwhere(play_col == 0))


def switch_player(player: BoardPiece):
    """ Takes a player argument, and returns the other player

    :param player: the active player
    :return: the inactive player
    """
    return BoardPiece(player % 2 + 1)

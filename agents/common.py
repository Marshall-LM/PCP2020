import numpy as np
from enum import Enum
from typing import Optional, Callable, Tuple


# Initialize data types
Board = np.ndarray
BoardPiece = np.int8
PlayerAction = np.int8  # The column to be played
Bitmap = np.int

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
    pass


# TODO: What does this do?
GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


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
            # TODO: USE THIS IF PRINTING 0s IN EMPTY SPACES
            # board_str += str(board[row,col]) + '  '

            # TODO: USE THIS BLOCK IF PRINTING SPACES IN EMPTY SPACES
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
    board_str += '|'
    for col in range(bd_shp[1] - 1):
        board_str += str(col) + '  '

    # Join
    # '  '.join(list(range()))
    board_str += str(col + 1) + '|\n'

    return board_str


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

    :return: returns a 2d array representing the updated board game state,
             if the player makes an invalid selection, raises an IndexError
    """

    # TODO: If the copy_ boolean is set, create a copy of the board
    #  It's useful in the context of minimax
    if copy_:
        board_copy = board.copy()

    # THIS MIGHT BE TAKEN CARE OF IN USER_MOVE
    # # If the column does not exist, raise an IndexError
    # board_cols = np.arange(board_copy.shape[1])
    # try:
    #     board_cols[action] = action
    # except IndexError:
    #     raise IndexError('You have selected an invalid column to play in. '
    #                      'Please choose again')

    # Play in the chosen column. If the column is full, raise an IndexError
    try:
        board_copy[top_row(board_copy, action), action] = player
    # except IndexError:
    #     raise IndexError('This column is full. Please choose again')
    except:
        print('This column is full. Please choose again')

    return board_copy


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

    # TODO: USE THIS IS PRINTING 0s IN EMPTY SPACES
    # For each character in pp_board, check whether it is a digit.
    # If it is, store it. If it is a new line character, start a new row.
    # for char in pp_board:
    #     if char.isdigit():
    #         board_arr[row,col] = np.int8(char)
    #         col += 1
    #     elif char == '\n':
    #         col = 0
    #         row -= 1
    #         # row += 1 # If we ignore orientation
    #
    #     # Break after all rows have been checked
    #     if row < 0:
    #     # if row >= board_arr.shape[0]:
    #         break

    # TODO: USE THIS BLOCK IF PRINTING SPACES IN EMPTY BOARD LOCATIONS
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

    # Set win boolean, default to no win
    win = False
    # Shape of board
    bd_shp = board.shape
    # Mask dimension (4 in a row wins)
    md = 4

    # Initialize a 4x4 matrix mask
    row_mask_init = np.repeat(np.arange(md), md).reshape(md, md)
    col_mask_init = np.repeat(np.arange(md), md).reshape(md, md).T

    # Improve computation speed by using last_action
    if last_action is None:
        col_upper = bd_shp[1]
        col_lower = 0
        row_top = bd_shp[0]
    else:
        col_upper = min(bd_shp[1], last_action + md - 1)
        col_lower = max(0, last_action - md + 1)
        row_top = min(bd_shp[0], top_row(board, last_action) + md - 1)

    # Slide the mask across the board and check for a win in each position
    # I could speed this up, but
    for i in range(row_top + 1 - md):
        row_mask = row_mask_init + i
        for j in range(col_lower, col_upper + 1 - md):
            col_mask = col_mask_init + j

            # Use mask to select 4x4 subset of board
            board_sub = board[row_mask, col_mask]

            # Check the diagonals for a win
            diag_vec = np.diagonal(board_sub)
            rev_diag_vec = [board_sub[md - 1 - i][i] for i in range(md - 1, -1, -1)]
            if ((np.all(diag_vec) and diag_vec[0] == player) or
                    (np.all(rev_diag_vec) and rev_diag_vec[0] == player)):
                win = True
                break

            # Check the rows and columns for a win
            for k in range(md):
                if (np.all(np.argwhere(board_sub[:, k] == player)) or
                        np.all(np.argwhere(board_sub[k, :] == player))):
                    win = True
                    break

    return win


def connect_four_bits(board_map: Bitmap, player: BoardPiece,
                      last_action: Optional[PlayerAction] = None) -> bool:
    """
    Identify whether the current bitmap of the board results in a win
    for the whom it belongs to.

    :param board_map: bitmap representing the state of a player's pieces
    :param player: the player who made the last move (active player)
    :param last_action: the column the last piece was played in

    :return: True if the player who just played has four adjacent pieces,
             False otherwise
    """

    # TODO: decide whether to use player and last_action args based on
    #  how the game is set up in main

    # Horizontal check
    m = board_map & (board_map >> 1)
    if m & (m >> 2):
        return True
    # Diagonal \ check
    m = board_map & (board_map >> 7)
    if m & (m >> 14):
        return True
    # Diagonal / check
    m = board_map & (board_map >> 9)
    if m & (m >> 18):
        return True
    # Vertical check
    m = board_map & (board_map >> 8)
    if m & (m >> 16):
        return True
    # Nothing found
    return False


def board_to_bitmap(board: Board, player: BoardPiece) -> [Bitmap, Bitmap]:
    """
    Converts the nd.array board into a bitmap for faster calculations. Bitmap used
    to improve computation speed so that agent can train faster.

    :param board: 2d array representing current state of the game
    :param player: the player who made the last move (active player)

    :return: two bitmaps representing active player's positions and the positions
             containing pieces of either player (mask)
    """

    # Initialize the bitmaps as strings (converted to int in return)
    position, mask = '', ''
    bd_shp = board.shape

    # Start with top row
    for row in range(bd_shp[0] - 1, -1, -1):
        # Add 0-bits to sentinel column to avoid rollover errors
        mask += '0'
        position += '0'

        # Start with right column
        for col in range(bd_shp[1] - 1, -1, -1):
            # if board[row, col] != 0:
            #     print('board[%i, %i] = %i' % (row, col, board[row, col]))
            mask += ('1' if board[row, col] != NO_PLAYER else '0')
            position += ('1' if board[row, col] == player else '0')

    return Bitmap(position, 2), Bitmap(mask, 2)


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
    # if connect_four(board, player, last_action):
    if connect_four(board, player):
        new_state = GameState.IS_WIN
    # If the game is not won, and there are no empty spots, the game is a draw
    elif np.all(board != 0):
        new_state = GameState.IS_DRAW
    # If using bit masks
    # elif (empty_bit_mask and board) > 0:
    #     new_state = GameState.IS_DRAW
    # If the game is neither won, nor drawn, continue playing
    else:
        new_state = GameState.STILL_PLAYING

    return new_state


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
    if play_col[-1] != 0:
        raise

    if np.argwhere(play_col == 0).size == 0:
        raise IndexError('This column is full')
    else:
        return min(np.argwhere(play_col == 0)[0])

    # top_row_open = min(np.argwhere(board[:, col] == 0))[0]
    # if not top_row_open:
    #     raise
    # return min(np.argwhere(board[:, col] == 0))[0]
    # try:
    #     return min(np.argwhere(board[:, col] == 0))[0]
    # except ValueError:
    #     return -1
    #     raise IndexError


# TODO: Check whether bitmap implementation actually improves computation
#  speed for the overall program, or whether converting back and forth
#  between a bitmap and array ends up slowing it down too much.

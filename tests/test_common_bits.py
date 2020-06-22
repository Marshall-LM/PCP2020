import numpy as np
import pytest
import agents.common_bits as cm
from agents.common_arrays import connect_four as cf_loop
from gmpy2 import popcount, bit_flip


def test_initialize_game_state():
    test_board = cm.initialize_game_state()

    assert isinstance(test_board, np.ndarray)
    assert test_board.dtype == np.int8
    assert test_board.shape == (6, 7)
    assert np.all(test_board == cm.NO_PLAYER)
    return


def test_pretty_print_board():
    test_board = cm.initialize_game_state()
    test_board = np.random.randint(0, 3, test_board.shape)
    print('')
    print(test_board)

    board_str = cm.pretty_print_board(test_board)
    print('')
    print(board_str)

    assert isinstance(board_str, str)
    return


def test_string_to_board():
    test_board = cm.initialize_game_state()
    test_board = np.random.randint(0, 3, test_board.shape)
    print('')
    print(test_board)
    board_str = cm.pretty_print_board(test_board)
    print('')
    print(board_str)
    board_arr = cm.string_to_board(board_str)
    print('')
    print(board_arr)

    assert isinstance(test_board, np.ndarray)
    assert board_arr.dtype == np.int8
    assert board_arr.shape == test_board.shape
    return


def test_board_to_bitmap():
    test_board = cm.initialize_game_state()
    test_board[0, 2] = cm.PLAYER1
    test_board[0, 3] = cm.PLAYER2
    test_board[0, 4] = cm.PLAYER2
    test_board[0, 5] = cm.PLAYER1
    test_board[1, 2] = cm.PLAYER1
    test_board[1, 3] = cm.PLAYER1
    test_board[1, 4] = cm.PLAYER2
    test_board[2, 2] = cm.PLAYER1
    test_board[2, 3] = cm.PLAYER2
    test_board[3, 2] = cm.PLAYER2
    test_board[3, 3] = cm.PLAYER2
    test_board[4, 3] = cm.PLAYER1

    board_str = cm.pretty_print_board(test_board)
    print('')
    print(board_str)

    # Indices count from bottom left, counting up rows then switching to the
    # next column (this example from blog)
    mask_pos = int('0000000000000100000110011111000111100000000000000', 2)
    # p1_posit = int('0000000000000100000000010010000011100000000000000', 2)
    p2_posit = int('0000000000000000000110001101000100000000000000000', 2)

    # print('')
    # print('Compare mask')
    # print(bin(mask_pos))
    # print(bin(cm.board_to_bitmap(test_board, cm.PLAYER2)[1]))
    # print('Compare P2 position')
    # print(bin(p2_posit))
    # print(bin(cm.board_to_bitmap(test_board, cm.PLAYER2)[0]))
    # print('Compare P1 position')
    # print(bin(mask_pos ^ p2_posit))
    # print(bin(cm.board_to_bitmap(test_board, cm.PLAYER1)[0]))

    assert cm.board_to_bitmap(test_board, cm.PLAYER2)[0] == p2_posit
    assert cm.board_to_bitmap(test_board, cm.PLAYER2)[1] == mask_pos
    assert cm.board_to_bitmap(test_board, cm.PLAYER1)[0] == \
        (mask_pos ^ p2_posit)
    return


def generate_full_board(arr_board, player, empty_spaces=0):
    # Convert board to bitmap
    bit_board, bit_mask = cm.board_to_bitmap(arr_board, player)
    # Calculate the board shape
    bd_shp = arr_board.shape

    # While the board is not full, continue placing pieces
    while popcount(bit_mask) != bd_shp[0] * bd_shp[1] - empty_spaces:
        # Select a random move in a column that is not full
        move = -1
        while not (0 <= move < bd_shp[1]):
            move = np.random.choice(bd_shp[1])
            try:
                move = cm.PlayerAction(move)
                cm.top_row(arr_board, move)
            except IndexError:
                move = -1

        # Apply the move to both boards
        cm.apply_player_action(arr_board, move, player)
        bit_board, bit_mask = cm.apply_player_action_cp(bit_board, bit_mask,
                                                        move, bd_shp[0])
        # Switch to the next player
        player = cm.BoardPiece(player % 2 + 1)

    return arr_board, bit_board, bit_mask, player


def test_apply_player_action():
    # Set the first player as P1
    player = cm.PLAYER1
    # Initialize an empty board and make a bitmap copy
    arr_board = cm.initialize_game_state()
    # Generate a full board
    arr_board, bit_b, bit_m, player = generate_full_board(arr_board, player)
    # Check that both boards are the same
    check_board, check_mask = cm.board_to_bitmap(arr_board, player)
    assert check_board == bit_b
    assert check_mask == bit_m

    # Check that playing in a full board raises an exception
    for i in range(100):
        with pytest.raises(IndexError):
            move = np.random.choice(arr_board.shape[1])
            cm.apply_player_action_cp(bit_b, bit_m, move, arr_board.shape[0])

    # Print the board
    board_str = cm.pretty_print_board(arr_board)
    print('')
    print(board_str)
    return


def test_connect_four_bits():
    # Load the test case file
    lines = open('Test_cases_wins')

    # Each line is a set of moves
    for line in lines:
        # Set the first player as P1
        player = cm.PLAYER1
        # Initialize an empty board and make a bitmap copy
        arr_board = cm.initialize_game_state()
        bit_b, bit_m = cm.board_to_bitmap(arr_board, player)
        # Calculate the board shape
        bd_shp = arr_board.shape

        # Each character is a move
        for char in line:
            if char == '\n':
                break
            # Apply the move
            move = cm.PlayerAction(char)
            bit_b, bit_m = cm.apply_player_action_cp(bit_b, bit_m,
                                                     move, bd_shp[0])
            cm.apply_player_action(arr_board, move, player)
            # Switch the player
            player = cm.BoardPiece(player % 2 + 1)
        # Print the board for visual check
        # board_str = cm.pretty_print_board(arr_board)
        # print('')
        # print(board_str)
        # Check for a win
        assert cm.connect_four((bit_b ^ bit_m), bd_shp[0])

    # Generate many random boards and check with both connect_four functions
    n_boards = 100
    for i in range(n_boards):
        # Set the first player as P1
        player = cm.PLAYER1
        # Initialize an empty board and make a bitmap copy
        arr_b = cm.initialize_game_state()
        # Generate a full board
        arr_b, bit_b, bit_m, player = generate_full_board(arr_b, player)
        # Test both connect_four functions and for both players
        assert cf_loop(arr_b, player) == cm.connect_four(bit_b, arr_b.shape[0])
        player = cm.BoardPiece(player % 2 + 1)
        assert cf_loop(arr_b, player) == cm.connect_four((bit_b ^ bit_m),
                                                         arr_b.shape[0])
    return


def test_check_end_state():
    # Generate many random boards and check the GameState of each
    n_boards = 10000
    # Set counters
    n_wins = 0
    n_draws = 0
    n_continues = 0
    for i in range(n_boards):
        # Set the first player as P1
        player = cm.PLAYER1
        # Initialize an empty board and make a bitmap copy
        arr_b = cm.initialize_game_state()
        # Generate a full board
        arr_b, bit_b, bit_m, player = generate_full_board(arr_b, player)
        # Set a variable to determine whether a bit is flipped
        p_flip = 0.3
        # Check for wins
        if cm.connect_four(bit_b, arr_b.shape[0]):
            assert cm.check_end_state(bit_b, bit_m, arr_b.shape) == \
                cm.GameState.IS_WIN
            n_wins += 1
        elif cm.connect_four((bit_b ^ bit_m), arr_b.shape[0]):
            bit_bn = (bit_b ^ bit_m)
            assert cm.check_end_state(bit_bn, bit_m, arr_b.shape) == \
                cm.GameState.IS_WIN
            n_wins += 1
        elif np.random.uniform() < p_flip:
            bit_m = bit_flip(bit_m, 1)
            assert cm.check_end_state(bit_b, bit_m, arr_b.shape) == \
                cm.GameState.STILL_PLAYING
            n_continues += 1
        else:
            assert cm.check_end_state(bit_b, bit_m, arr_b.shape) == \
                cm.GameState.IS_DRAW
            n_draws += 1
            # Print the board for visual check
            board_str = cm.pretty_print_board(arr_b)
            print('')
            print(board_str)

    print('Number of wins = {}\nNumber of draws = {}\nNumber of '
          'continuations = {}'.format(n_wins, n_draws, n_continues))
    return


def test_top_row():
    # Generate many random boards and check the top row of a random column
    n_boards = 500
    for i in range(n_boards):
        # Set the first player as P1
        player = cm.PLAYER1
        # Initialize an empty board and make a bitmap copy
        arr_b = cm.initialize_game_state()
        # Generate a full board
        arr_b, bit_b, bit_m, player = generate_full_board(arr_b, player)
        # Set a variable to determine whether a bit is flipped
        p_flip = 0.5
        # Randomly choose a column to flip
        fcol = np.random.choice(arr_b.shape[1])
        for col in range(arr_b.shape[1]):
            if col == fcol:
                if np.random.uniform() < p_flip:
                    n_rows = arr_b.shape[0]
                    n_remove = np.random.choice(np.arange(1, n_rows))
                    arr_b[-n_remove:, col] = cm.BoardPiece(np.zeros(n_remove))
                    assert cm.top_row(arr_b, col) == n_rows - n_remove
                else:
                    with pytest.raises(IndexError):
                        cm.top_row(arr_b, col)
            else:
                with pytest.raises(IndexError):
                    cm.top_row(arr_b, col)


def test_check_top_row():
    # Generate many random boards and check the top row of a random column
    n_boards = 500
    for i in range(n_boards):
        # Set the first player as P1
        player = cm.PLAYER1
        # Initialize an empty board and make a bitmap copy
        arr_b = cm.initialize_game_state()
        # Generate a full board
        arr_b, bit_b, bit_m, player = generate_full_board(arr_b, player)
        # Set a variable to determine whether a bit is flipped
        p_flip = 0.5
        # Randomly choose a column to flip
        fcol = np.random.choice(arr_b.shape[1])
        for col in range(arr_b.shape[1]):
            if col == fcol:
                if np.random.uniform() < p_flip:
                    bit_pos = fcol * arr_b.shape[1] + arr_b.shape[0] - 1
                    bit_m = bit_flip(bit_m, bit_pos)
                    assert not cm.check_top_row(bit_m, col, arr_b.shape)
                else:
                    assert cm.check_top_row(bit_m, col, arr_b.shape)
            else:
                assert cm.check_top_row(bit_m, col, arr_b.shape)

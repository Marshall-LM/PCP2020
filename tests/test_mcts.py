import numpy as np
import agents.common as cm
import agents.agent_mcts.agent_mcts_random as agmcts
from gmpy2 import popcount


def generate_full_board(player, empty_spaces=0):
    # Generate an empty board
    arr_board = cm.initialize_game_state()
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
        cm.apply_action(arr_board, move, player)
        bit_board, bit_mask = cm.apply_action_cp(bit_board, bit_mask,
                                                 move, bd_shp)
        # Switch to the next player
        player = cm.BoardPiece(player % 2 + 1)

    return arr_board, bit_board, bit_mask, player


def test_node_initialization():
    player = cm.PLAYER1
    arr_bd, bit_bd, mask_bd, player = generate_full_board(player, 2)
    print(cm.pretty_print_board(arr_bd))
    bd_shp = arr_bd.shape
    empty_col = cm.valid_actions(mask_bd, bd_shp)[0]
    print(empty_col)
    test_node = agmcts.Connect4Node(bit_bd, mask_bd, bd_shp, empty_col, True)
    print(test_node.actions)
    # assert test_node.actions[0] == empty_col

    while cm.check_end_state(bit_bd, mask_bd, bd_shp) == cm.GameState.IS_WIN:
        arr_bd, bit_bd, mask_bd, player = generate_full_board(player, 2)
    print(cm.pretty_print_board(arr_bd))
    move = agmcts.generate_move_mcts(arr_bd, player, None)[0]
    print(move)

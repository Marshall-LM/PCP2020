import numpy as np
import agents.common as cm
import agents.agent_minimax.agent_minimax as agm

def test_heuristic_solver():
    # test_board = cm.initialize_game_state()
    # test_board[0, 1:4] = cm.PLAYER1
    # test_board[1, 3] = cm.PLAYER2
    # board_str = cm.pretty_print_board(test_board)
    # print('')
    # print(board_str)
    # assert agm.heuristic_solver(test_board, cm.PLAYER1, True) == 5
    #
    # test_board = cm.initialize_game_state()
    # test_board[0, 2:5] = cm.PLAYER1
    # test_board[1, 2:5] = cm.PLAYER2
    # board_str = cm.pretty_print_board(test_board)
    # print('')
    # print(board_str)
    # assert agm.heuristic_solver(test_board, cm.PLAYER2, False) == -6

    test_board = cm.initialize_game_state()
    test_board[0, 2:4] = cm.PLAYER1
    test_board[1, 2] = cm.PLAYER1
    test_board[0, 1] = cm.PLAYER2
    test_board[2, 2] = cm.PLAYER2
    test_board[1, 3] = cm.PLAYER2
    board_str = cm.pretty_print_board(test_board)
    print('')
    print(board_str)
    assert agm.heuristic_solver(test_board, cm.PLAYER1, True) == 2
    assert agm.heuristic_solver(test_board, cm.PLAYER2, False) == 2
    test_board_cp = test_board.copy()
    assert agm.minimax(test_board_cp, cm.PLAYER1, True, 0) == (10, 4)

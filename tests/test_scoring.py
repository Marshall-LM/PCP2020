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

    # test_board = cm.initialize_game_state()
    # test_board[0, 2:4] = cm.PLAYER1
    # test_board[1, 2] = cm.PLAYER1
    # test_board[0, 1] = cm.PLAYER2
    # test_board[2, 2] = cm.PLAYER2
    # test_board[1, 3] = cm.PLAYER2
    # board_str = cm.pretty_print_board(test_board)
    # print('')
    # print(board_str)
    # assert agm.heuristic_solver(test_board, cm.PLAYER1, True) == 0
    # test_board_cp = test_board.copy()
    # assert agm.minimax(test_board_cp, cm.PLAYER1, True, 0)[1] == 4

    # test_board = cm.initialize_game_state()
    # test_board[0, 3:6] = cm.PLAYER1
    # test_board[1, 3] = cm.PLAYER2
    # test_board[0, 2] = cm.PLAYER2
    # board_str = cm.pretty_print_board(test_board)
    # print('')
    # print(board_str)
    # assert agm.heuristic_solver(test_board, cm.PLAYER2, True) == -3
    # test_board_cp = test_board.copy()
    # assert agm.minimax(test_board_cp, cm.PLAYER2, True, 0)[1] == 6

    # test_board = cm.initialize_game_state()
    # test_board[:3, 1] = cm.PLAYER1
    # test_board[0, 3] = cm.PLAYER1
    # test_board[3, 1] = cm.PLAYER2
    # test_board[0, 2] = cm.PLAYER2
    # test_board[1, 3] = cm.PLAYER2
    # test_board[0, 4] = cm.PLAYER2
    # board_str = cm.pretty_print_board(test_board)
    # print('')
    # print(board_str)
    # assert agm.heuristic_solver(test_board, cm.PLAYER2, True) == 5
    # test_board_cp = test_board.copy()
    # assert not agm.minimax(test_board_cp, cm.PLAYER2, True, 0)[1] == 2

    # test_board = cm.initialize_game_state()
    # moves = np.array([3, 1, 2, 2, 5, 4, 4, 1, 2, 4, 2, 4, 2, 2, 3])
    # player = cm.PLAYER1
    # for mv in moves:
    #     cm.apply_player_action(test_board, mv, player, False)
    #     player = cm.switch_player(player)
    #
    # board_str = cm.pretty_print_board(test_board)
    # print('')
    # print(board_str)
    # test_board_cp = test_board.copy()
    # assert agm.minimax(test_board_cp, cm.PLAYER2, True, 0)[1] == 3

    test_board = cm.initialize_game_state()
    moves = np.array([3, 1, 4, 5, 3, 3, 4, 5, 4, 4, 3, 5, 5, 5, 4, 6, 3, 3,
                      5, 0, 1, 1, 1, 1, 1, 0, 4, 6, 0, 6])
    player = cm.PLAYER1
    for mv in moves:
        cm.apply_player_action(test_board, mv, player, False)
        player = cm.switch_player(player)

    board_str = cm.pretty_print_board(test_board)
    print('')
    print(board_str)
    test_board_cp = test_board.copy()
    assert agm.minimax(test_board_cp, cm.PLAYER1, True, 0)[1] == 6

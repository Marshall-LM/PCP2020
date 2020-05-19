import numpy as np
import agents.common as cm
# import agents.agent_minimax.agent_minimax as agm
# import agents.agent_alpha_beta.agent_alpha_beta as agm
# import agents.agent_alpha_beta.alpha_beta_experiment as agm
import agents.agent_alpha_beta.agent_alpha_beta_22 as agm


def test_heuristic_solver():
    max_depth = 0
    # test_board = cm.initialize_game_state()
    # test_board[0, 1:4] = cm.PLAYER1
    # test_board[1, 3] = cm.PLAYER2
    # board_str = cm.pretty_print_board(test_board)
    # print('')
    # print(board_str)
    # assert (agm.heuristic_solver_bits(test_board, cm.PLAYER1, True) ==
    #         agm.heuristic_solver(test_board, cm.PLAYER1, True))
    #
    # test_board = cm.initialize_game_state()
    # test_board[0, 2:5] = cm.PLAYER1
    # test_board[1, 2:5] = cm.PLAYER2
    # board_str = cm.pretty_print_board(test_board)
    # print('')
    # print(board_str)
    # assert (agm.heuristic_solver_bits(test_board, cm.PLAYER2, False) ==
    #         agm.heuristic_solver(test_board, cm.PLAYER1, True))

    test_board = cm.initialize_game_state()
    test_board[0, 2:4] = cm.PLAYER1
    test_board[1, 2] = cm.PLAYER1
    test_board[0, 1] = cm.PLAYER2
    test_board[2, 2] = cm.PLAYER2
    test_board[1, 3] = cm.PLAYER2
    board_str = cm.pretty_print_board(test_board)
    print('')
    print(board_str)
    # assert agm.heuristic_solver(test_board, cm.PLAYER1, True) == 0
    test_board_cp = test_board.copy()
    assert agm.alpha_beta(test_board_cp, cm.PLAYER1, True, max_depth,
                          -1000, 1000)[1] == 4

    test_board = cm.initialize_game_state()
    test_board[0, 3:6] = cm.PLAYER1
    test_board[1, 3] = cm.PLAYER2
    test_board[0, 2] = cm.PLAYER2
    board_str = cm.pretty_print_board(test_board)
    print('')
    print(board_str)
    # assert agm.heuristic_solver(test_board, cm.PLAYER2, True) == -3
    test_board_cp = test_board.copy()
    assert agm.alpha_beta(test_board_cp, cm.PLAYER2, True, max_depth,
                          -1000, 1000)[1] == 6

    test_board = cm.initialize_game_state()
    test_board[:3, 1] = cm.PLAYER1
    test_board[0, 3] = cm.PLAYER1
    test_board[3, 1] = cm.PLAYER2
    test_board[0, 2] = cm.PLAYER2
    test_board[1, 3] = cm.PLAYER2
    test_board[0, 4] = cm.PLAYER2
    board_str = cm.pretty_print_board(test_board)
    print('')
    print(board_str)
    # assert agm.heuristic_solver(test_board, cm.PLAYER2, True) == 5
    test_board_cp = test_board.copy()
    assert not agm.alpha_beta(test_board_cp, cm.PLAYER2, True, max_depth,
                              -1000, 1000)[1] == 2

    test_board = cm.initialize_game_state()
    moves = np.array([3, 1, 2, 2, 5, 4, 4, 1, 2, 4, 2, 4, 2, 2, 3])
    player = cm.PLAYER1
    for mv in moves:
        cm.apply_player_action(test_board, mv, player, False)
        player = cm.switch_player(player)

    board_str = cm.pretty_print_board(test_board)
    print('')
    print(board_str)
    test_board_cp = test_board.copy()
    assert agm.alpha_beta(test_board_cp, cm.PLAYER2, True, max_depth,
                          -1000, 1000)[1] == 3

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
    assert agm.alpha_beta(test_board_cp, cm.PLAYER1, True, max_depth,
                          -1000, 1000)[1] == 6

    test_board = cm.initialize_game_state()
    moves = np.array([3, 1, 2, 3, 2, 4, 2, 2, 3, 3, 2, 3, 2, 0, 3, 0, 0,
                      5, 0, 6, 0, 0, 6, 5, 5, 5, 5, 5])
    player = cm.PLAYER1
    for mv in moves:
        cm.apply_player_action(test_board, mv, player, False)
        player = cm.switch_player(player)

    board_str = cm.pretty_print_board(test_board)
    print('')
    print(board_str)
    test_board_cp = test_board.copy()
    assert agm.alpha_beta(test_board_cp, cm.PLAYER1, True, max_depth,
                          -1000, 1000)[1] == 6

    test_board = cm.initialize_game_state()
    moves = np.array([3, 3, 2, 4, 0, 1, 0, 3, 0, 0, 2, 3, 3, 2, 2, 2, 0, 6,
                      2, 5, 6, 5, 5, 5, 3, 5])
    player = cm.PLAYER1
    for mv in moves:
        cm.apply_player_action(test_board, mv, player, False)
        player = cm.switch_player(player)

    board_str = cm.pretty_print_board(test_board)
    print('')
    print(board_str)
    test_board_cp = test_board.copy()
    assert agm.alpha_beta(test_board_cp, cm.PLAYER1, True, max_depth,
                          -1000, 1000)[1] != 4

    # test_board = cm.initialize_game_state()
    # # test_board[0, 2:5] = cm.PLAYER1
    # test_board[0, 2:4] = cm.PLAYER1
    # test_board[0, 1] = cm.PLAYER2
    # test_board[0, 4] = cm.PLAYER2
    # test_board[0, 5] = cm.PLAYER1
    # # test_board[0, 5] = cm.PLAYER1
    # # test_board[0, 2] = cm.PLAYER1
    # board_str = cm.pretty_print_board(test_board)
    # print('')
    # print(board_str)
    # test_board_cp = test_board.copy()
    # print(agm.heuristic_solver_bits(test_board, cm.PLAYER1, True))
    # print(agm.heuristic_solver(test_board, cm.PLAYER1, True))
    # # print(agm.alpha_beta(test_board_cp, cm.PLAYER1, True, max_depth,
    # #                      -1000, 1000, True)[1])
    # # assert agm.heuristic_solver_bits(test_board, cm.PLAYER2, False) == -6
    # # assert agm.alpha_beta(test_board_cp, cm.PLAYER1, True, max_depth,
    # #                       -1000, 1000, True)[1] == 6

    # test_board = cm.initialize_game_state()
    # moves = np.array([3, 3, 2, 0, 4, 6])
    # player = cm.PLAYER1
    # for mv in moves:
    #     cm.apply_player_action(test_board, mv, player, False)
    #     player = cm.switch_player(player)
    # board_str = cm.pretty_print_board(test_board)
    # print('')
    # print(board_str)
    # test_board_cp = test_board.copy()
    # # print(agm.heuristic_solver_bits(test_board, cm.PLAYER1, True))
    # # print(agm.alpha_beta(test_board_cp, cm.PLAYER1, True, max_depth,
    # #                      -1000, 1000, True)[1])
    # assert (agm.heuristic_solver_bits(test_board, cm.PLAYER2, True) ==
    #         agm.heuristic_solver(test_board, cm.PLAYER2, True))
    # assert agm.alpha_beta(test_board_cp, cm.PLAYER1, True, max_depth,
    #                       -1000, 1000, True)[1] == 5

    # test_board = cm.initialize_game_state()
    # moves = np.array([3, 3, 3, 2, 2, 2, 2, 4, 1, 3, 4, 1, 0, 5])
    # player = cm.PLAYER1
    # for mv in moves:
    #     cm.apply_player_action(test_board, mv, player, False)
    #     player = cm.switch_player(player)
    # board_str = cm.pretty_print_board(test_board)
    # print('')
    # print(board_str)
    # test_board_cp = test_board.copy()
    # print(agm.heuristic_solver_bits(test_board, cm.PLAYER1, True))
    # print(agm.alpha_beta(test_board_cp, cm.PLAYER1, True, max_depth,
    #                      -1000, 1000, True)[1])
    # # assert (agm.heuristic_solver_bits(test_board, cm.PLAYER2, True) ==
    # #         agm.heuristic_solver(test_board, cm.PLAYER2, True))
    # assert agm.alpha_beta(test_board_cp, cm.PLAYER1, True, max_depth,
    #                       -1000, 1000, True)[1] != 1


# It didn't take the win
# 3, 1, 4, 5, 5, 4, 4, 6, 2, 3, 3, 2, 1, 2, 3, 5, 3, 3, 4, 5, 5, 4, 6

import numpy as np
from numba import njit
from timeit import timeit
from agents.common import initialize_game_state, switch_player, \
    apply_player_action, PLAYER1
# from agents.agent_alpha_beta.agent_alpha_beta import heuristic_solver, \
#     heuristic_solver_bits
from agents.agent_alpha_beta.alpha_beta_experiment import heuristic_solver, \
    heuristic_solver_bits
from main import human_vs_agent
from agents.agent_alpha_beta import generate_move
import cProfile


# Can disable njit
# import os
# os.environ['NUMBA_DISABLE_NJIT'] = '1'


# Compare numba compiled loop and
def test_heuristic_speeds():
    test_board = initialize_game_state()
    moves = np.array([3, 3, 2, 4, 0, 1, 0, 3, 0, 0, 2, 3, 3, 2, 2, 2, 0, 6,
                      2, 5, 6, 5, 5, 5, 3, 5])
    player = PLAYER1
    for mv in moves:
        apply_player_action(test_board, mv, player, False)
        player = switch_player(player)

    num_runs = 10000
    # Default version
    run_time = timeit('bit_solver(board, player, True)',
                      number=num_runs,
                      globals=dict(bit_solver=heuristic_solver_bits,
                                   board=test_board,
                                   player=PLAYER1))
    print('\nThe bit solver takes {:0.3} seconds to run'.format(run_time))

    # Use setup argument if adding @njit before function (compile before runtime)
    run_time = timeit('loop_solver(board, player, True)',
                      setup='loop_solver(board, player, True)',
                      number=num_runs,
                      globals=dict(loop_solver=heuristic_solver,
                                   board=test_board,
                                   player=PLAYER1))
    print('The loop solver takes {:0.3} seconds to run'.format(run_time))


def test_alpha_beta_speed():
    pass


# Profile my agent
def test_profile():
    # cProfile.runctx("human_vs_agent(generate_move, generate_move)", None, None, filename='mmab')
    cProfile.run("human_vs_agent(generate_move, generate_move)", "mmab")
    # profile.run('human_vs_agent(generate_move, generate_move)', filename='mmab')


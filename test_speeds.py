import numpy as np
from numba import njit
from timeit import timeit

# Can disable njit
# import os
# os.environ['NUMBA_DISABLE_NJIT'] = '1'

from agents.common import connect_four, initialize_game_state, PLAYER1

board = initialize_game_state()
num_runs = 10e4
# Default version
run_time = timeit('connect_four(board, player)',
                  number=num_runs,
                  globals=dict(connect_four=connect_four,
                               board=board,
                               player=PLAYER1))

# Use setup argument if adding @njit before function (compile before runtime)
run_time = timeit('connect_four(board, player)',
                  setup='connect_four(board, player)',
                  number=num_runs,
                  globals=dict(connect_four=connect_four,
                               board=board,
                               player=PLAYER1))

import cProfile
from main import human_vs_agent
from agents.agent_minimax import generate_move
cProfile.run('human_vs_agent(generate_move, generate_move)', 'output_file_name')
p = pstats.Stats('output_file_name')
p.sort_stats('tottime').print_stats(50)

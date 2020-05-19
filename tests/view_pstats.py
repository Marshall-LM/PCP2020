import pstats

p = pstats.Stats('mmab')
p.sort_stats("tottime").print_stats(50)

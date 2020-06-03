import pstats


n_stats = 12

p = pstats.Stats('mmab')
p.sort_stats("tottime").print_stats(n_stats)

p_bits = pstats.Stats('mmab_bits')
p_bits.sort_stats("tottime").print_stats(n_stats)

p_bits = pstats.Stats('mmab_all_bits')
p_bits.sort_stats("tottime").print_stats(n_stats)

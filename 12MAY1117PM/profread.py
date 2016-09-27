import pstats
p = pstats.Stats('profile.txt')
p.sort_stats('cumulative').print_stats(10)
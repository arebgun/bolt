#!/usr/bin/python
import shelve
from matplotlib import pyplot as plt
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    args = parser.parse_args()

    f = shelve.open(args.filename)
    golden_log_probs = f['golden_log_probs']
    avg_golden_log_probs = f['avg_golden_log_probs']
    golden_entropies = f['golden_entropies']
    avg_golden_entropies = f['avg_golden_entropies']
    golden_ranks = f['golden_ranks']
    avg_golden_ranks = f['avg_golden_ranks']
    min_dists = f['min_dists']
    avg_min = f['avg_min']
    cheating = f['cheating']
    explicit_pointing = f['explicit_pointing']
    ambiguous_pointing = f['ambiguous_pointing']
    f.close()

    if cheating:
        title = 'Cheating (Telepathy)'
    if explicit_pointing:
        title = 'Explicit Pointing\n(Telepath Landmark only)'
    if ambiguous_pointing:
        title = 'Ambiguous Pointing'
    plt.plot(avg_min, 'o-', color='RoyalBlue')
    # plt.plot(max_mins, 'x-', color='Orange')
    plt.ylabel('Edit Distance')
    plt.title(title)
    plt.show()
    plt.draw()

    plt.figure()
    plt.suptitle(title)
    plt.subplot(211)
    plt.plot(golden_log_probs, 'o-', color='RoyalBlue')
    plt.plot(avg_golden_log_probs, 'x-', color='Orange')
    plt.ylabel('Golden Probability')

    plt.subplot(212)
    plt.plot(golden_ranks, 'o-', color='RoyalBlue')
    plt.plot(avg_golden_ranks, 'x-', color='Orange')
    plt.ylim([0,max(avg_golden_ranks)+10])
    plt.ylabel('Golden Rank')
    plt.ioff()
    plt.show()
    plt.draw()

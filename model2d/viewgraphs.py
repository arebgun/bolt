#!/usr/bin/python
import shelve
from matplotlib import pyplot as plt
import argparse
import numpy as np
import sys
sys.path.append("..")
from semantics import relation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-w', '--window-size', type=int, default=20)
    parser.add_argument('--post', action='store_true')
    args = parser.parse_args()

    f = shelve.open(args.filename)

    lmk_priors        = f['lmk_priors']
    rel_priors        = f['rel_priors']
    lmk_posts         = f['lmk_posts']
    rel_posts         = f['rel_posts']
    golden_probs      = f['golden_log_probs']
    golden_entropies  = f['golden_entropies']
    golden_ranks      = f['golden_ranks']
    rel_types         = f['rel_types']
    student_probs     = f['student_probs']    
    student_entropies = f['student_entropies']
    student_ranks     = f['student_ranks']    
    student_rel_types = f['student_rel_types']
    total_mass        = f['total_mass']
    min_dists         = f['min_dists']

    if args.post:
        golden_probs = np.array(lmk_posts)*np.array(rel_posts)

    golden_probs_tofrom = np.ma.masked_array(golden_probs, 
        mask=np.invert(np.equal(rel_types, relation.ToRelation) + np.equal(rel_types,relation.FromRelation)))

    golden_probs_nextto = np.ma.masked_array(golden_probs, mask=np.invert(np.equal(rel_types,relation.NextToRelation)))

    golden_probs_lrfb = np.ma.masked_array(golden_probs, 
        mask=np.invert(np.equal(rel_types, relation.LeftRelation) + np.equal(rel_types,relation.RightRelation) +
              np.equal(rel_types, relation.InFrontRelation) + np.equal(rel_types,relation.BehindRelation)) )

    golden_probs_on = np.ma.masked_array(golden_probs, mask=np.invert(np.equal(rel_types,relation.OnRelation)))

    window = args.window_size
    def running_avg(arr):
        return  [None] + [np.mean(arr[:i]) for i in range(1,window)] + [np.mean(arr[i-window:i]) for i in range(window,len(arr))]

    avg_lmk_priors        = running_avg(lmk_priors)
    avg_rel_priors        = running_avg(rel_priors)
    avg_lmk_posts         = running_avg(lmk_posts)
    avg_rel_posts         = running_avg(rel_posts)
    avg_golden_probs      = running_avg(golden_probs)
    avg_golden_entropies  = running_avg(golden_entropies)
    avg_golden_ranks      = running_avg(golden_ranks)
    avg_student_probs     = running_avg(student_probs)
    avg_student_entropies = running_avg(student_entropies)
    avg_student_ranks     = running_avg(student_ranks)
    avg_min               = running_avg(min_dists)
    
    avg_tofrom            = running_avg(golden_probs_tofrom)
    avg_nextto            = running_avg(golden_probs_nextto)
    avg_lrfb              = running_avg(golden_probs_lrfb)
    avg_on                = running_avg(golden_probs_on)

    # initial_training = f['initial_training']
    cheating = f['cheating']
    explicit_pointing = f['explicit_pointing']
    ambiguous_pointing = f['ambiguous_pointing']

    f.close()

    
    title = ''
    # if initial_training:
    #     title = 'Initial Training'
    if cheating:
        title = 'Cheating (Telepathy)'
    if explicit_pointing:
        title = 'Explicit Pointing\n(Telepath Landmark only)'
    if ambiguous_pointing:
        title = 'Ambiguous Pointing'
    plt.ion()

    plt.plot(total_mass, 'o-', color='RoyalBlue')
    plt.ylabel('Total counts in db')
    plt.title(title)
    plt.show()
    plt.draw()

    plt.figure()
    plt.suptitle(title)
    plt.subplot(211)
    # plt.plot(golden_probs, 'o-', color='RoyalBlue')
    plt.plot(avg_tofrom, 'x-', color='Blue', label='To/From')
    plt.plot(avg_nextto, 'x-', color='Red', label='NextTo')
    plt.plot(avg_lrfb, 'x-', color='Green', label='L/R/F/B')
    plt.plot(avg_on, 'x-', color='Salmon', label='On')
    plt.plot(avg_golden_probs, 'x-', color='Black', label='All')
    plt.ylim((0,1))
    plt.ylabel('Golden Probability')
    leg = plt.legend()
    leg.draggable(state=True)

    plt.subplot(212)
    plt.plot(golden_ranks, 'o-', color='RoyalBlue')
    plt.plot(avg_golden_ranks, 'x-', color='Orange')
    plt.ylim([0,max(avg_golden_ranks)+10])
    plt.ylabel('Golden Rank')

    plt.figure()
    plt.suptitle(title)
    plt.subplot(211)
    plt.plot(student_probs, 'o-', color='RoyalBlue')
    plt.plot(avg_student_probs, 'x-', color='Orange')
    plt.ylim((0,1))
    plt.ylabel('Student Probability')

    plt.subplot(212)
    plt.plot(student_ranks, 'o-', color='RoyalBlue')
    plt.plot(avg_student_ranks, 'x-', color='Orange')
    plt.ylim([0,max(avg_student_ranks)+10])
    plt.ylabel('Student Rank')

    plt.figure()
    plt.title(title)
    
    # plt.plot(lmk_priors, 'o-', color='RoyalBlue', label='lmk_priors')
    # plt.plot(rel_priors, 'o-', color='Orange', label='rel_priors')
    # plt.plot(lmk_posts, 'o-', color='MediumSpringGreen', label='lmk_posts')
    # plt.plot(rel_posts, 'o-', color='Salmon', label='rel_posts')

    plt.plot(avg_lmk_priors, '-', linewidth=3, color='DarkBlue', label='avg_lmk_priors')
    plt.plot(avg_rel_priors, '-', linewidth=3, color='DarkOrange', label='avg_rel_priors')
    plt.plot(avg_lmk_posts, '-', linewidth=3, color='MediumSeaGreen', label='avg_lmk_posts')
    plt.plot(avg_rel_posts, '-', linewidth=3, color='DarkRed', label='avg_rel_posts')
    plt.ylim([0,1])
    leg = plt.legend()
    leg.draggable(state=True)

    plt.ioff()
    plt.show()
    plt.draw()

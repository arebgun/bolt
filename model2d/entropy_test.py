#!/usr/bin/env python
# coding: utf-8

from models import CProduction
import shelve


if __name__ == '__main__':

    golden = False
    verbose = False
    group_by_parent = True
    lhs = 'E52'
    rhs = 'black'

    columns = [
        CProduction.landmark_class,
        CProduction.landmark_orientation_relations,
        CProduction.landmark_color,
        CProduction.relation,
        CProduction.relation_distance_class,
        CProduction.relation_degree_class
    ]

    sorted_map = {}

    print 'Golden: %s, Group By Parent: %s' % (golden, group_by_parent)

    # get all top level grammar elements, e.g. only lhs
    # unique_prods = CProduction.get_unique_productions(group_by_rhs=False, group_by_parent=group_by_parent, golden=golden)
    # print 'Found %d unique top level productions in the database' % len(unique_prods)

    # for prod in unique_prods:
    #     print '%s [%s]' % (prod.lhs, prod.parent)
    #     ratios = []

    #     for col in columns:
    #         # ratio = CProduction.get_entropy_ratio(lhs=prod.lhs, rhs=None, column=col, parent=prod.parent, golden=golden, verbose=verbose)
    #         ratio = CProduction.get_entropy_ratio_sample_dependent(lhs=prod.lhs, rhs=None, column=col, parent=prod.parent, golden=golden, verbose=verbose)
    #         ratios.append( (ratio, col) )
    #         if str(col) not in sorted_map: sorted_map[str(col)] = []
    #         sorted_map[str(col)].append( (ratio, prod.lhs, None, prod.parent) )

    #     for ratio, col in sorted(ratios):
    #         print '\t%f --- %s' % (ratio, col)

    #     print '\n\n'

    # get all unique productions of the form lhs -> rhs
    unique_prods = CProduction.get_unique_productions(group_by_rhs=True, group_by_parent=group_by_parent, golden=golden)

    print 'Found %d unique productions in the database' % len(unique_prods)

    for prod in unique_prods:
        print '%s -> %s [%s]' % (prod.lhs, prod.rhs, prod.parent)
        ratios = []

        for col in columns:
            # ratio = CProduction.get_entropy_ratio(lhs=prod.lhs, rhs=prod.rhs, column=col, golden=golden, verbose=verbose)
            ratio = CProduction.get_entropy_ratio_sample_dependent(lhs=prod.lhs, rhs=prod.rhs, column=col, golden=golden, verbose=verbose)
            ratios.append( (ratio, col) )
            if str(col) not in sorted_map: sorted_map[str(col)] = []
            sorted_map[str(col)].append( (ratio, prod.lhs, prod.rhs, prod.parent) )

        for ratio, col in sorted(ratios):
            print '\t%f --- %s' % (ratio, col)

        print '\n\n'

    for c in sorted_map:
        sorted_map[str(c)] = sorted(sorted_map[str(c)])

    f = shelve.open('entropies_%s_%s.shelf' % ( ('golden' if golden else 'trained'), ('grouped' if group_by_parent else 'ungrouped') ))
    f['entropies'] = sorted_map
    f.close()

    for k in sorted_map:
        print k
        for ratio,lhs,rhs,parent in sorted_map[k]:
            print '\t[%f] %s -> %s [%s]' % (ratio,lhs,rhs,parent)

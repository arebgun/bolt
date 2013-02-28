#!/usr/bin/env python
# coding: utf-8

from models import CProduction


if __name__ == '__main__':

    golden = True
    verbose = True
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

    print 'Golden: %s, Group By Parent: %s' % (golden, group_by_parent)

    # get all top level grammar elements, e.g. only lhs
    unique_prods = CProduction.get_unique_productions(group_by_rhs=False, group_by_parent=group_by_parent, golden=golden)
    print 'Found %d unique top level productions in the database' % len(unique_prods)

    for prod in unique_prods:
        print '%s [%s]' % (prod.lhs, prod.parent)
        ratios = []

        for col in columns:
            ratios.append( (CProduction.get_entropy_ratio(lhs=prod.lhs, rhs=None, column=col, golden=golden, verbose=verbose), col) )

        for ratio, col in sorted(ratios):
            print '\t%f --- %s' % (ratio, col)

        print '\n\n'

    # get all unique productions of the form lhs -> rhs
    unique_prods = CProduction.get_unique_productions(group_by_rhs=True, group_by_parent=group_by_parent, golden=golden)

    print 'Found %d unique productions in the database' % len(unique_prods)

    for prod in unique_prods:
        print '%s -> %s [%s]' % (prod.lhs, prod.rhs, prod.parent)
        ratios = []

        for col in columns:
            ratios.append( (CProduction.get_entropy_ratio(lhs=prod.lhs, rhs=prod.rhs, column=col, golden=golden, verbose=verbose), col) )

        for ratio, col in sorted(ratios):
            print '\t%f --- %s' % (ratio, col)

        print '\n\n'

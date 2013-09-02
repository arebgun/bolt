#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import numpy as np
# from nltk.tree import ParentedTree
from lmk_rel_observations import LandmarkObservation
import pprint
pp = pprint.PrettyPrinter(indent=4)

def get_landmark_prob(session, utterance, speaker, landmark, 
                      default_prob=0.001, default_ent=1000, printing=True):


    sem_feats=LandmarkObservation.extract_semantic_features(speaker, landmark)
    ling_feats = LandmarkObservation\
                    .extract_linguistic_features(session, utterance)
    variable_part_name = LandmarkObservation.variable_part_name
    # pp.pprint(ling_feats)

    q = session.query(LandmarkObservation)
    for feat_name, feat in sem_feats.items():
        q = q.filter(getattr(LandmarkObservation,feat_name) == feat)

    ling_feat_proportions = []
    for feat in ling_feats:
        qfeat = q
        for part_name, part in feat.items():
            if part_name != variable_part_name:
                # print 'Filtering on',part_name,'==',part
                qfeat = qfeat.filter(
                    getattr(LandmarkObservation,part_name) == part)

        # print feat,qfeat.count()
        if qfeat.count() == 0:
            this_feat_proportion = 0.0
        else:
            counter = dict()
            for row in qfeat.all():
                # print '  ',getattr(row,variable_part_name)
                variable = getattr(row,variable_part_name)
                if variable in counter:
                    counter[getattr(row,variable_part_name)] += row.landmark_prior
                else:
                    counter[getattr(row,variable_part_name)] = row.landmark_prior+1
            if feat[variable_part_name] not in counter:
                counter[feat[variable_part_name]] = 1

            keys, counts = zip(*counter.items())
            counts = np.array(counts, dtype=float)
            # props = ccounts/ccounts.sum()
            this_feat_proportion = counter[feat[variable_part_name]]/counts.sum()
            # print "Proportion =", this_feat_proportion
        ling_feat_proportions.append(this_feat_proportion)

    return np.prod(ling_feat_proportions)
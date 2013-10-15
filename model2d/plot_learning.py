#!/usr/bin/env python
# coding: utf-8

from __future__ import division

import utils
from utils import (get_meaning,
                   rel_type,
                   m2s,
                   get_lmk_ori_rels_str,
                   get_landmark_parent_chain,
                   logger)

from models import CProduction

from semantics.run import construct_training_scene

from semantics.representation import PointRepresentation
from semantics.landmark import Landmark
from semantics.relation import Degree, Measurement

from myrandom import random
random = random.random
from planar import Vec2
import shelve

from matplotlib.pyplot import figure, show
import numpy as npy
import scipy as scy
from random import choice
from collections import Counter


columns = [
    CProduction.landmark_class,
    CProduction.landmark_orientation_relations,
    CProduction.landmark_color,
    CProduction.relation,
    CProduction.relation_distance_class,
    CProduction.relation_degree_class
]

def running_avg(arr, window=15):
    return [npy.mean(arr[i:i+window]) for i in range(len(arr)-window)]

if __name__ == '__main__':
    f = shelve.open('learning_data.shelf')
    scores = f['scores']
    raw_scores = f['raw_scores']
    teach_raw_scores = f['teach_raw_scores']
    raw_scores_by_meaning = f['raw_scores_by_meaning']
    f.close()

    scores_by_col = {}
    teach_scores_by_col = {}
    batch = 20

    for s,t in zip(raw_scores,teach_raw_scores):
        for col in s:
            score = s[col]
            try:
                scores_by_col[col].append(score)
            except:
                scores_by_col[col] = [score]

        for col in t:
            score = t[col]
            try:
                teach_scores_by_col[col].append(score)
            except:
                teach_scores_by_col[col] = [score]

    for col,tcol in zip(scores_by_col, teach_scores_by_col):
        col_scores = scores_by_col[col]
        col_batch_avg = []
        col_batch_std = []
        col_batch_quant = []
        print col, len(col_scores)

        teach_col_scores = teach_scores_by_col[tcol]
        teach_col_batch_avg = []
        teach_col_batch_std = []
        teach_col_batch_quant = []
        print tcol, len(teach_col_scores)

        for i in range(0, len(col_scores), batch):
            j = i + batch
            print i, j#, col, col_scores[i:j]
            avg = sum(col_scores[i:j])/float(len(col_scores))
            col_batch_avg.append(avg)
            stds = [(x - avg)**2 for x in col_scores[i:j]]
            col_batch_std.append(sum(stds) / len(stds))
            col_batch_quant.append( scy.stats.mstats.mquantiles(col_scores[i:j], 0.25)[0] )

            tavg = sum(teach_col_scores[i:j])/float(len(teach_col_scores))
            teach_col_batch_avg.append(tavg)
            tstds = [(x - tavg)**2 for x in teach_col_scores[i:j]]
            teach_col_batch_std.append(sum(tstds) / len(tstds))
            teach_col_batch_quant.append( scy.stats.mstats.mquantiles(teach_col_scores[i:j], 0.25)[0] )

        fig = figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(col_batch_avg)
        ax1.plot(teach_col_batch_avg)
        fig.savefig(col + '_batch_avg.png')

        fig = figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(col_batch_std)
        ax1.plot(teach_col_batch_std)
        fig.savefig(col + '_stds.png')

        fig = figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(col_batch_quant)
        ax1.plot(teach_col_batch_quant)
        fig.savefig(col + '_quant.png')

        window = 100
        fig = figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(running_avg(col_scores, window))
        ax1.plot(running_avg(teach_col_scores, window))
        fig.savefig(col + '_runavg.png')
        # show()

    # scores_by_m = {}

    # for m in raw_scores_by_meaning:
    #     for s in raw_scores_by_meaning[m]:
    #         for col in s:
    #             score = s[col]
    #             try:
    #                 scores_by_m[m+col].append(score)
    #             except:
    #                 scores_by_m[m+col] = [score]

    # for m in scores_by_m:
    #     m_scores = scores_by_m[m]
    #     m_batch_avg = []
    #     print m, len(m_scores)

    #     for i in range(batch, len(m_scores), batch):
    #         m_batch_avg.append(sum(m_scores[:i])/float(len(m_scores)))

    #     fig = figure()
    #     ax1 = fig.add_subplot(111)
    #     ax1.plot(m_batch_avg)
    #     fig.savefig(m + '1.png')

#!/usr/bin/python

import sys
sys.path.insert(1,'..')
from matplotlib import pyplot as plt
import gen2_features as g2f
import collections as coll
import sqlalchemy as alc
import numpy as np
import IPython

db_name = '2rels1.db'

engine = alc.create_engine('sqlite:///'+db_name, echo=False)
meta = alc.MetaData()
meta.reflect(bind=engine)
scenes = meta.tables['scenes']
unknown_structs = meta.tables['unknown_structs']
connection = engine.connect()


word = 'blue'
query = alc.sql.select([unknown_structs]).where(
        unknown_structs.c.string.like('%%%s%%' % word))

results = list(connection.execute(query))

if len(results) > 0:
    separated = zip(*results)
    # IPython.embed()
    # exit()
    classes = np.array(separated[5])
    probs = np.array(separated[6])
    feature_values = np.array(separated[7:])
    print classes.shape, probs.shape, feature_values.shape
    # IPython.embed()
    for feature, values in zip(g2f.feature_list, feature_values):
        count = coll.Counter(values)
        count1 = coll.Counter(values[np.where(classes)])
        keys, values = zip(*count.items())
        values1 = [count1[key] for key in keys]
        if len(keys) > 1:
            fig2, (ax1, ax2) = plt.subplots(2)
            width = 1
            ind = np.array(range(len(keys)))
            ax1.bar(ind, values)
            ax1.set_xticks(ind+width*0.5)
            ax1.set_xticklabels( keys )
            ax2.bar(ind, values1)
            ax2.set_xticks(ind+width*0.5)
            ax2.set_xticklabels( keys )
            plt.show()
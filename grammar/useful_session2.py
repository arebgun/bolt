#!/usr/bin/python
# coding: utf-8
import sys
sys.path.insert(1,'..')
import language_user as lu
import sqlalchemy as alc
import numpy as np
import probability_function as pf
import gen2_features as g2f
import domain as dom
import IPython
import itertools as it
import skipgram as sg

strings = ['on', 'near to', 'far from', 'to the right of', 'to the left of', 
           'to the front of', 'to the back of']
# strings = ['to the back of','to the front of','to the right of', 'to the left of']
# colors = ['red', 'green', 'blue']
# nouns = ['block', 'sphere']
# strings = [' '.join([a,b]) for a,b in it.product(colors, nouns)]
student = lu.LanguageUser('Student',None,None,None)
student.connect_to_memories()

def classification_cost(probs, classes, weights):
    new_classes = np.array(probs) > 0.5
    a = new_classes==classes
    return a.sum()/float(len(a))

def powerset(iterable):
    xs = list(iterable)
    # note we return an iterator rather than a list
    return it.chain.from_iterable( it.combinations(xs,n) for n in range(2,len(xs)+1) )

semtups = {}
for s in strings:
    print s
    sem, _ = student.cv_build('Relation', s)
    print sem
    # semtups[s] = set(sem.tuples())
print

# maxlen = 4
# k = maxlen
# substrings = {}
# for string in strings:
#     phrase = string.split()
#     # print phrase
#     subs = []
#     for n in range(1,maxlen+1):
#         skipgrams = sg.kskipngrams(phrase,k,n)
#         if skipgrams:
#             subs.extend(skipgrams)
#     # subs = map(tuple, subs)
#     # subs = set(subs)
#     substrings[string] = subs

# # for key, val in substrings.items():
# #     print key
# #     print val
# #     print
# # exit()

# commons = []
# for stringset in powerset(strings):
#     str1 = stringset[0]
#     sgs1 = substrings[str1]
#     commonsem = semtups[str1]
#     for str2 in stringset[1:]:
#         commonsem = commonsem.intersection(semtups[str2])
#         sgs2 = substrings[str2]
#         common = []
#         for sg1 in sgs1:
#             if sg1 in sgs2:
#                 sg2 = sgs2[sgs2.index(sg1)]
#                 common.append(sg.SkipGram(sg1.gram,sg1.skippedlist+sg2.skippedlist))
#         sgs1 = common
#     common = sgs1
#     # common = list(common)
#     common.sort(reverse=True, key=lambda x: len(x))
#     if len(common) > 0:
#         longest = common[0]
#     else:
#         longest = ()
#     if len(commonsem) > 0:
#         new_commonsem = []
#         for i, t1 in enumerate(commonsem):
#             add = True
#             for t2 in commonsem:
#                 if t1[0] == t2[0] and len(t1) < len(t2):
#                     add = False
#             if add:
#                 new_commonsem.append(t1)
#         commonsem = new_commonsem
#     commons.append((stringset,longest,commonsem))

# # for stringset, longest in commons:
# #     print stringset
# #     print longest
# #     print
# # exit()

# new_commons = []
# for i, (stringset1, longest1, csem1) in enumerate(commons):
#     stringset1 = set(stringset1)
#     add = True
#     for stringset2, longest2, csem2 in commons[i+1:]:
#         if longest1 == longest2 and \
#            stringset1.issubset(set(stringset2)):
#            add = False
#            break
#     if add:
#         new_commons.append((stringset1, longest1, csem1))
# commons = new_commons


# commons.sort(reverse=True, key=lambda x: (len(x[1]),len(x[0])))
# for common in commons[:10]:
#     print 'Generalizable:   ', list(common[0])
#     print 'Common syntax:   ', common[1]
#     print 'Common semantics:', common[2]
#     print 


    # continue
    # query = alc.sql.select([student.unknown_structs]).where(
    # 	student.unknown_structs.c.construction=='Relation').where(
    # 	student.unknown_structs.c.string==s)
    # results = list(student.connection.execute(query))
    # print len(results)
    # separated = zip(*results)
    # classes = np.array(separated[5])
    # weights = np.array(separated[6])
    # feature_values = separated[7:]
    # # contains = np.array(feature_values[-3])==0

    # for feature, values in zip(g2f.feature_list, feature_values):
    #     values = np.array(values)
    #     # print '%s positive Values: %s\n' %(feature,values[np.where(classes==True)])
    #     # print '%s negative Values: %s\n' %(feature,values[np.where(classes==False)])
    #     if isinstance(feature.domain,dom.DiscreteDomain):
    #         values = np.array(['' if v is None else v for v in values])
    #         pfunc, result1=pf.DiscreteProbFunc.build_binary(values,
    #                                                classes,weights,
    #                                                student.binomial_cost)
    #         # result += result1
    #         if pfunc:
    #             cost = student.binomial_cost(pfunc(values),
    #                                          classes,
    #                                          weights)
    #             print '  ',cost,'DPF',feature
    #             # result += '%s: %s, %s\n' % (feature,
    #             #                             cost,
    #             #                             pfunc)
    #             # constraints.append((cost,pfunc,feature))
    #     else:
    #         if len(filter(None,values)) == 0:
    #             break
    #         values = np.array([float('nan') if v is None else v for v in values])
    #         w = np.where(np.logical_not(np.isnan(values)))
    #         classes_ = classes[w]
    #         values_ = values[w]
    #         weights_ = weights[w]
    #         pfunc=pf.LogisticBell.build(feature.domain,
    #                                     values_,
    #                                     classes_,
    #                                     weights_)
    #         cost = student.binomial_cost(pfunc(values),
    #                                   classes,
    #                                   weights)
    #         print '  ',cost,'LB',feature, pfunc
    #         # result += '%s: %s, %s\n' % (feature,
    #         #                             cost,
    #         #                             pfunc)
    #         # constraints.append((cost,pfunc,feature))

    #         pfunc=pf.LogisticSigmoid.build(feature.domain,
    #                                        values_,
    #                                        classes_,
    #                                        weights_)
    #         cost = student.binomial_cost(pfunc(values),
    #                                   classes,
    #                                   weights)
    #         print '  ',cost,'LS',feature, pfunc
    #         # result += '%s: %s, %s\n' % (feature,
    #         #                             cost,
    #         #                             pfunc)
    #         # constraints.append((cost,pfunc,feature))
    # print
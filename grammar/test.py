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

strings = ['on','near to', 'far from', 'to the right of', 'to the left of', 'to the front of', 'to the back of']
# strings = ['on']
student = lu.LanguageUser('Student',None,None,None)
student.connect_to_memories()

def classification_cost(probs, classes, weights):
    new_classes = np.array(probs) > 0.5
    a = new_classes==classes
    return a.sum()/float(len(a))

for s in strings:
    print student.cv_build('Relation', s)
    continue
    print s
    query = alc.sql.select([student.unknown_structs]).where(
        student.unknown_structs.c.construction=='Relation').where(
        student.unknown_structs.c.string==s)
    results = list(student.connection.execute(query))
    print len(results)
    separated = zip(*results)
    classes = np.array(separated[5])
    weights = np.array(separated[6])
    feature_values = np.array(separated[7:]).T
    IPython.embed()
    # contains = np.array(feature_values[-3])==0

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
    #         # values = np.array([float('nan') if v is None else v for v in values])
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
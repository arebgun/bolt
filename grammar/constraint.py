import sys
sys.path.insert(1,'..')
import utils
import numpy as np
import collections as coll
from common import Applicabilities

class ApplicabilityRegister:
    scene = None
    apps = dict()

    @staticmethod
    def reset(scene):
        ApplicabilityRegister.scene = scene
        ApplicabilityRegister.apps = dict()

class Constraint(object):

    def __init__(self, feature, prob_func):
        self.feature = feature
        self.probability_func = prob_func

    def __key(self):
        return (self.__class__.__name__, self.feature, self.probability_func)

    def __repr__(self):
        return '%s(feature=%s,prob_func=%s)' % self.__key()

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__key()==other.__key()

    def __hash__(self):
        return hash(self.__key())

    @property
    def domain(self):
        return self.feature.domain

    def ref_applicability(self, potential_referent, **kwargs):
        return self.probability_func(self.feature.observe(potential_referent))


    # def applicabilities(self, potential_referents, **kwargs):
    #     return dict([(ref, self.applicability(ref, **kwargs)) 
    #                  for ref in potential_referents])



# class GroupConstraint(Constraint):
#     '''Constraint on groups of entities'''

# class MetaConstraint(Constraint):
#     pass

class PropertyConstraint(Constraint):
    '''Constraint on individual entities'''

    def ref_applicabilities(self, context, potential_referents, **kwargs):
        # if ApplicabilityRegister.scene == context.scene:
        #     if self in ApplicabilityRegister.apps:
        #         # logger('Recalling apps')
        #         return ApplicabilityRegister.apps[self]
        # else:
        #     logger('Resetting ApplicabilityRegister')
        #     ApplicabilityRegister.reset(context.scene)

        apps = Applicabilities([(ref, self.ref_applicability(ref)) 
                                for ref in potential_referents])
        # ApplicabilityRegister.apps[self] = apps
        return apps

    def ref_applicability(self, potential_referent, **kwargs):
        probs = [self.quantity_applicability(self.feature.observe(entity)) 
                 for entity in potential_referent]
        return np.product(probs)

    def quantity_applicability(self, quantity):
        return self.probability_func(quantity)


class Degree(object):
    def __init__(self, degree):
        self.degree = degree

    def modify(self, gradable):
        gradable = gradable.copy()
        gradable.degree = self.degree
        return gradable

class ComparativeProperty(PropertyConstraint):
    def replace(self, other):
        raise NotImplementedError



class RelationConstraint(Constraint):
    '''Constraint on pairs of entities'''

    def ref_applicabilities(self, context, potential_referents, 
                               relata_apps, **kwargs):
        assert(relata_apps is not None)
        apps = Applicabilities([(ref, self.ref_applicability(context, ref, 
                                                             relata_apps,
                                                             **kwargs))
                                for ref in potential_referents])
        return apps

    def ref_applicability(self, context, potential_referent, 
                            relata_apps, **kwargs):
        probs = [self.entity_applicability(context, entity, relata_apps, 
                                           **kwargs)
                 for entity in potential_referent]
        return np.product(probs)

    def entity_applicability(self, context, entity, relata_apps, **kwargs):
        # entity_app = 0
        # relatum_app_sum = sum(relata_apps.values())
        ps = []
        for relatum, relatum_app in relata_apps.items():
            p = np.product(
                [self.probability_func(self.feature.observe(
                    entity, 
                    relentity,
                    viewpoint=context.speaker.get_head_on_viewpoint(relentity),
                    **kwargs))
                 for relentity in relatum])

            # entity_app += p*relatum_app/relatum_app_sum
            ps.append(p*relatum_app)
        ps = np.nan_to_num(ps)
        pssum = ps.sum()
        if pssum == 0:
            return 0
        else:
            ps = ps**2/pssum
            # return entity_app
            return max(ps)


class ConstraintCollection(coll.OrderedDict):
    def __init__(self, constraints):
        if isinstance(constraints, ConstraintCollection):
            super(ConstraintCollection, self).__init__(constraints)
            self.relatum_constraints = constraints.relatum_constraints
        else:
            pairs = [(c.domain,c) for c in constraints]
            super(ConstraintCollection, self).__init__(pairs)
            self.relatum_constraints = None

    def __key(self):
        return (self.__class__.__name__, self.relatum_constraints)

    def __repr__(self):
        return self.__class__.__name__+'(constraints=['+\
               ', '.join(c.__repr__() for c in self.values())+\
               '],relatum_constraints='+self.relatum_constraints.__repr__()

    def __hash__(self):
        hsh = hash(self.__key())
        for c in self.values():
            hsh ^= hash(c)
        return hsh

    def modify(self, other):
        other = other.copy()
        for space in self:
            if space in other:
                other[space] = self[space].replace(other[space])
            else:
                other[space] = self[space]
        return other

    def ref_applicabilities(self, context, potential_referents, **kwargs):
        # if ApplicabilityRegister.scene == context.scene:
        #     if self in ApplicabilityRegister.apps:
        #         # logger('Recalling apps')
        #         return ApplicabilityRegister.apps[self]
        # else:
        #     logger('Resetting ApplicabilityRegister')
        #     ApplicabilityRegister.reset(context.scene)

        if self.relatum_constraints is not None:
            # potential recursion here
            relata_apps = context.get_potential_referent_scores()
            relata_apps *= self.relatum_constraints.ref_applicabilities(context, 
                            relata_apps.keys())
            # viewpoint = context.speaker.location
        else:
            relata_apps = None
            # viewpoint = None
            for constraint in self.values():
                if isinstance(constraint, RelationConstraint):
                    raise Exception('What')

        apps = Applicabilities([(ref,1.0) for ref in potential_referents])
        for constraint in self.values():
            ref_apps = constraint.ref_applicabilities(
                        context=context, # <- Ignored if irrelevant
                        potential_referents=potential_referents,
                        relata_apps=relata_apps)
                        # viewpoint=viewpoint) # <- Ignored if irrelevant
            # utils.logger(constraint)
            # utils.logger(apps)
            # utils.logger(ref_apps)
            apps *= ref_apps

        # ApplicabilityRegister.apps[self] = apps
        return apps

    def quantity_applicability(self, context, quantity):
        app = 1
        for constraint in self.values():
            app *= constraint.quantity_applicability(context=context,
                                                     quantity=quantity)
        return app
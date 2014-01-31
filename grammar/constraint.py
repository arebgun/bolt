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
        #         utils.logger('Recalling apps')
        #         return ApplicabilityRegister.apps[self]
        # else:
        #     utils.logger('Resetting ApplicabilityRegister')
        #     ApplicabilityRegister.reset(context.scene)
        # utils.logger('Calculating')

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
        # utils.logger('Calculating (relation)')
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
                    context=context,
                    referent=entity, 
                    relatum=relentity,
                    **kwargs))
                 for relentity in relatum])
            # utils.logger(self.probability_func)
            # utils.logger(self.feature)
            # utils.logger(relatum)
            # utils.logger(relatum_app)
            # utils.logger(p)
            # utils.logger(p*relatum_app)
            # entity_app += p*relatum_app/relatum_app_sum
            ps.append(p*relatum_app)
        ps = np.nan_to_num(ps)
        return max(ps)
        # pssum = ps.sum()
        # if pssum == 0:
        #     return 0
        # else:
        #     ps = ps**2/pssum
        #     # return entity_app
        #     return max(ps)


class ConstraintSet(coll.MutableMapping):
    def __init__(self, constraints=[]):
        pairs = [(c.domain.name,c) for c in constraints]
        self.odict = coll.OrderedDict(pairs)
        # if isinstance(constraints, ConstraintSet):
        #     super(ConstraintSet, self).__init__(constraints)
        #     self.relatum_constraints = constraints.relatum_constraints
        # else:
        #     super(ConstraintSet, self).__init__(pairs)
        self.relatum_constraints = None

    def copy(self):
        copy = object.__new__(self.__class__)
        copy.odict = coll.OrderedDict(self.items())
        copy.relatum_constraints = self.relatum_constraints
        return copy

    def __key(self):
        return (self.__class__.__name__, self.relatum_constraints)

    def __repr__(self):
        return self.__class__.__name__+'(constraints=['+\
               ',\n '.join(c.__repr__() for c in self.values())+\
               '],relatum_constraints='+self.relatum_constraints.__repr__()+')'

    def __hash__(self):
        hsh = hash(self.__key())
        for c in self.values():
            hsh ^= hash(c)
        return hsh

    def __getitem__(self, key):
        return self.odict[key]

    def __setitem__(self, key, value):
        self.odict[key] = value

    def __delitem__(self, key):
        del self.odict[key]

    def __iter__(self):
        return iter(self.odict)

    def __len__(self):
        return len(self.odict)

    def values(self):
        return self.odict.values()

    def keys(self):
        return self.odict.keys()

    def items(self):
        return self.odict.items()

    def modify(self, other):
        other = other.copy()
        for space in self:
            # utils.logger(space)
            if space in other:
                other[space+'2'] = self[space]#.replace(other[space])
            else:
                other[space] = self[space]
        assert(self.relatum_constraints==None or 
               other.relatum_constraints==None)
        if self.relatum_constraints != None:
            other.relatum_constraints = self.relatum_constraints
        return other

    def ref_applicabilities(self, context, potential_referents, **kwargs):
        # if ApplicabilityRegister.scene == context.scene:
        #     if self in ApplicabilityRegister.apps:
        #         utils.logger('Recalling apps')
        #         return ApplicabilityRegister.apps[self]
        # else:
        #     utils.logger('Resetting ApplicabilityRegister')
        #     ApplicabilityRegister.reset(context.scene)
        # utils.logger('Calculating')

        if self.relatum_constraints is not None:
            # utils.logger(self.relatum_constraints)
            # potential recursion here
            relata_apps = context.get_all_potential_referent_scores()
            relata_apps *= self.relatum_constraints.ref_applicabilities(context, 
                            relata_apps.keys())
            relata, apps = zip(*relata_apps.items())
            apps = np.array(apps)
            appsum = apps.sum()
            if appsum == 0:
                apps[:] = 0
            else:
                apps = (apps**2)/float(appsum)
            relata_apps = Applicabilities(zip(relata,apps))
            # for item, i in relata_apps.items():
            #     utils.logger('%s %s'%(i,item))
            # viewpoint = context.speaker.location
        else:
            relata_apps = None
            # viewpoint = None
            for constraint in self.values():
                # utils.logger(constraint)
                if isinstance(constraint, RelationConstraint):
                    raise Exception('What')

        # utils.logger(self)
        apps = Applicabilities([(ref,1.0) for ref in potential_referents])
        for constraint in self.values():
            ref_apps = constraint.ref_applicabilities(
                        context=context, # <- Ignored if irrelevant
                        potential_referents=potential_referents,
                        relata_apps=relata_apps)
                        # viewpoint=viewpoint) # <- Ignored if irrelevant
            # utils.logger(constraint)
            # utils.logger('                            Apps')
            # for item,i in apps.items():
            #     utils.logger('%s %s'%(i,item))
            # utils.logger('                            Ref_apps')
            # for item,i in ref_apps.items():
            #     utils.logger('%s %s'%(i,item))
            apps *= ref_apps

        # ApplicabilityRegister.apps[self] = apps
        return apps

    def quantity_applicability(self, context, quantity):
        app = 1
        for constraint in self.values():
            app *= constraint.quantity_applicability(context=context,
                                                     quantity=quantity)
        return app
import feature as feat
import domain as dom
import utils

########## Feature extraction functions

# For group constraints
def __group_cardinality(referent, **kwargs):
    assert(isinstance(referent, tuple))
    return len(referent)

def __referent_known(referent, **kwargs):
    return True


# For properties
# 1-part features
def __referent_rep(referent, **kwargs):
    return referent.representation.__class__.__name__

def __relatum_rep(relatum, **kwargs):
    if relatum is None:
        return None
    return relatum.representation.__class__.__name__

def __relatum_color(relatum, **kwargs):
    if relatum is None:
        return None
    return relatum.color

def __relatum_class(relatum, **kwargs):
    if relatum is None:
        return None
    return relatum.object_class

def __referent_color(referent, **kwargs):
    return referent.color

def __referent_class(referent, **kwargs):
    return referent.object_class

def __referent_height(referent, **kwargs):
    raise NotImplementedError

def __referent_width(referent, **kwargs):
    raise NotImplementedError

def __referent_length(referent, **kwargs):
    raise NotImplementedError

def __referent_volume(referent, **kwargs):
    raise NotImplementedError

# 2-part features (for relations)

def __not_equal(referent, relatum, **kwargs):
    if relatum is None:
        return None
    return referent != relatum

def __part_of(referent, relatum, **kwargs):
    if relatum is None:
        return None
    return bool(referent.get_parent_landmark() == relatum)

def __contains(referent, relatum, string=False, **kwargs):
    if relatum is None:
        return None
    c = relatum.contains(referent)
    # utils.logger("%s %s %s" % (c,referent,relatum))
    b = bool(c >= 0.2)#0.8)
    if string:
        b = str(b)
    return b

def __distance_between(referent, relatum, **kwargs):
    if relatum is None:
        return float('nan')
    distance = referent.distance_to(relatum.representation)
    # print referent, relatum, distance
    return distance

def __angle_between(referent, relatum, context, **kwargs):
    if relatum is None:
        return float('nan')
    viewpoint = context.speaker.get_head_on_viewpoint(relatum)
    # viewpoint = speaker.get_headon_viewpoint(referent)
    angle = relatum.angle_between(viewpoint, referent)
    # print 'gen2_features: 56', referent, relatum, angle
    return angle




group_cardinality = feat.Feature(measure_func=__group_cardinality,
                                 domain=dom.NumericalDomain('group_cardinality', int))
referent_known = feat.Feature(measure_func=__referent_known,
                              domain=dom.DiscreteDomain('referent_known', bool))
referent_rep = feat.Feature(measure_func=__referent_rep,
                            domain=dom.DiscreteDomain('referent_rep', str))
referent_color = feat.Feature(measure_func=__referent_color,
                              domain=dom.DiscreteDomain('referent_color', str))
referent_class = feat.Feature(measure_func=__referent_class,
                              domain=dom.DiscreteDomain('referent_class', str))
referent_height = feat.Feature(measure_func=__referent_height,
                               domain=dom.Domain('referent_height', float))
referent_width = feat.Feature(measure_func=__referent_width,
                              domain=dom.Domain('referent_width', float))
referent_length = feat.Feature(measure_func=__referent_length,
                               domain=dom.Domain('referent_length', float))
referent_volume = feat.Feature(measure_func=__referent_volume,
                               domain=dom.Domain('referent_volume', float))
relatum_rep = feat.Feature(measure_func=__relatum_rep,
                           domain=dom.DiscreteDomain('relatum_rep', str))
relatum_color = feat.Feature(measure_func=__relatum_color,
                           domain=dom.DiscreteDomain('relatum_color', str))
relatum_class = feat.Feature(measure_func=__relatum_class,
                           domain=dom.DiscreteDomain('relatum_class', str))
part_of = feat.Feature(measure_func=__part_of,
                       domain=dom.DiscreteDomain('part_of', bool))
contains = feat.Feature(measure_func=__contains,
                        domain=dom.DiscreteDomain('contains', bool))
distance_between = feat.Feature(measure_func=__distance_between,
                        domain=dom.NumericalDomain('distance_between', float))
angle_between = feat.Feature(measure_func=__angle_between,
                             domain=dom.CircularDomain('angle_between',float,
                                                          -180, 180))

feature_list = [
    # group_cardinality,
    referent_known,
    referent_rep,
    referent_color,
    referent_class,
    # relatum_rep,
    # relatum_color,
    # relatum_class, 
    part_of,
    contains,
    distance_between,
    angle_between
]